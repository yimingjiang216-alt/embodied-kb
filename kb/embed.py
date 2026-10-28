#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量层：用 Google gemini-embedding-2（走 OpenAI 兼容端点）给知识库加语义检索。

设计原则：
- 可选。没设环境变量就完全不影响 BM25 单路模式。
- 向量存 kb/index/vectors.npy + embed_meta.json（和 chunks.jsonl 行序对齐）。
- 建库和查询必须用同一个模型——换模型要重建。

用法：
  set KB_EMBED_BASE=https://pool.autelrobotics.com/v1
  set KB_EMBED_KEY=sk-xxx
  python kb/embed.py                    # 给当前索引建向量
  python kb/search.py "..." --semantic  # 混合检索（BM25 + 向量 RRF 融合）
"""
import json
import os
import ssl
import time
import urllib.request
from pathlib import Path

INDEX_DIR = Path(__file__).resolve().parent / "index"
VEC_FILE = INDEX_DIR / "vectors.npy"
META_FILE = INDEX_DIR / "embed_meta.json"

MODEL = os.environ.get("KB_EMBED_MODEL", "vertex_ai/gemini-embedding-2")
BATCH = 32
RETRY = 3


def _cfg():
    base = os.environ.get("KB_EMBED_BASE", "").rstrip("/")
    key = os.environ.get("KB_EMBED_KEY", "")
    if not base or not key:
        return None
    return base, key


def available():
    """环境变量配好没有 + 向量建好没有。"""
    return _cfg() is not None and VEC_FILE.exists()


def _post(url, key, payload, timeout=120):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST"
    )
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return json.loads(r.read().decode("utf-8"))


def embed_texts(texts, progress=False):
    """逐条拿向量。返回 list[list[float]]。

    注意：这个端点对批量输入会静默只返回第 1 条，
    所以必须一条一请求（实测 32 条只回 1 条）。
    """
    cfg = _cfg()
    if cfg is None:
        raise RuntimeError("未配置 KB_EMBED_BASE / KB_EMBED_KEY")
    base, key = cfg
    out = []
    for n, text in enumerate(texts, 1):
        piece = text[:8000]  # 模型上限 8192 token
        for attempt in range(RETRY):
            try:
                resp = _post(f"{base}/embeddings", key, {"model": MODEL, "input": piece})
                items = resp["data"]
                if not items:
                    raise RuntimeError("空响应")
                out.append(items[0]["embedding"])
                break
            except Exception:
                if attempt == RETRY - 1:
                    raise
                time.sleep(2 * (attempt + 1))
        if progress and (n % 25 == 0 or n == len(texts)):
            print(f"  向量化 {n}/{len(texts)}", flush=True)
    if len(out) != len(texts):
        raise RuntimeError(f"向量数不匹配：拿到 {len(out)}，期望 {len(texts)}")
    return out


def build_vectors(chunks):
    """给 chunks 建向量并存盘。"""
    import numpy as np
    texts = [c["text"] for c in chunks]
    print(f"正在向量化 {len(texts)} 个块（模型 {MODEL}）...")
    vecs = embed_texts(texts, progress=True)
    arr = np.array(vecs, dtype="float32")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(VEC_FILE, arr)
    META_FILE.write_text(
        json.dumps({"model": MODEL, "count": len(texts), "dim": arr.shape[1]},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"✓ 向量已存：{arr.shape} → {VEC_FILE}")
    return arr


def load_vectors():
    import numpy as np
    if not VEC_FILE.exists():
        return None
    return np.load(VEC_FILE)


def semantic_rank(query, k=None, chunks=None, per_paper_norm=False):
    """
    返回 [(chunk_idx, score), ...] 按余弦相似度降序。

    per_paper_norm 默认关闭：实测"减去篇内均值"会反向奖励块数少的论文
    （gemini4-argon 只有 1 块，均值极低，减完反而霸榜），B 组召回从 86.7% 掉到 66.7%。
    保留参数以便将来换更好的归一化方式（如篇内 z-score 或按篇 rank 混合）。
    """
    import numpy as np
    arr = load_vectors()
    if arr is None:
        return []
    qv = np.array(embed_texts([query])[0], dtype="float32")
    # 归一化后点积 = 余弦相似度
    arr_n = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
    qv_n = qv / (np.linalg.norm(qv) + 1e-9)
    sims = arr_n @ qv_n

    if per_paper_norm and chunks is not None and len(chunks) == len(sims):
        from collections import defaultdict
        groups = defaultdict(list)
        for i, c in enumerate(chunks):
            groups[c["paper"]].append(i)
        for paper, idxs in groups.items():
            if len(idxs) < 10:      # 块太少不归一化（样本不足，均值不可靠）
                continue
            m = float(np.mean(sims[idxs]))
            for i in idxs:
                sims[i] -= m

    order = np.argsort(-sims)
    if k:
        order = order[:k]
    return [(int(i), float(sims[i])) for i in order]


def detect_query_type(query):
    """
    判断查询偏向术语还是口语，用于自适应权重。
    返回 'term' 或 'colloquial'。

    优先级：
      1. 强技术标识（技术缩写 / 希腊型号 / 英文术语连用）→ 一律 term
      2. 纯中文 → colloquial
      3. 其余按英文占比
    """
    import re
    q = query.strip()

    # ---- 1. 强技术标识，命中就是术语（哪怕夹中文）----
    if re.search(r"[A-Z]{2,}", q):                 # BM25 / KV / SPL / MoE / GR00T
        return "term"
    if re.search(r"[πτΨ][\s\*]?\d", q):             # π0.5 / π*0.6 / τ0.5
        return "term"
    # 英文技术词连用（2 个以上英文词，且不是普通名词）
    if re.search(r"[A-Za-z]{3,}[\s\-]+[A-Za-z]{3,}", q) and re.search(r"[A-Za-z]", q):
        # 排除"普通单词+普通单词"但整体是中文提问的情况
        if not re.search(r"(怎么|为什么|哪些|什么|哪篇|有没有|是不是)", q):
            return "term"

    # ---- 2. 纯中文 → 口语 ----
    en = len(re.findall(r"[A-Za-z]", q))
    zh = len(re.findall(r"[\u4e00-\u9fff]", q))
    if zh > 0 and en == 0:
        return "colloquial"
    if zh == 0:
        return "term"

    # ---- 3. 中英混合：按英文占比 ----
    ratio = en / max(en + zh, 1)
    return "term" if ratio >= 0.4 else "colloquial"


def adaptive_weights(query):
    """按查询类型返回 (w_sem, w_bm25)。"""
    if detect_query_type(query) == "term":
        return 0.4, 0.6      # 术语：保词频精度
    return 0.75, 0.25        # 口语：保语义召回


def diversify(hits, chunks, max_per_paper=2, k=None):
    """
    结果多样化：同一篇论文最多保留 max_per_paper 个块。

    解决"块数多的论文霸榜"问题（pi07 有 59 块、genie 只有 18 块）。
    hits: [(chunk_idx, fused, sem_score), ...] 已按 fused 降序
    k 给定时最多返回 k 条；被压制的高分块会跳过，用下一篇的最高分顶上。
    """
    out, count = [], {}
    for idx, fused, sem in hits:
        paper = chunks[idx]["paper"]
        if count.get(paper, 0) >= max_per_paper:
            continue
        count[paper] = count.get(paper, 0) + 1
        out.append((idx, fused, sem))
        if k and len(out) >= k:
            break
    return out


def weighted_fuse(bm25_hits, sem_hits, w_sem=0.7, w_bm25=0.3):
    """
    加权融合（这个库小、块少，加权比 RRF 更合适——RRF 会把强语义信号拉平）。

    bm25_hits: [(chunk_idx, bm25_score), ...]
    sem_hits : [(chunk_idx, cosine), ...]  已在 0~1 附近
    两路分数各自 min-max 归一化后线性加权。
    返回 [(chunk_idx, fused, sem_score), ...] 按 fused 降序。
    """
    def norm(hits):
        if not hits:
            return {}
        vals = [s for _, s in hits]
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            return {i: 1.0 for i, _ in hits}
        return {i: (s - lo) / (hi - lo) for i, s in hits}

    bn = norm(bm25_hits)
    sn = norm(sem_hits)
    sem_map = {i: s for i, s in sem_hits}

    all_idx = set(bn) | set(sn)
    out = []
    for idx in all_idx:
        fused = w_sem * sn.get(idx, 0.0) + w_bm25 * bn.get(idx, 0.0)
        out.append((idx, fused, sem_map.get(idx)))
    return sorted(out, key=lambda x: -x[1])


def rrf_fuse(bm25_hits, sem_hits, k=60):
    """
    RRF 融合（保留作对照）。
    bm25_hits / sem_hits: [(chunk_idx, score), ...] 按各自排名已排好
    返回 [(chunk_idx, rrf_score, sem_score), ...] 按融合分降序。
    第三项是原始余弦相似度（用于展示，没有语义分时为 None）。
    """
    scores = {}
    sem_map = {}
    for rank, (idx, _) in enumerate(bm25_hits, 1):
        scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank)
    for rank, (idx, s) in enumerate(sem_hits, 1):
        scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank)
        sem_map[idx] = s
    fused = sorted(scores.items(), key=lambda x: -x[1])
    return [(idx, sc, sem_map.get(idx)) for idx, sc in fused]
