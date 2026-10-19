#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
建检索库：读 papers/（论文材料）+ notes/（精读笔记），切块，建 BM25 索引。

用法：
  python kb/build.py          # 全量重建（几秒）

目录约定（新仓库结构）：
  papers/<名字>/paper.tex + paper.txt    ← 用 kelip-paper-reading 的 fetch_paper.py 拉
  notes/<名字>.md                        ← 精读笔记（含 00-谱系总图.md）

产出：
  kb/index/chunks.jsonl   每行一个块 {text, paper, route, section, line, kind}
  kb/index/bm25.pkl       索引（search.py / mcp_server.py 共用）
"""
import json
import math
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPERS = ROOT / "papers"
NOTES = ROOT / "notes"
INDEX_DIR = Path(__file__).resolve().parent / "index"

# ---------------- 路线标签 ----------------
ROUTE = {
    "pi0": "VLA", "pi05": "VLA", "pi07": "VLA", "fast": "VLA-工具",
    "gr00t": "VLA", "pistar06": "VLA+RL",
    "worldmodels": "WM-latent", "dreamer": "WM-latent", "dreamerv3": "WM-latent",
    "planet": "WM-latent",
    "vjepa2": "WM-JEPA", "unipi": "WM-视频", "cosmos": "WM-平台",
    "genie": "WM-交互", "walloss": "VLA", "lingbot": "VLA", "openvla": "VLA",
    "dreamgen": "WM-数据工厂", "dreamzero": "WAM", "vpp2": "WAM",
    "fdm": "范式修正", "gemini4-argon": "LLM路线",
}

# ---------------- 简易 BM25（不依赖第三方库） ----------------
class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = docs                       # 每篇是 token 列表
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(self.N, 1)
        self.tf = [Counter(d) for d in docs]
        self.df = Counter()
        for d in docs:
            for w in set(d):
                self.df[w] += 1

    def idf(self, w):
        n = self.df.get(w, 0)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)

    def score(self, qi, di):
        s, tf_d, dl = 0.0, self.tf[di], len(self.docs[di])
        for w in qi:
            f = tf_d.get(w, 0)
            if f == 0:
                continue
            s += self.idf(w) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def search(self, query_tokens, k=8):
        scores = [(self.score(query_tokens, i), i) for i in range(self.N)]
        scores.sort(key=lambda x: -x[0])
        return [(s, i) for s, i in scores[:k] if s > 0]


TOKEN_RE = re.compile(r"[A-Za-zα-ωΑ-Ωπ0-9][A-Za-zα-ωΑ-Ωπ0-9_.\-]*", re.UNICODE)

def tokenize(text):
    """英文术语为主 + 中文按 bigram（让中文查询也能命中）。"""
    toks = [t.lower().strip(".-") for t in TOKEN_RE.findall(text) if len(t) > 1]
    # 中文 bigram："世界模型路线" → [世界, 界模, 模型, 型路, 路线] + 整串
    han = re.findall(r"[\u4e00-\u9fff]+", text)
    for seg in han:
        if len(seg) >= 2:
            toks += [seg[i:i+2] for i in range(len(seg) - 1)]
            if len(seg) >= 3 and len(seg) <= 8:
                toks.append(seg)  # 短语整串，如"世界模型"
    return toks


# ---------------- 切块 ----------------
SEC_RE = re.compile(r"\\(?:sub)*section\*?\s*\{([^}]*)\}")

def split_tex(paper_dir):
    """按 LaTeX 章节切块；公式块整块保留。行号跟着走。"""
    tex = paper_dir / "paper.tex"
    if not tex.exists():
        # 没有源码（官方 PDF）：用 paper.txt 按 ===== 第 N 页 ===== 切
        return split_txt(paper_dir)
    lines = tex.read_text(encoding="utf-8", errors="ignore").split("\n")
    chunks, cur, cur_sec, cur_start = [], [], "前言", 0

    def flush(end_line):
        if cur:
            text = "\n".join(cur).strip()
            if len(text) > 120:  # 太碎的不要
                chunks.append({"text": text, "section": cur_sec,
                               "line": f"L{cur_start}-L{end_line}", "kind": "paper"})
            cur.clear()

    for i, line in enumerate(lines, 1):
        m = SEC_RE.search(line)
        if m:
            flush(i - 1)
            cur_sec, cur_start = m.group(1).strip(), i
        cur.append(line)
    flush(len(lines))
    return chunks


def split_txt(paper_dir):
    """无 LaTeX 源码时：paper.txt 按页切块，每 ~60 行并一块。"""
    txt = paper_dir / "paper.txt"
    if not txt.exists():
        return []
    lines = txt.read_text(encoding="utf-8", errors="ignore").split("\n")
    chunks, cur, cur_page, cur_start = [], [], 1, 0
    PAGE_RE = re.compile(r"=====\s*第\s*(\d+)\s*页\s*=====")

    def flush(end):
        if cur:
            text = "\n".join(cur).strip()
            if len(text) > 300:
                chunks.append({"text": text[:4000], "section": f"第{cur_page}页",
                               "line": f"L{cur_start}-L{end}", "kind": "paper"})
            cur.clear()

    for i, line in enumerate(lines, 1):
        m = PAGE_RE.search(line)
        if m:
            flush(i - 1)
            cur_page, cur_start = int(m.group(1)), i
            continue
        cur.append(line)
        if len(cur) >= 60:  # 页内再细分
            flush(i)
    flush(len(lines))
    return chunks


def split_notes_md(note_path):
    """notes/ 下的一篇笔记整篇当一个块（本来就不长）。"""
    if not note_path.exists():
        return []
    text = note_path.read_text(encoding="utf-8", errors="ignore").strip()
    return [{"text": text, "section": "精读笔记", "line": "", "kind": "note"}]


def split_any(path, label):
    """谱系总图等独立 md。"""
    if not path.exists():
        return []
    return [{"text": path.read_text(encoding="utf-8", errors="ignore").strip(),
             "section": label, "line": "", "kind": "overview"}]


# ---------------- 主流程 ----------------
def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks, docs_tokens = [], []

    # 各论文：papers/ 提供正文，notes/<名字>.md 提供笔记（缺任一也能建库）
    paper_dirs = sorted(d for d in PAPERS.iterdir() if d.is_dir()) if PAPERS.exists() else []
    for d in paper_dirs:
        route = ROUTE.get(d.name, "未分类")
        chunks = split_tex(d)
        note = NOTES / f"{d.name}.md"
        if note.exists():
            chunks += split_notes_md(note)
        for c in chunks:
            meta = {"paper": d.name, "route": route, **c}
            all_chunks.append(meta)
            docs_tokens.append(tokenize(c["text"]))
        if chunks:
            print(f"  {d.name:<16} {route:<10} {len(chunks)} 块")

    # 只有笔记、没有 papers/ 材料的条目（如谱系总图，或未拉材料的论文）
    if NOTES.exists():
        for note in sorted(NOTES.glob("*.md")):
            stem = note.stem
            if stem.startswith("00-"):            # 谱系总图
                for c in split_any(note, "谱系总图"):
                    meta = {"paper": "总览", "route": "总览", **c}
                    all_chunks.append(meta); docs_tokens.append(tokenize(c["text"]))
                print(f"  {'总览':<16} {'总览':<10} 1 块")
            elif not any(d.name == stem for d in paper_dirs):
                route = ROUTE.get(stem, "未分类")
                for c in split_notes_md(note):
                    meta = {"paper": stem, "route": route, **c}
                    all_chunks.append(meta); docs_tokens.append(tokenize(c["text"]))
                print(f"  {stem:<16} {route:<10} 1 块（仅笔记）")

    if not all_chunks:
        sys.exit("没有任何块：请先拉论文材料（papers/）或放笔记（notes/）。见 README。")

    bm25 = BM25(docs_tokens)
    with open(INDEX_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump({"tokens": docs_tokens}, f)

    print(f"\n共 {len(all_chunks)} 块，索引已写入 {INDEX_DIR}")


if __name__ == "__main__":
    main()
