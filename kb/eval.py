"""评测 embodied-kb 检索质量：recall@5 / MRR。

测试集设计（30 条，人工审核过）：
- A 组 15 条：术语查询（BM25 主场，预期高分）
- B 组 15 条：口语化/中文查询（BM25 的已知弱区，量化"agent 改写够不够"）

每条标注 expected：应命中的论文目录名（允许命中该论文的任意块即算对）。
跑法：python kb/eval.py
"""

# ---------------- 测试集 ----------------
# A 组：术语查询（英文，BM25 应主场）
QUERIES_A = [
    {"q": "flow matching timestep beta distribution", "expected": ["pi0", "gr00t"]},
    {"q": "KV cache attention keys values", "expected": ["pi0", "dreamzero"]},
    {"q": "action expert 860M parameters", "expected": ["pi07", "pistar06"]},
    {"q": "RSSM deterministic stochastic path", "expected": ["planet", "dreamer", "dreamerv3"]},
    {"q": "advantage conditioning value function", "expected": ["pistar06"]},
    {"q": "latent action model VQ codebook unsupervised", "expected": ["genie"]},
    {"q": "DCT discrete cosine transform action tokenization BPE", "expected": ["fast"]},
    {"q": "compounding error world model exploit", "expected": ["worldmodels", "dreamzero"]},
    {"q": "neural trajectory synthetic data generation", "expected": ["dreamgen", "gr00t"]},
    {"q": "subgoal image generation world model BAGEL", "expected": ["pi07"]},
    {"q": "CEM cross entropy method planning latent space", "expected": ["planet", "vjepa2"]},
    {"q": "twohot symlog distributional critic", "expected": ["dreamerv3", "pistar06"]},
    {"q": "video tokenizer FSQ causal wavelet", "expected": ["cosmos"]},
    {"q": "MoT mixture of transformers action expert LoRA", "expected": ["vpp2", "pi07"]},
    {"q": "inverse dynamics model video to action", "expected": ["unipi", "dreamgen"]},

    # B 组：口语化/中文查询（BM25 弱区——量化 agent 改写的必要性）
    {"q": "机器人怎么会自己想", "expected": ["dreamer", "worldmodels"]},           # 想象训练
    {"q": "视频模型会做梦吗", "expected": ["dreamgen", "worldmodels"]},             # 神经轨迹/梦里训练
    {"q": "哪个模型能走进没见过的家", "expected": ["pi05"]},                        # 开放世界泛化
    {"q": "怎么让机器人从错误里学习", "expected": ["pistar06"]},                      # Recap RL
    {"q": "动作太慢怎么办 推理延迟", "expected": ["dreamzero", "vpp2", "fast"]},    # 加速线
    {"q": "两篇文章都说世界模型 打架了吗", "expected": ["vjepa2", "cosmos"]},        # 之争
    {"q": "机器人叠衣服是哪篇", "expected": ["pi0", "pistar06", "vpp2"]},           # 叠衣服任务
    {"q": "有没有不用人教自己学会的", "expected": ["genie"]},                        # 无监督动作发现
    {"q": "世界模型的祖师爷是哪篇", "expected": ["worldmodels"]},                    # 2018 起点
    {"q": "怎么给机器人造假数据", "expected": ["dreamgen", "gr00t"]},                # 合成数据
    {"q": "机器人能听懂人话吗 指令跟随", "expected": ["pi05", "pi07"]},              # 语言指令
    {"q": "Minecraft 挖钻石", "expected": ["dreamerv3"]},                           # 150 任务
    {"q": "预测未来画面有什么用", "expected": ["unipi", "dreamzero", "fdm"]},        # 视频预测用途
    {"q": "机器人手臂末端七个数字是什么", "expected": ["cosmos", "vjepa2"]},          # 7 维动作
    {"q": "大模型能直接操控机器人吗", "expected": ["gemini4-argon"]},                 # Argon
]

# ---------------- 评测逻辑 ----------------
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import BM25, tokenize, INDEX_DIR
import json, pickle


def load():
    chunks = [json.loads(l) for l in open(INDEX_DIR / "chunks.jsonl", encoding="utf-8")]
    with open(INDEX_DIR / "bm25.pkl", "rb") as f:
        tokens = pickle.load(f)["tokens"]
    return chunks, tokens


def evaluate(queries, label, k=5):
    chunks, tokens = load()
    bm25 = BM25(tokens)
    recalls, mrrs, misses = [], [], []
    for item in queries:
        q, expected = item["q"], item["expected"]
        toks = [t for t in tokenize(q) + [q.lower()] if t]
        hits = bm25.search(toks, k=k)
        hit_papers = [chunks[i]["paper"] for _, i in hits] if hits else []
        # recall@k: 命中任意一个 expected 即算命中（宽松口径，标注可能多篇都对）
        hit = any(p in hit_papers for p in expected)
        recalls.append(1.0 if hit else 0.0)
        # MRR: 第一个命中的 expected 排第几
        rr = 0.0
        for rank, p in enumerate(hit_papers, 1):
            if p in expected:
                rr = 1.0 / rank
                break
        mrrs.append(rr)
        if not hit:
            misses.append((q, hit_papers[:3], expected))
    r = sum(recalls) / len(recalls)
    m = sum(mrrs) / len(mrrs)
    print(f"\n=== {label}（{len(queries)} 条，k={k}）===")
    print(f"  Recall@{k}: {r:.1%}   MRR: {m:.3f}")
    if misses:
        print(f"  未命中 {len(misses)} 条：")
        for q, got, exp in misses:
            print(f"    ✗ 「{q}」→ 得到 {got}，期望 {exp}")
    return r, m


if __name__ == "__main__":
    a = [q for q in QUERIES_A[:15]]
    b = [q for q in QUERIES_A[15:]]
    r_a, m_a = evaluate(a, "A 组 · 术语查询（BM25 主场）")
    r_b, m_b = evaluate(b, "B 组 · 口语化/中文查询（BM25 弱区）")
    r_all = (r_a + r_b) / 2
    m_all = (m_a + m_b) / 2
    print(f"\n=== 总计（30 条）===")
    print(f"  Recall@5: {r_all:.1%}   MRR: {m_all:.3f}")
    print(f"\n解读：A 组分数 = BM25 选型的实证；B 组分数 = agent 改写/向量层的真实需求大小")
