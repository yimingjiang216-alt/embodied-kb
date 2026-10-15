#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
具身智能论文检索库 · MCP server

启动：
  python kb/mcp_server.py            # stdio 模式（Claude Code / Cursor 用这个）

客户端配置（以 Claude Code 为例，加到 .mcp.json 或 claude mcp add）：
  {
    "mcpServers": {
      "paper-kb": {
        "command": "python",
        "args": ["<绝对路径>/kb/mcp_server.py"]
      }
    }
  }

暴露 5 个工具：
  roadmap()                        学习路线总图（先看这个，理清 VLA / 世界模型的脉络）
  search(query, k, paper, route)   BM25 检索，返回带出处的段落
  list_papers()                    库里有什么论文、什么路线
  read_section(paper, section)     取某一篇某一节的完整文本（需拉过论文材料）
  routes()                         路线标签的含义

首次启动若还没有索引，会自动用 notes/ 建一次（几秒）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import BM25, tokenize, INDEX_DIR, ROUTE, NOTES, main as _build_index

from fastmcp import FastMCP

mcp = FastMCP("paper-kb")


def _load():
    """加载索引；第一次用时自动建库，而不是抛堆栈。"""
    if not (INDEX_DIR / "chunks.jsonl").exists():
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):   # stdio 协议不能被 print 污染
            _build_index()
    chunks = [json.loads(l) for l in open(INDEX_DIR / "chunks.jsonl", encoding="utf-8")]
    import pickle
    with open(INDEX_DIR / "bm25.pkl", "rb") as f:
        data = pickle.load(f)
    return chunks, data["tokens"]


@mcp.tool
def roadmap() -> str:
    """具身智能学习路线总图：VLA / 世界模型 / WAM 的谱系、每篇论文一句话、跨论文的三个机制。

    学习者问"该怎么入门 / 这些路线什么关系 / 某篇论文在哪个位置"时，先调这个。
    定位任何论文问两句：预测什么空间？预测出来给谁用？
    """
    f = NOTES / "00-谱系总图.md"
    return f.read_text(encoding="utf-8") if f.exists() else "谱系总图缺失（notes/00-谱系总图.md）"


@mcp.tool
def search(query: str, k: int = 8, paper: str = "", route: str = "") -> str:
    """BM25 检索具身智能知识库（精读笔记 + 谱系总图；若本地拉过论文材料，还包含论文正文并带行号）。

    何时用：用户问到 VLA / 世界模型 / WAM 某篇论文的具体内容
    （架构、数字、机制、与其他论文的关系），需要有出处的回答时。
    先看返回里的 kind：note=笔记摘要，paper=论文正文。

    Args:
        query: 查询词。英文术语最准（如 "flow matching timestep"、"KV cache"、
               "advantage conditioning"）；中文仅对笔记和总图有效。
        k: 返回条数，默认 8。
        paper: 只查某篇，如 pi0 / dreamerv3 / vpp2（目录名，可部分匹配）。
        route: 只查某路线：VLA / VLA+RL / WM-latent / WM-JEPA / WM-视频 /
               WM-平台 / WM-数据工厂 / WAM / 范式修正 / LLM路线 / 总览。
    """
    chunks, tokens = _load()
    if not chunks:
        return "索引为空：先跑 python kb/build.py"

    idxs = list(range(len(chunks)))
    if paper:
        idxs = [i for i in idxs if paper.lower() in chunks[i]["paper"].lower()]
    if route:
        idxs = [i for i in idxs if route.lower() in chunks[i]["route"].lower()]
    if not idxs:
        return f"过滤后没有块（paper={paper!r} route={route!r}）"

    bm25 = BM25([tokens[i] for i in idxs])
    q = tokenize(query) + [query.lower()]
    hits = bm25.search([t for t in q if t], k=min(k, 20))
    if not hits:
        return (f"没搜到：{query!r}\n"
                "提示：BM25 是词频匹配。口语化问题请换成术语，"
                '如"为什么预测错了动作会错" → "compounding error"')

    out = []
    for rank, (s, j) in enumerate(hits, 1):
        c = chunks[idxs[j]]
        # 笔记是主打内容、本来就短 → 多给；论文正文块是片段 → 截短
        limit = 1800 if c["kind"] in ("note", "overview") else 500
        head = c["text"][:limit].replace("\n", " ")
        out.append(f"[{rank}] ({c['kind']}) {c['paper']} · {c['route']} · {c['section']} {c['line']} (score={s:.2f})\n{head}")
    has_papers = any(c["kind"] == "paper" for c in chunks)
    tail = "\n— 回答请基于以上内容并注明来自哪篇；检索不到的不要编造。"
    if not has_papers:
        tail += ("\n— 当前只加载了精读笔记（未拉论文原文）。需要原文级细节时，"
                 "按 README 用 fetch_all.py 拉取论文后重建索引。")
    out.append(tail)
    return "\n\n".join(out)


@mcp.tool
def list_papers() -> str:
    """列出库里的全部论文、路线标签和块数。"""
    chunks, _ = _load()
    stats = {}
    for c in chunks:
        key = (c["paper"], c["route"])
        stats[key] = stats.get(key, 0) + 1
    lines = ["论文目录（目录名 · 路线 · 块数）："]
    for (p, r), n in sorted(stats.items(), key=lambda x: (x[0][1], x[0][0])):
        lines.append(f"  {p:<16} {r:<10} {n}")
    lines.append(f"\n共 {len(chunks)} 块。查询用 search，读某节用 read_section。")
    return "\n".join(lines)


@mcp.tool
def read_section(paper: str, keyword: str, max_chars: int = 6000) -> str:
    """读某篇论文里匹配关键词的章节全文（比 search 返回更多上下文）。

    Args:
        paper: 目录名，如 pi0 / pistar06。
        keyword: 章节名或其中的词，如 "Model Architecture"、"Inference"。
        max_chars: 最多返回字符数，默认 6000。
    """
    chunks, _ = _load()
    cand = [c for c in chunks
            if paper.lower() in c["paper"].lower() and c["kind"] == "paper"]
    if not cand:
        return (f"没有 {paper!r} 的论文正文块。可能原因：① 目录名不对（用 list_papers 看）；"
                "② 还没拉论文原文——当前可能只有精读笔记，请用 search 查笔记，"
                "或按 README 用 fetch_all.py 拉取论文后重建索引。")
    best = [c for c in cand if keyword.lower() in c["section"].lower()]
    if not best:  # 章节名没匹配上，退化为文本包含
        best = [c for c in cand if keyword.lower() in c["text"].lower()]
    if not best:
        return f"{paper!r} 里没有匹配 {keyword!r} 的章节"
    texts = [f"== {c['section']} {c['line']} ==\n{c['text']}" for c in best[:3]]
    return ("\n\n".join(texts))[:max_chars]


@mcp.tool
def routes() -> str:
    """路线标签的含义和归类逻辑（检索库的组织方式）。"""
    return """路线标签（--route 过滤 / 理解结果用）：

定位任何论文问两句：预测什么空间？预测出来给谁用？

光谱：纯文本预测(GPT) → 多模态理解(VLM) → 目标图生成(π0.7 BAGEL)
      → 视频预测(Cosmos) → 动作条件预测(V-JEPA 2-AC) → 手写物理(MuJoCo)
      （越往右：物理越具体、动作接口越真、错误越滚雪球）

VLA          视觉-语言-动作，模仿学习，不预测未来：pi0, pi05, pi07, gr00t
VLA-工具     VLA 配套技术：fast（动作 tokenizer）
VLA+RL       VLA+强化学习：pistar06（advantage 条件化）
WM-latent    latent 世界模型：worldmodels(2018), dreamer, dreamerv3
WM-JEPA      表示空间预测（非生成式）：vjepa2
WM-视频      视频扩散当规划器：unipi
WM-平台      视频世界基础模型平台：cosmos
WM-数据工厂  世界模型造合成数据：dreamgen（神经轨迹）
WAM          世界动作模型，视频+动作联合生成：dreamzero, vpp2
范式修正     action-to-future，世界模型只当老师：fdm
LLM路线      通用大模型操纵机器人工具栈：gemini4-argon
"""


if __name__ == "__main__":
    mcp.run()
