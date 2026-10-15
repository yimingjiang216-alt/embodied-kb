#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查检索库。

用法：
  python kb/search.py "flow matching beta distribution"
  python kb/search.py "KV cache" --paper pi0
  python kb/search.py "世界模型 谱系"        # 中文关键词也行（靠元数据/笔记里的中文）
  python kb/search.py "advantage" -k 5 --route VLA
"""
import argparse
import json
import math
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import BM25, tokenize, INDEX_DIR


def load():
    chunks = [json.loads(l) for l in open(INDEX_DIR / "chunks.jsonl", encoding="utf-8")]
    with open(INDEX_DIR / "bm25.pkl", "rb") as f:
        data = pickle.load(f)
    return chunks, data["tokens"]


def main():
    ap = argparse.ArgumentParser(description="论文检索库查询")
    ap.add_argument("query", help="查询词（英文术语最佳；支持 --paper/--route 过滤）")
    ap.add_argument("-k", type=int, default=8, help="返回条数（默认 8）")
    ap.add_argument("--paper", help="只查某篇，如 pi0、dreamerv3")
    ap.add_argument("--route", help="只查某路线，如 VLA、WAM、WM-latent")
    ap.add_argument("--kind", help="只查 paper / note / overview")
    a = ap.parse_args()

    chunks, tokens = load()
    if not chunks:
        sys.exit("索引为空，先跑 python kb/build.py")

    # 过滤
    idxs = list(range(len(chunks)))
    if a.paper:
        idxs = [i for i in idxs if a.paper.lower() in chunks[i]["paper"].lower()]
    if a.route:
        idxs = [i for i in idxs if a.route.lower() in chunks[i]["route"].lower()]
    if a.kind:
        idxs = [i for i in idxs if chunks[i]["kind"] == a.kind]
    if not idxs:
        sys.exit("过滤后没有块")

    sub_docs = [tokens[i] for i in idxs]
    bm25 = BM25(sub_docs)
    q = tokenize(a.query) + [a.query.lower()]  # 整串也当一个 token（型号名如 π0.5）
    hits = bm25.search([t for t in q if t], k=a.k)

    if not hits:
        print(f"没搜到：{a.query}")
        print("提示：换英文术语试试（如 'KV cache'、'flow matching timestep'）")
        return

    for rank, (s, j) in enumerate(hits, 1):
        c = chunks[idxs[j]]
        head = c["text"][:400].replace("\n", " ")
        print(f"[{rank}] {c['paper']} · {c['route']} · {c['section']} {c['line']}  (score={s:.2f})")
        print(f"    {head}...")
        print()


if __name__ == "__main__":
    main()
