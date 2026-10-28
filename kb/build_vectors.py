#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""建向量索引：给 kb/index/chunks.jsonl 里每个块生成 embedding。

用法：
  set KB_EMBED_BASE=https://pool.autelrobotics.com/v1
  set KB_EMBED_KEY=sk-xxx
  python kb/build_vectors.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import INDEX_DIR
import embed


def main():
    cfg = embed._cfg()
    if cfg is None:
        sys.exit(
            "未配置 embedding 端点。\n"
            "  set KB_EMBED_BASE=https://pool.autelrobotics.com/v1\n"
            "  set KB_EMBED_KEY=sk-xxx\n"
            "然后重跑本脚本。"
        )
    chunks_file = INDEX_DIR / "chunks.jsonl"
    if not chunks_file.exists():
        sys.exit("没有索引，先跑 python kb/build.py")

    chunks = [json.loads(l) for l in open(chunks_file, encoding="utf-8")]
    print(f"读取 {len(chunks)} 个块")
    embed.build_vectors(chunks)
    print("\n完成。现在可以用：python kb/search.py \"你的问题\" --semantic")


if __name__ == "__main__":
    main()
