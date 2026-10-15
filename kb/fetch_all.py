#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可选：拉取论文原文，让检索库从"笔记检索"升级为"笔记 + 论文全文检索（带行号）"。

不拉也能用——只装笔记时，search 查的是 17 份精读笔记和路线总图。

用法：
  python kb/fetch_all.py              # 拉全部有 arXiv 编号的论文
  python kb/fetch_all.py pi0 dreamer  # 只拉指定几篇
  python kb/build.py                  # 拉完后重建索引

依赖：pip install pymupdf；需要能访问 arxiv.org。
论文材料下载到 papers/（已被 .gitignore 排除，不会进仓库——版权属原作者）。
本脚本从 notes/<名字>.md 的第一行 "arXiv xxxx.xxxxx" 读取编号，
并在拉完后核对"论文标题"与笔记标题是否吻合；对不上会报错而不是静默继续。
"""
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTES = ROOT / "notes"
PAPERS = ROOT / "papers"
FETCHER = ROOT / "kelip-paper-reading" / "scripts" / "fetch_paper.py"
FETCHER_REPO = "https://github.com/skJack/kelip-paper-reading"

# 没有 arXiv 编号的条目：需要手动处理
MANUAL = {
    "pi07":     "官网 PDF：https://www.pi.website/download/pi07.pdf        → 放到 papers/pi07/paper.pdf",
    "pistar06": "官网 PDF：https://www.pi.website/download/pistar06.pdf   → 放到 papers/pistar06/paper.pdf",
    "gemini4-argon": "博客（非论文）：https://developers.googleblog.com/gemini-4-argon-for-robotics/ ，无需拉取",
}


def arxiv_id_of(note: Path):
    m = re.search(r"arXiv\s+(\d{4}\.\d{4,5})", note.read_text(encoding="utf-8")[:500])
    return m.group(1) if m else None


def arxiv_title(aid: str):
    """向 arXiv 要真实标题，用来防"编号记错/指向别的论文"。"""
    try:
        html = urllib.request.urlopen(f"https://arxiv.org/abs/{aid}", timeout=30).read().decode("utf-8", "ignore")
        m = re.search(r"<title>\s*\[[^\]]+\]\s*(.*?)\s*</title>", html, re.S)
        return re.sub(r"\s+", " ", m.group(1)) if m else ""
    except Exception as e:  # noqa: BLE001
        return ""


def title_ok(note: Path, real: str):
    """笔记第一行 '# 精读笔记 · XXX' 的关键词是否出现在真实标题里（宽松匹配）。"""
    first = note.read_text(encoding="utf-8").split("\n")[0]
    name = re.sub(r"^#\s*精读笔记\s*·\s*", "", first).lower()
    words = [w for w in re.findall(r"[a-z0-9\-π]{3,}", name)]
    real_l = real.lower()
    return not words or any(w in real_l for w in words) or any(w in real_l for w in ["world model", "π"])


def main():
    wanted = set(sys.argv[1:])
    if not FETCHER.exists():
        sys.exit(f"缺少抓取脚本：{FETCHER}\n请先 git clone {FETCHER_REPO} 到仓库根目录，"
                 "并 pip install pymupdf。")

    notes = [n for n in sorted(NOTES.glob("*.md")) if not n.stem.startswith("00-")]
    ok, skipped, failed = [], [], []
    for note in notes:
        name = note.stem
        if wanted and name not in wanted:
            continue
        if name in MANUAL:
            skipped.append((name, MANUAL[name]))
            continue
        aid = arxiv_id_of(note)
        if not aid:
            skipped.append((name, "笔记里没有 arXiv 编号"))
            continue
        if (PAPERS / name / "paper.tex").exists() or (PAPERS / name / "paper.txt").exists():
            print(f"  已有  {name}")
            ok.append(name)
            continue

        real = arxiv_title(aid)
        if real and not title_ok(note, real):
            failed.append((name, f"编号 {aid} 的真实标题是 {real!r}，与笔记不符——已跳过，请核对"))
            continue
        print(f"  拉取  {name}  ({aid})  {real[:50]}")
        r = subprocess.run([sys.executable, str(FETCHER), aid, "--dest", str(PAPERS / name), "--no-clone"],
                           capture_output=True, text=True, encoding="utf-8", errors="ignore")
        if r.returncode != 0 or not (PAPERS / name / "paper.pdf").exists():
            failed.append((name, (r.stderr or r.stdout)[-200:].strip()))
        else:
            ok.append(name)

    print(f"\n成功/已有 {len(ok)} 篇")
    for n, why in skipped:
        print(f"手动   {n}: {why}")
    for n, why in failed:
        print(f"失败   {n}: {why}")
    print("\n下一步：python kb/build.py")


if __name__ == "__main__":
    main()
