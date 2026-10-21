# embodied-kb

**帮你理清具身智能的学习路线：VLA 与世界模型，到底是怎么一回事、谁和谁什么关系。**
26 份精读笔记 + 一张路线总图，做成 RAG 知识库，并用 MCP 封装——任何 MCP 客户端（Claude Code / Cursor / Codex / 自建 harness）连上就能问。

> 学具身智能最先遇到的不是难，是**乱**：π0、Dreamer、Cosmos、DreamZero……每篇都自称"世界模型"或"基础模型"，路线图上没人告诉你它们怎么分。这个库就是为解决这个问题做的。

## 给学习者的使用路径（15 分钟上手）

**这个库不是教材，是地图。** 它不教你梯度下降和 transformer（那去上课），它帮你把 200+ 篇论文的领域摆成一张能看懂的图。四步：

```
第 0 步 · 装上（10 分钟）
  git clone 本仓库 → pip install fastmcp → python kb/build.py

第 1 步 · 先看地图，不读论文（20 分钟）
  打开 notes/00-谱系总图.md，看懂三样：
  ① 光谱图（6 档：从纯文本预测到手写物理）
  ② 两问定位法（拿到任何论文先问：预测什么空间？给谁用？）
  ③ 26 篇的相对位置

第 2 步 · 连上 MCP 问问题（日常）
  下面"用法一"配置好后，像跟读过 26 篇的学长聊天：
  "VLA 和世界模型什么区别？" / "DreamZero 和 VPP2 分歧在哪？"
  每个回答带出处（论文+章节+行号），查不到就说查不到，不编

第 3 步 · 按图索骥读原文（想深入时）
  笔记开头都有 arXiv 编号，照"用法三"拉原文精读；
  read_section 能直接取某篇某节

第 4 步 · 长自己的图（进阶）
  自己读第 26 篇：fetch → 照模板写笔记（两问定位）→ build 入库
  fork 走，长成你自己的版本——这是这个库最好的结局
```

**适合谁**：会跑模型/写代码、想快速建立领域地图的人；想要"带出处的 AI 问答"的人。
**不适合谁**：零基础找入门课的；想不动手就学会的。

## 它给你什么

**一个判断框架**（比记路线名有用）——拿到任何一篇论文，只问两句：

> ① 它预测的是什么空间？（像素 / 抽象表示 / 目标图 / 物理状态 / 什么都不预测）
> ② 预测出来给谁用？（当练习场 / 当参考答案 / 当数据工厂 / 只当老师）

用这两问，"五条路线"塌缩成"一个功能 + 四种用法"。完整的光谱图在 [`notes/00-谱系总图.md`](notes/00-谱系总图.md)：

```
纯文本预测 ──▶ 多模态理解 ──▶ 目标图生成 ──▶ 视频预测 ──▶ 动作条件预测 ──▶ 手写物理
   GPT            VLM          π0.7 BAGEL      Cosmos       V-JEPA 2-AC      MuJoCo
   ──── 越往右：物理越具体、动作接口越真、错误越滚雪球 ────▶
```

**26 篇论文的精读笔记**，每篇 30–60 行：路线标签 / 一句话 / 关键数字 / 它修了前人的什么 / 谱系位置。

| 路线 | 论文 |
|---|---|
| **VLA** | **OpenVLA**（开源标杆，离散 token）｜π0 → FAST → π0.5 → GR00T N1 → π\*0.6（RL，从 π0.6 学经验）→ π0.7 |
| **世界模型** | World Models(2018) → PlaNet → Dreamer → DreamerV3 → V-JEPA 2 |
| | UniPi → Cosmos → DreamGen（视频系的三种用法） |
| | Genie 1（动作接口从视频无监督发现） |
| **WAM** | DreamZero、VPP2 |
| **范式修正** | FDM（action-to-future） |
| **导航** | VLN-CE（连续环境基准）→ VLN 综述（三层框架）→ NaVid（视频 VLM，不用深度） |
| **仿真器** | Isaac Lab（数据从哪来 / sim2real 差距归因） |
| **LLM 路线** | Gemini 4 Argon（博客） |

## 用法一：连上 MCP，直接问（推荐）

任何支持 MCP 的客户端都可以——Claude Code、Cursor、Codex、 Windsurf、自建的 agent harness……协议是标准的，server 就是一个 stdio 进程。

```bash
git clone https://github.com/yimingjiang216-alt/embodied-kb.git
pip install fastmcp
```

通用配置（各客户端的 MCP 配置文件里都是这个结构）：

```json
{ "mcpServers": { "paper-kb": {
    "command": "python",
    "args": ["<仓库绝对路径>/kb/mcp_server.py"]
} } }
```

Claude Code 一条命令等效：
```bash
claude mcp add paper-kb -- python "<仓库绝对路径>/kb/mcp_server.py"
```

首次启动会自动建索引（几秒）。连上后 agent 多出五个工具：

| 工具 | 干什么 |
|---|---|
| `roadmap()` | **先调这个**：路线总图 + 两问定位法 |
| `search(query, k, paper, route)` | BM25 检索，返回带出处的笔记/段落 |
| `list_papers()` | 库里有哪些论文、什么路线 |
| `read_section(paper, keyword)` | 读某篇某节全文（需先拉论文原文，见下） |
| `routes()` | 路线标签的含义 |

然后直接问：

- "VLA 和世界模型到底什么区别？"
- "DreamZero 和 VPP2 都是 WAM，分歧在哪？"
- "π0.7 用的世界模型和 Cosmos 是一回事吗？"

## 用法二：命令行

```bash
python kb/build.py
python kb/search.py "advantage conditioning"
python kb/search.py "compounding error" --route WAM
```

## 可选：把检索从"笔记"升级到"论文全文"（= 使用路径第 3 步）

默认只索引笔记与总图，开箱 10 秒可用。想让 `search` 也能命中论文正文（并带 `paper.tex` 行号核对原文）：

```bash
git clone https://github.com/skJack/kelip-paper-reading.git   # 抓取脚本
pip install pymupdf
python kb/fetch_all.py        # 按笔记里的 arXiv 编号拉取，并核对标题防拉错
python kb/build.py            # 重建索引
```

诚实说明：论文材料因版权**不随仓库分发**；`fetch_all.py` 需要能访问 arxiv.org，约十几分钟。
π0.7、π\*0.6 没有 arXiv 版本，脚本会提示你到官网手动下载 PDF；Gemini Argon 是博客，无需拉取。

## 检索评测（30 条人工标注测试集）

```bash
python kb/eval.py
```

| 查询类型 | Recall@5 | MRR |
|---|---|---|
| 术语查询（英文，如 "flow matching timestep"） | **100%** | **0.967** |
| 口语化/中文（如 "机器人怎么会自己想"） | 80% | 0.644 |
| 总计 | **90%** | **0.806** |

这组数字回答了两个问题：**① BM25 选型的实证**——术语场景全中、正确答案平均排第一位（MRR 0.967）；**② 向量层的量化需求**——口语化查询 20% 未命中，且未命中的全是语义鸿沟（"自己想"→imagination、词面零重合），这正是嵌入模型的主场。结论：当前以 agent 查询改写补口语需求，向量层留作验证过的升级路径（API 嵌入几分钱可加），而非默认依赖。

测试集在 [kb/eval.py](kb/eval.py)，欢迎扩充——每条 = query + 期望命中的论文。

## 它是怎么搭的（一个最小的 RAG + MCP）

```
 notes/*.md  ─┐                      ┌─ roadmap()   直接返回路线总图
 (papers/)  ──┼─▶ 切块 ─▶ BM25 索引 ─┼─ search()    检索 → 带出处的块
              │   (带 论文/路线/行号) └─ read_section()
              └ 首次调用自动建库                     ▲
                                                     │ stdio MCP
                                          任何 MCP 客户端 …
```

几个有意的取舍：

- **用 BM25，不上嵌入模型。** 这批内容的关键词全是术语和型号（`KV cache`、`flow matching`、`π0.5`），词频检索正好擅长；零模型下载，秒级建库，检索结果可逐条解释命中原因。
- **口语化查询交给 agent 改写。** BM25 只做词面匹配，"机器人怎么会自己想"这类问法需要 agent 先翻译成术语（如 `world model imagination`）再检索。这样不需要额外的 embedding 模型和索引，代价是效果依赖 agent 的改写质量。
- **向量检索留作升级路径，现在不默认启用。** 若要引入，做法是：切块编码一次存成向量文件，查询时只编码查询本身，与 BM25 结果做 RRF 排名融合，BM25 那一路保持不变。接口上只需要实现一个 `embed(texts) -> ndarray` 函数，换模型就换实现；换模型后须重建索引（维度不同）。是否引入应以现有 30 条测试集为基线做对照——口语化那 20% 未命中是否被补上，是判断它值不值的唯一标准。
- **R 与 G 分离。** 库只负责"找"，"答"由你的 agent 完成；工具描述里要求它注明出处、检索不到就说查不到。
- **笔记优先。** 笔记是主体内容，本身已按"预测什么 / 给谁用"整理过；论文原文是可选的深挖层。

已知局限（别被坑）：

- BM25 是词频匹配，不是语义理解。口语化问题（"机器人怎么会自己想"）搜不到，换成术语（`world model imagination`）才行——解法见上文"口语化查询交给 agent 改写"。
- 中文查询靠 bigram，只对笔记和总图有效；论文正文是英文。
- 笔记是**我读论文时写的学习笔记**，不是论文本身；关键数字请以原文为准。π0.6/π0.7 的笔记基于官方博客与 PDF，非同行评审论文。

## 文件结构

```
kb/
├── mcp_server.py   MCP server（5 个工具，stdio）
├── build.py        切块 + 建 BM25 索引（缺 papers/ 时只建笔记）
├── search.py       命令行检索
└── fetch_all.py    可选：拉论文原文（带标题核对）
notes/              26 份精读笔记 + 路线总图
```

## 新增一篇

笔记按 `notes/<名字>.md` 放入，第一行写 `# 精读笔记 · 标题`，开头注明 `arXiv 编号`，在 `kb/build.py` 的 `ROUTE` 字典加一行路线标签，重启 MCP 或重跑 `build.py` 即可。

## FAQ

**为什么不直接把论文丢给大模型读？**
大模型不知道你库里有什么，容易凭印象作答。这里先检索、再回答，每条答案带出处（论文 + 章节 + 行号），检索不到就明确说查不到。

**为什么用 BM25 而不是向量检索？**
这批内容的关键词全是术语和型号，词面匹配正好擅长，且零模型下载、可逐条解释命中原因。口语化换问法的需求交给 agent 改写（见"几个有意的取舍"）。向量层怎么加、按什么标准判断值不值，也写在同一节。

**笔记和论文原文哪个更可信？**
笔记是读论文时的理解，用来快速建立地图；关键数字、公式、实验设置请以原文为准，笔记开头都标了 arXiv 编号。

**能换成我自己的领域吗？**
能。把 `notes/` 换成你自己的笔记，在 `kb/build.py` 的 `ROUTE` 字典里加上领域标签，重跑 `build.py` 即可。检索层和 MCP 封装都不用改。

**对话里问不出来怎么办？**
先换术语再问。比如"机器人怎么会自己想"→ `world model imagination`、"为什么不用视频当训练数据"→ `DreamGen`。

## 致谢

论文抓取与切块思路源自 [kelip-paper-reading](https://github.com/skJack/kelip-paper-reading)（MIT）；笔记按其"动机 → 架构 → 训练 → 数据 → 信息流 → 例子"的读法写成。

## License

MIT（代码与笔记）。论文原文版权归原作者，请通过 arXiv / 官方渠道获取。
