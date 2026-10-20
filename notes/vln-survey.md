# 精读笔记 · VLN 综述（基础模型时代）

> arXiv 2407.07035 · 2024-07（v2 2024-12）· Zhang, Ma, Li, Qiao, Wang, Chai, Wu, Bansal, Kordjamshidi（密歇根/阿德莱德/UNC 等）· 仓库 zhangyuejoslin/VLN-Survey-with-Foundation-Models

## 基本情况

- 路线：**VLN 全景综述**（不是单一方法）
- 一句话：用"世界模型 / 人类模型 / 智能体"三层框架，把基础模型时代（2023 后）的 VLN 整条线梳理成一张图
- 价值：**代替 3-5 篇分散论文**，一次看清这条线的核心问题和各派解法

## 三层框架（这是综述的骨架，也值得借来用）

```
① World Model（世界模型）——理解并表征环境
   核心两个问题：历史/记忆怎么编码？如何泛化到没见过的环境？
② Human Model（人类模型）——理解人的指令
   核心两个问题：指令有歧义怎么办？跨环境怎么 grounding？
③ VLN Agent（智能体）——推理与规划
   核心三个问题：grounding、planning、要不要直接用基础模型当大脑
```

**这套框架和你的两问定位法互补**：两问是"预测什么/给谁用"，这三层是"环境/人/自己"，合起来就能给任何 VLN 论文定位。

## ① World Model：记忆与泛化

**历史编码的演进**：
- LSTM 隐状态（隐式记忆）→ 多模态 Transformer 序列编码
- 单 `[CLS]` token 记历史（RecurrentVLNBERT）→ 变长 memory bank（MTVM）
- 全景编码器 + 历史编码器分层设计（HAMT）——**摆脱了循环状态 token，大幅提升预训练效率**

**图结构的记忆**：拓扑图、俯视图、栅格图、语义地图、局部邻域图
- 当代：MapGPT 用"语言形式的图"存拓扑信息；MC-GPT 用拓扑图存视角/物体/空间关系

**泛化的四条路**：
1. 预训练视觉表征：ResNet → CLIP（图文对齐更好）→ 视频预训练（时序信息关键）
2. 环境增强：EnvEdit / EnvMix / KED / FDA（改房间、改风格）
3. **生成未来视角**：Pathdreamer、SE3DS——用合成的未来视角当增强数据（**这和你读的 DreamGen 同思路**）
4. 大规模域内预训练：已证明比直接用 Oscar/LXMERT 等 VLM 初始化更有效

## ② Human Model：歧义与 grounding

**指令歧义的解法**：
- **感知上下文 + 常识**：CLIP 匹配视觉语义 → 生成易跟随的子指令（VLN-Trans）；用 LLM 常识澄清地标（SayCan 把指令拆成可行动项）
- **主动求援（Information Seeking）**：三个问题——何时问（chi2020）、问什么、谁回答（人/规则/神经模型）
- **VLN-Copilot**：让 agent 在困惑时主动求助，LLM 当副驾驶

**grounding 泛化**：预训练表征 + 指令生成做数据增强

## ③ VLN Agent：三条路

**路线 A：VLM 直接当 agent**（主流）
- 单流 VLM 处理语言+视觉+历史 token，自注意力算动作概率
- **关键分水岭**：VLN-CE 的低层动作 vs VLN-DE 的图节点选择
- **waypoint predictor** 成了关键桥梁——用本地可通行图把 DE 的方法搬到 CE

**路线 B：LLM 当 agent**（把视觉转成文字描述）
- NavGPT（GPT-4 零样本）、MapGPT（拓扑图转全局探索提示）
- DiscussNav（多个领域专家分工）、InstructNav（拆子任务+多源价值图）
- **NavCoT**：让 LLM 当世界模型，模拟未来环境后再决策

**路线 C：规划器**
- 图规划器：全局图信息增强局部动作空间
- LLM 规划器：LLM-Planner 生成子目标计划、VL-Map 用 Code-as-Policy 拆成代码函数
- **DreamWalker / Lookahead**：用视频预测模型或神经辐射场预测未来 waypoint（**和你读的 WAM 系直接同源**）

## 未来方向（综述列的挑战）

- 基准：数据质量和多样性不足，R2R 指令偏向最短路径
- 世界模型、人类模型、智能体模型、真实机器人部署各自的缺口

## 谱系定位

- 这篇是你 VLN 线的**地图**：VLN-CE 定基准（2020）→ 本综述把 2020-2024 的分支理清 → NaVid 是 VLM 时代的代表作
- 与你的操作线对照：**VLN 的记忆问题（历史编码）在 VLA 里几乎不存在**（操作是短时程、马尔可夫的）——这正是两条线最本质的差异
- **DreamWalker 用视频预测做规划**，和你读的 WAM 系（DreamZero/VPP2）是同一技术路线在导航上的应用
