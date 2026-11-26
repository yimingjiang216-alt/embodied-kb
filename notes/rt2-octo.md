# 精读笔记 · RT-2 & Octo（VLA 概念诞生与开源跨本体鼻祖）

> RT-2: arXiv 2307.15818 · Google DeepMind（Brohan et al.）· 2023-07（闭源）
> Octo: arXiv 2405.12213 · RSS 2024 · UC Berkeley / Stanford 等（Octo Model Team）· 2024-05（开源）

## 基本情况

- 路线：**VLA 第一代奠基（离散动作 Token 路线 vs. 模块化扩散跨本体路线）**
- 两问定位：**① 预测什么空间？** RT-2 预测离散化为文本字符串的 256-bin 动作 Token；Octo 预测连续扩散动作块（Diffusion Action Head）；**② 预测出来给谁用？** 直接作为机械臂 6-DoF 末端位姿增量 + 夹爪开合指令执行
- 一句话：**RT-2** 正式命名了 **VLA（Vision-Language-Action）**，证明把动作当成一种“外语”与互联网图文数据联合微调（Co-Fine-Tuning），能让机器人直接继承 VLM 的常识推理；**Octo** 则在 Open X-Embodiment（80 万条轨迹）上训出了第一个开源、支持任意相机配置与语言/目标图双条件的通用策略底座

## 核心设计对比

| 维度 | RT-2（Google DeepMind, 55B PaLI-X / 12B PaLM-E） | Octo（UC Berkeley, 27M / 93M 开源 Transformer） |
|---|---|---|
| **核心思路由来** | 把大视觉语言模型直接变成控制策略 | 把跨机构异构机器人数据集（OXE）变成即插即用的预训练初始化权重 |
| **动作怎么表示** | 每个动作维度均匀切 256 个桶，直接复用词表里的数字/罕见字符 Token：`1 128 91 241 5 101 127` | 轻量级 3 层 MLP 扩散头（Diffusion Head），接在 Transformer 的读出（Readout）Token 后连续去噪 |
| **关键设计亮点** | **Co-Fine-Tuning（图文与机器人轨迹联合微调）**：若只训机器人数据会灾难性遗忘互联网语义，联合微调让未见物体/背景/语义推理提升 **3×（32%→62%）**，还能做具身思维链（Chain-of-Thought）先输出子目标再出动作 | **Block-Wise Attention 模块化输入**：支持在微调新机器人时动态增删相机视角、切换文本指令或目标图像（Goal Image），无需改动主干架构 |
| **核心局限** | 55B 闭源模型仅能跑 **1~3 Hz**；256-bin 均匀离散化在高频精细动作上精度受限 | 未借力十亿级预训练 VLM 主干（语言仅用冻结 T5-base），开放词汇语义理解弱于 VLM 底座 |

## 谱系定位

- **这两篇的优缺点直接催生了 2024 下半年的两座高峰**：
  1. 为了把闭源的 RT-2 变成人人能跑的开源 VLM，Stanford/Berkeley 做了 **OpenVLA（2024-06，Llama 2 7B + DINOv2 + SigLIP，复用 RT-2 的 256-bin 离散 Token，7B 击败 55B RT-2-X）**；
  2. 为了把 RT-2 的“VLM 语义大脑”和 Octo / Diffusion Policy 的“高频连续扩散动作头”合二为一，Physical Intelligence 做了 **$\pi_0$（2024-10，PaliGemma VLM + Flow Matching Action Expert）**。
