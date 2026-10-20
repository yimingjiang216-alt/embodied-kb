# 精读笔记 · NaVid

> arXiv 2402.15852 · RSS 2024 · Zhang, Wang, Xu, Zhou, Hong, Fang, Wu, Zhang, Wang（北大/智源/阿德莱德/ANU/Galbot）· 2024-02 · 代码 jzhzhang/NaVid-VLN-CE

## 基本情况

- 路线：**VLN（VLM 时代的代表作）**
- 一句话：第一个用**视频 VLM**做 VLN 的工作——只要单目 RGB 视频流 + 指令，**不用地图、不用里程计、不用深度**，直接输出下一步动作
- 结果：VLN-CE R2R Val-Unseen **SR 37.4% / SPL 35.9%**，超过所有用了全景+深度+里程计的方法
- 真机：约 **66%** 成功率（200 次试验），**跨数据集**和 **Sim2Real** 双强

## 核心主张：三无

```
没有地图、没有里程计、没有深度
输入：单目 RGB 视频流（在线）+ 语言指令
输出：下一步动作（语言学形式的动作 token）
```

**为什么这是对的**（论文的论证）：
- 模拟了**人类怎么导航**——人也不靠精确里程计
- **天然绕过 odometry 噪声**问题
- **绕过地图/深度输入带来的 Sim2Real gap**——深度估计在仿真和真实里差别很大，不用就不存在这个 gap

## 架构

```
视频帧 → EVA-CLIP 视觉编码器 → Q-Former 查询生成器
                                   ├─ instruction-queried tokens（指令相关的视觉特征）
                                   └─ instruction-agnostic tokens（细粒度全局视觉）
                                        ↓
                    与指令 token 拼接 → LLM（Vicuna-7B）→ 动作（语言形式）
```

- 基座：**LLaMA-VID**（视频 VLM）
- 每帧两种 token：指令查出来的 + 指令无关的——**这个设计是它比 LLaVA/LLaMA-VID 原生版强的关键**
- 动作输出成**自然语言**（如 "forward"），而非数值向量

## 数据

- **510k 导航样本**（连续环境采集，含 action-planning 和 instruction-reasoning 两类）
- **763k 网页数据**（大规模文本/图像 QA，保住 VLM 的通用能力）
- 训练流程：先 pre-dagger（330k）→ 再 DAgger 扩到 510k

## 结果

**VLN-CE R2R Val-Unseen**（Table I）：
- NaVid 只用 RGB，**SR 37.4% / SPL 35.9%**
- 对比方法全都用了全景（Pan.）、立体 RGB、深度或里程计（Odo.）
- **只用最少的输入，拿到 SOTA**——这是全文最硬的论点

**跨数据集（VLN-CE RxR 零样本）**：同样最优

**消融**：
- 去掉视频（用单帧）→ SPL 从 35.9 掉到 20.8 → **视频历史是核心**
- LLaVA-Nav、LLaMA-VID-Nav 直接微调都远不如 NaVid → 说明"指令查询 token + 无关 token"的设计是必要的

**真机**：Turtlebot4 上约 66% 成功率

## 谱系定位

- 两问：**不预测**（端到端，但输出语言形式动作）/ **直接控制**
- 在 VLN 线上的位置：

```
2020 VLN-CE     ：GRU + 注意力，深度的命根子（去深度 20%→4%）
2024 NaVid      ：视频 VLM，连深度都不要了（只用 RGB 反而更好）
```

**这个反差很有意思**：VLN-CE 时代"没有深度就学不会"，NaVid 时代"不用深度反而更泛化"——**因为 VLM 的预训练视觉表征自带了空间理解**，不再需要显式深度。这就是基础模型带来的范式转移。

- 与你的操作线对照：NaVid（Vicuna-7B + Q-Former）和 π0（PaliGemma + action expert）**是同一技术代际**，但 NaVid 输出语言 token，π0 输出连续动作——**离散 vs 连续之争在导航线也重演了**
