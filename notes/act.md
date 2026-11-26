# 精读笔记 · ACT（Action Chunking with Transformers）

> arXiv 2304.13705 · RSS 2023 · Stanford / UC Berkeley（Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn）· 代码开源（tonyzhaozh/act, LeRobot 内置）

## 基本情况

- 路线：**模仿学习底座（VLA 动作分块的共同鼻祖）**
- 两问定位：**① 预测什么空间？** 机器人未来 $k$ 步连续关节位置序列（Action Chunking，$k=100$ 即 2 秒动作块，不预测未来画面）；**② 预测出来给谁用？** 直接经时序加权融合（Temporal Ensembling）发给底层位置控制器执行
- 一句话：用 2 万美元的低成本双臂遥操作硬件（ALOHA）+ **CVAE Transformer 一次预测未来 $k$ 步动作块**，在撕胶带、穿鞋带、装电池等精细双臂任务上把单步行为克隆的成功率从 **1%~20% 拉到 80%~96%**

## 核心设计（为什么单步 BC 会崩，而 ACT 能成）

1. **复合误差（Compounding Error）与动作分块（Action Chunking）**：
   - 传统单步行为克隆 $\pi(a_t | o_t)$ 每一步微小的预测偏差都会把机器人推离演示轨迹分布，进入未见状态后误差雪崩；
   - ACT 将决策周期从 1 步拉长到 $k$ 步（$\pi(a_{t:t+k} | o_t)$），把任务的有效时间跨度（Effective Horizon）缩短为原来的 $1/k$，同时天然克服人类遥操作演示中的“停顿犹豫”（非马尔可夫性）。
2. **时序集成（Temporal Ensembling）消除块边界抖动**：
   - 若每 $k$ 步才推理一次，两个动作块交界处会出现明显顿挫；
   - ACT 每一步都查询一次策略， sehingga 当前时刻 $t$ 会同时收到来自 $t, t-1, \dots, t-k+1$ 共 $k$ 个历史预测块的动作估计，按指数衰减权重 $w_i = \exp(-m \cdot i)$ 加权平均，既平滑又保留闭环纠偏能力。
3. **CVAE 隐变量 $z$ 解决人类演示的多模态分布**：
   - 同一个任务人类有时从左边绕、有时从右边绕，直接用 L1/L2 损失回归会让网络输出“左右平均”的错误动作；
   - 训练时用 BERT 式编码器把当前关节状态 + 真实动作序列压入高斯分布潜变量 $z \in \mathbb{R}^{32}$（加 KL 散度正则），解码器以 $(z + \text{多视角 ResNet18 特征} + \text{当前关节角})$ 生成动作块；**推理时直接令 $z = 0$（高斯均值）**，实现确定性快速推理。

## 关键遗产与谱系定位

- **所有现代 VLA 动作头的共同标准**：后来的 $\pi_0$（50Hz 出 50 步动作块）、OpenVLA-OFT、GR00T、Wall-OSS（32 步动作块）全部沿用了 ACT 确立的 **Action Chunking + 多视角图像联合注意力** 范式；
- **与 Diffusion Policy 的分工对比**：ACT 用 **CVAE + L1 损失** 单次前向出动作（80M 参数，推理只需几毫秒，50Hz 极轻快）；Diffusion Policy 用多步迭代去噪拟合更复杂的多峰分布（表达力更强但推理慢 10×），这一对张力后来演变为 $\pi_0$ 的 Flow Matching（10 步 ODE）与 FAST（单步离散 Token）。
