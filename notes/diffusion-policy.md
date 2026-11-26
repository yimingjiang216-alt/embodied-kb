# 精读笔记 · Diffusion Policy

> arXiv 2303.04137 · RSS 2023 / IJRR · Columbia / Toyota Research / MIT（Cheng Chi, Siyuan Feng, Yilun Du,jiajun Wu, Russ Tedrake, Shuran Song）· 代码开源（real-stanford/diffusion_policy）

## 基本情况

- 路线：**生成式策略底座（将图像扩散模型引入机器人连续控制）**
- 两问定位：**① 预测什么空间？** 未来 $T_p$ 步连续动作序列的噪声梯度场 $\nabla_a \log p(a | o)$（在动作空间做条件 DDPM/DDIM 去噪，不生成图像像素）；**② 预测出来给谁用？** 采用滚动时域控制（Receding Horizon Control，预测 $T_p=16$ 步、执行前 $T_a=8$ 步）直接驱动机械臂
- 一句话：把机器人视觉运动策略建模为**条件去噪扩散过程（Conditional Denoising Diffusion Process）**，在 4 大基准 15 项操作任务上平均比此前 SOTA（IBC、BET、LSTM-GMM）提升 **46.9%**，奠定了“生成模型做连续控制”的主流范式

## 核心设计（它修了前人的什么）

1. **彻底解决多模态演示的「模式平均」与「能量模型失稳」**：
   - 显式回归策略（MSE / 高斯混合）遇到左右两条对称绕障轨迹时会平均成直撞障碍物；隐式能量模型（IBC）靠负采样近似配分函数，高维动作空间下训练极易发散；
   - Diffusion Policy 通过学习分数函数（Score Function）$\varepsilon_\theta(o_t, a_t^k, k)$，经过 $K$ 步（DDIM 可压至 10~16 步）随机朗之万动力学迭代去噪，每次采样自然落入某一个干净的模式分支。
2. **滚动时域控制（Receding Horizon Control）兼顾时序连贯与闭环响应**：
   - 历史观测窗口取 $T_o=2$ 帧，一次去噪生成未来 $T_p=16$ 步动作序列，但只执行前 $T_a=8$ 步就重新根据最新视觉观测规划——既防止单步动作高频抖动，又避免开环走完 16 步导致的累积漂移。
3. **1D 时序卷积（CNN）vs. Time-Series Diffusion Transformer（DiT）两套骨干对比**：
   - **CNN 版（FiLM 条件注入）**：开箱即用、对超参不敏感，但遇到高频剧烈变化的动作容易过平滑（低通滤波倾向）；
   - **Transformer 版（交叉注意力注入观测）**：擅长复杂多阶段与高频动作变化，但对学习率和正则化敏感——为后来 $\pi_0$、RDT-1B 采用 Transformer 动作专家提供了直接实证。

## 关键遗产与谱系定位

- **从 Diffusion Policy 到 $\pi_0$ 的演化线**：`Diffusion Policy (2023，小模型 + DDPM/DDIM 动作头)` $\to$ `Octo / RDT-1B (2024，跨本体预训练 + 扩散动作头)` $\to$ `π0 (2024-10，把 3B 预训练 VLM 的 KV Cache 通过 MoT 喂给 Flow Matching 动作头)`；
- **位置控制 vs. 速度控制的实证结论**：论文实验证明，在扩散策略下**预测绝对末端/关节位置设定点（Position Control）显著优于预测速度增量（Velocity Control）**，因为速度预测在多步积分中更容易累积漂移。
