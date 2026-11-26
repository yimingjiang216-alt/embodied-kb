# 精读笔记 · BeyondMimic（人形机器人高动态全身动捕跟踪与引导扩散控制）

> arXiv 2508.08241 · UC Berkeley Hybrid Robotics（Qiayuan Liao, Takara E. Truong, Xiaoyu Huang, Guy Tevet, Koushil Sreenath, C. Karen Liu）· 2025-08 · 代码开源（HybridRobotics/whole_body_tracking）

## 基本情况

- 路线：**小脑全身控制（Humanoid Whole-Body Control）+ 测试时引导扩散策略（Guided Diffusion Policy）**
- 两问定位：**① 预测什么空间？** 29 自由度人形机器人（Unitree G1）的 50Hz 关节位置设定点序列；**② 预测出来给谁用？** 经 200Hz 低层 PD 执行器转化为关节力矩，在真实物理世界中复现 LAFAN1 等高动态人类动作（后空翻、凌空回旋踢、格斗、摔倒爬起）并同时响应高层任务目标（走到指定航点、避障、操纵物体）
- 一句话：定义了一套免逐动作调参的 **29-DoF Unitree G1 物理执行器建模与 8 项刚体/接触跟踪奖励标准**，并将训练好的强化学习全身跟踪专家蒸馏进统一的**状态-动作扩散模型（State-Action Diffusion Model）**，使得在测试时只需加一个可微损失梯度（Test-Time Classifier Guidance）就能实时操控人形机器人边打拳/跳舞边走向任意目标点

## 核心设计（SuperMimic 直接继承的物理与 MDP 底座）

1. **免魔法调参的物理等效执行器参数设计（Actuator-Aware PD Tuning）**：
   - 以往人形 Sim2Real 每个动作都要手调一套关节 $K_p, K_d$；
   - BeyondMimic 直接从电机转子惯量与减速比计算各关节等效反射惯量（Reflected Inertia / Armature）$I_{\text{arm}}$，按统一自然频率 $\omega_n = 10\text{ Hz}$（$20\pi\text{ rad/s}$）和过阻尼比 $\zeta = 2.0$ 解析算出每个关节的刚度 $K_p = I_{\text{arm}} \omega_n^2$ 与阻尼 $K_d = 2 \zeta I_{\text{arm}} \omega_n$，一套参数跑通全部高动态真机技能。
2. **朝向相对坐标系下的 14 刚体指数核奖励体系**：
   - 将参考动作旋转对齐到机器人当前骨盆偏航角（Heading Frame），在 14 个关键刚体（骨盆、躯干、双肩、双肘、双腕、双髋、双膝、双踝）上计算世界/局部相对 3D 位置、6D 旋转、线速度与角速度误差，经高斯指数核 $\exp(-\|e\|^2 / \sigma^2)$ 映射为有界稠密奖励——**这套 MDP 与奖励公式被后来的 `SuperMimic` 与 `BlinkMimic` 100% 原样继承**。
3. **测试时损失引导扩散（Loss-Guided Diffusion at Test Time）**：
   - 将多个 RL 跟踪策略蒸馏进一个联合预测未来状态与动作的扩散模型；推理时无需重新训练 RL，直接将高层任务目标（如“骨盆在 2 秒后到达坐标 $(x^*, y^*)$”或“避开半径 0.5m 障碍物”）写成关于预测状态的解析代价函数 $J(\hat{s})$，在 DDIM 去噪每一步加上 $-\nabla J(\hat{s})$ 引导梯度。

## 谱系定位

- **从 BeyondMimic 到 SuperMimic 的跃迁**：BeyondMimic 解决了**「G1 全身物理建模 + 奖励统一 + 下游任务引导」**，但它的底层教师策略依然靠无模型 PPO 在海量并行仿真里硬刷几千 GPU 小时；半年后同一团队（Qiayuan Liao 等）提出 **SuperMimic（2026-10）**，用 **1 个 8.82M 参数的可微物理世界模型 + 32 步想象一阶 BPTT** 替代了 PPO，将训练算力砍掉了 **1000 倍**。
