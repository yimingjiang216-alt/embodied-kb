# 精读笔记 · PlaNet

> arXiv 1811.04551 · ICLR 2019 · Hafner, Lillicrap, Fischer, Villegas, **Ha**, Lee, Davidson（Google Brain / DeepMind）· 2018-11 · 代码 danijar.com/planet

## 基本情况

- 路线：**世界模型（latent 动力学，规划分支）**
- 一句话：把 World Models 的 V+M 缝成 RSSM 联合训练；**不学策略网络**，每步用 CEM 在 latent 空间搜动作
- 结果：6 个像素级连续控制任务，比 D4PG 少 200× 数据达同级性能；cheetah 超 D4PG 26%

## RSSM（核心贡献：那个"缝合"）

```
确定性路径：h_t = f(h_{t-1}, s_{t-1}, a_{t-1})   ← GRU 200 单元，记忆
随机路径：  s_t ~ p(s_t | h_t)                    ← 30 维对角高斯，分支
观测模型：  o_t ~ p(o_t | h_t, s_t)               ← 只训练用（重建）
奖励模型：  r_t ~ p(r_t | h_t, s_t)               ← 规划的评分
```

- 动机：World Models 的 V（VAE）和 M（RNN）分开训、目标不一致——z 好不好重建 ≠ z 好不好预测；缝合后用 ELBO 联合训练
- 消融：纯 RNN 不行（表达不了多种未来、易被规划器作弊）；**纯 SSM 完全学不起来**（采样噪声洗掉记忆）；两条路都要
- 确定性模型可被规划器钻空子——World Models"作弊"问题的架构根源，RSSM 用随机性治

## 规划：无策略 agent

- CEM（J=1000 采样 → top-K=100 → 拟合高斯 → 迭代 I=10 轮），latent 空间 rollout 不生成图像
- MPC：每步重新规划（新观测进来从头初始化分布，避免局部最优）
- 数据收集循环：种子随机 episode → 训模型 → 用模型+CEM 跑新 episode → 入库再训

## Latent Overshooting（第二贡献）

- 标准 ELBO 只约束单步预测；overshooting 对跨度 d=1..D 全部加 KL 约束
- 改善长程视频预测质量，但对规划性能增益在误差内（论文自认）

## 实验

| 方法 | Episodes | Cheetah | Walker |
|---|---|---|---|
| D4PG | 100,000 | 524 | 968 |
| **PlaNet** | **1,000** | **662** | 951 |
| CEM+真模拟器 | 0 | 656 | 994 |

- 消融三连：RSSM>RNN>SSM；CEM>随机采样；规划收集>随机收集
- 单 agent 学全 6 任务（从图像推断任务）——多任务世界模型最早演示

## 谱系定位

```
World Models 2018（V/M/C 分开训）
  ↓ 缝合 V+M，扔掉 C
PlaNet（RSSM + CEM 规划）← 本文
  ↓ CEM 每步太慢 → 学 value function（本文讨论章节明确递了纸条）
Dreamer 2020 → DreamerV3
```

- V-JEPA 2 的规划 = PlaNet 的 CEM，只是 latent 换成 JEPA 表示
- 治好：V/M 目标不一致、控制器太小；没治：CEM 算力贵（Dreamer 的动机）、梦里训练稳定性（V3 才解决）
