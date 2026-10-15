# 精读笔记 · V-JEPA 2

> arXiv 2506.09985 · Meta（Assran, Bardes 等 30 人）· 2025-06 · 代码开源（facebookresearch/vjepa2）

## 基本情况

- 路线：**世界模型（JEPA 表示空间预测，非生成式）**
- 一句话：两阶段——100 万小时视频无监督预训练（mask-denoising 学表示）+ 62h 无标签 DROID 后训练动作条件预测器 → 零样本规划
- 核心主张（对 LeCun 蓝图的兑现）：在表示空间预测，不生成像素——忽略不可预测细节（草叶怎么晃），容量集中在物理规律

## 架构

- 阶段 1：ViT-g 1B encoder + predictor，L1 损失对 EMA 目标，22M 视频（VM22M）
- 阶段 2（V-JEPA 2-AC）：冻结 encoder + 300M transformer（block-causal），teacher-forcing + rollout 双损失防误差累积
- 输入：4s 片段 256×256 @4fps；动作 = 末端执行器状态差分（7 维）

## 规划（测试时）

- 能量 E = ‖P(a_1:T; s_k, z_k) − z_g‖₁，CEM 采样 80×10 迭代优化
- **16 秒/动作**（vs Cosmos 规划 4 分钟）——表示空间 rollout 快 15 倍
- receding horizon：只执行第一步，重新规划

## 结果

- Reach 100% / Grasp 65%（杯）/ Pick-and-Place 80%（杯）
- Scaling 四干预累积 +4.0：数据 +1.0、模型 +1.5、训练时长 +0.8、分辨率/时长 +0.7
- Progressive resolution training（256→384 cooldown）省 8× 算力

## 谱系定位

- JEPA 分支的机器人落地；对"视频生成=世界模型"的正面反驳
- 缺动作接口的批评不成立（AC 版有）；但规划慢于 WAM 的直接出动作
