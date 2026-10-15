# 精读笔记 · GR00T N1

> arXiv 2503.14734 · NVIDIA · 2025-03 · 代码/权重/数据全开（Isaac-GR00T）

## 基本情况

- 路线：**VLA（人形机器人基础模型）+ 数据金字塔**
- 一句话：双系统架构（Eagle-2 VLM 当 System 2 + DiT flow matching 当 System 1）+ 三层数据金字塔（88h 真机 → 827h 神经轨迹）
- System 2：Eagle-2 VLM（SmolLM2 1.34B + SigLIP-2），10Hz，取**第 12 层**（中间层）特征——更快且下游更好
- System 1：DiT flow matching（4 步去噪，16 步动作块），cross-attention 吃 VLM 特征，120Hz
- 损失与 π0 同款：p(τ)=Beta((s−τ)/s; 1.5, 1)

## 数据金字塔

```
真机遥操作 88h（最贵最少）
  ↓ 视频生成模型（Cosmos/CogVideoX/Wan2.1）合成神经轨迹 → 827h（10×）
仿真轨迹（DexMimicGen）
人类视频 + VLM 预训练数据（最多最便宜）
```

- 神经轨迹 = 视频生成 + VQ-VAE 提取 latent action（统一潜动作空间）
- 跨本体：单臂/双臂/人形灵巧手，embodiment-aware 状态/动作编码器

## 谱系定位

- NVIDIA 的 VLA 答卷；双系统 = Kahneman 快慢思考的机器人版
- 数据金字塔中间层就是 DreamGen 方法论（NVIDIA GEAR 自己的论文）
- DreamZero 论文里的对比基线 GR00T N1.6 是它的后继
