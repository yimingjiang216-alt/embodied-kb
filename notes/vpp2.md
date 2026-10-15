# 精读笔记 · VPP2

> arXiv 2610.10270 · 星动纪元（Robotera）/清华/HKUST(GZ)/Berkeley/SJTU · 2026-10 · 代码+HF 开源

## 基本情况

- 路线：**WAM（视频世界模型 + 动作专家，解耦式）**
- 一句话："Predict Better, Act Better"——视频预测和动作学习分阶段解耦训练（vs DreamZero 的端到端联合）
- 基座：Wan2.1-I2V-14B 继续预训练；动作模块 0.9B DiT（MoT 架构）

## 三阶段训练

```
1. 事件级视频预训练：整段操作事件（首帧→完成），30k 步 batch 1024
2. chunk 后训练 + 一致性蒸馏：固定预测 8s（机器人）/2s（人手）；蒸馏到单步生成 0.12s
3. 动作专家：冻结视频 DiT 基础参数（只 LoRA 适配），动作专家吃视频 KV cache
```

- 解耦的理由（对 DreamZero 的批评）：动作梯度会破坏视频预训练表示

## 数据管线

- 三类：机器人（OXE/RoboMIND/DROID/AgiBot/自采 ALOHA）、人手（EgoDex/Ego4D/自采）、通用视频
- 切分 1-12s 语义片段（启发式 + VLM 精修）+ 详细 caption（明确哪只臂、哪个物体、视角）

## 结果

- 视频指令遵循：VPP2-14B 90% vs Cosmos3-64B 78%（14B 打赢 64B）
- 真机零样本 ALOHA：58.5% vs π0.5 40%（10 类任务 9 类最佳）
- RoboDojo 榜首（32.26%/39.26）；LIBERO-Pro 45.0 vs 最强基线 11.0
- 推理延迟 0.22s/chunk（视频 0.12 + 动作 0.1）

## 谱系定位

- WAM 内部的路线分歧：DreamZero（联合）vs VPP2（解耦+排序）——视频质量决定动作上限
- 与 π0.5 同款 MoT 思想（两套专家权重共享注意力）
