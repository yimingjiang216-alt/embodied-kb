# 精读笔记 · OpenVLA

> arXiv 2406.09246 · CoRL 2024 · Stanford/Berkeley/TRI/DeepMind/PI/MIT（Kim, Pertsch, Karamcheti, Finn, Levine, Liang 等 18 人）· 2024-06 · 代码 openvla/openvla

## 基本情况

- 路线：**VLA（LLM 派，离散 token 路线）**
- 一句话：第一个开源通用 VLA，7B 参数，97 万条真机演示，29 任务上超过 55B 闭源 RT-2-X **16.5 个点**，参数量少 7 倍
- 动机两个都是非技术的：① 现有 VLA 全闭源（RT-2/Covariant/Wayve）；② **没人研究过怎么高效微调**
- 价值：你库里的 VLA 全是 PI 一家 → 这篇补上"斯坦福/伯克利派"（Chelsea Finn、Sergey Levine 系）

## 架构

```
图像 + 指令 → [DINOv2 + SigLIP 特征拼接] → 2层MLP投影 → Llama 2 7B → 7维动作（离散token）
```

- **双视觉编码器**是关键：SigLIP 给语义（"是什么"），DINOv2 给空间/几何细粒度（"在哪"）——机器人控制特别需要后者
- 对比：π0 用 PaliGemma 自带 SigLIP 单编码器；GR00T 用 Eagle-2

## 动作表示：离散 token（vs π0 的连续 flow matching）

- 每维离散成 **256 bins**，变成 token，语言模型自回归预测，交叉熵损失
- 细节 ①：用 **1%~99% 分位数**而非 min-max —— 离群动作不会把区间拉宽
- 细节 ②：Llama 只有 100 个特殊 token 位不够 → **直接覆盖词表里最不常用的 256 个 token**
- 细节 ③：损失只算动作 token
- 代价（论文自认）：离散化有精度损失，狭窄高灵巧任务不如 Diffusion Policy 平滑

## 数据（97 万条）

- 50+ 数据集组成 OpenX，筛选：第三人称相机 + 单臂末端控制
- 混合权重直接用 **Octo 的**（多样性高的加权）
- **DROID 试了放弃**：动作 token 准确率一直低，最后 1/3 训练去掉

## 设计决策（几个反直觉结论）

| 决策 | 结论 |
|---|---|
| VLM 骨干 | Prismatic > LLaVA > IDEFICS-1（多物体语言 grounding 差距大） |
| 图像分辨率 | 224 vs 384 **无性能差异**，384 训练慢 3× → 选 224 |
| **视觉编码器微调** | **必须微调！**（VLM 领域常识是冻结更好，这里相反——预训练特征不够细粒度） |
| 训练轮数 | **比 LLM 多得多**，直到动作 token 准确率 >95% |
| 学习率 | 固定值，不用 warmup |

## 效率三件套（第二贡献，很实用）

**LoRA 微调**（Franka-Tabletop）：

| 策略 | 成功率 | 参数量 | 显存 |
|---|---|---|---|
| 全量 | 69.7% | 7188M | 163GB(2卡) |
| 只调末层 | **30.3%** | 465M | 51GB |
| 冻结视觉 | 47.0% | 6760M | 156GB |
| Sandwich | 62.1% | 914M | 64GB |
| **LoRA r=32** | **68.2%** | **98M(1.4%)** | **60GB** |

→ 只调末层只有 30%，说明**视觉特征必须适配目标场景**

**量化推理**：bf16 71.3% / **int8 58.1%** / **int4 71.9%**（显存 16.8→7.0GB）
- int8 掉的不是精度，是**慢到 1.2Hz**，偏离了训练的 5Hz 系统动态——时间尺度问题
- int4 快且准，**8GB 消费卡可跑 7B VLA**

## 与 π0 的正面对比（核心收获）

| 维度 | OpenVLA | π0 |
|---|---|---|
| 动作表示 | 离散 token (256 bins) | 连续 flow matching |
| 骨干 | Llama2 7B + 双编码器 | PaliGemma 3B + action expert |
| 损失 | 交叉熵（只算动作 token） | flow matching MSE |
| 推理 | 6Hz (4090) | 更高 |
| 微调 | LoRA 1.4% | 全量 |
| 定位 | **开源可复现、消费级** | **性能上限、工业级** |

## 谱系定位

```
RT-2（2023，闭源 55B）
  ↓ 开源 + 缩小
OpenVLA（7B，离散 token）← 本文
  ↓ 离散 token 优化
FAST（DCT+BPE）→ Wall-OSS 的 RVQ
  ‖（另一条路）
π0（连续 flow matching）→ π0.5 → π0.7
```

- **离散 token 派**：RT-2 → OpenVLA → FAST（+ Wall-OSS RVQ）
- **连续生成派**：π0 → GR00T → Wall-OSS flow 分支
- 价值观差异：π 系做"最强"，OpenVLA 做"所有人能用"——两条都有价值
