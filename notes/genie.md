# 精读笔记 · Genie 1

> arXiv 2402.15391 · ICML 2024 · DeepMind 26 人（Bruce, Clune, Rocktäschel, Singh 等）· 2024-02 · 项目页 genie-2024

## 基本情况

- 路线：**世界模型（交互式环境，动作无监督发现）**
- 一句话：第一个从无标注互联网视频学出的可交互世界模型——没人给按键标注，它自己从画面变化里"发明"了 8 个按钮
- 规模：11B（10.1B dynamics + 300M LAM + 200M tokenizer），256×TPUv5p，942B token
- 数据：Platformers 680 万个 16s 片段（3 万小时，2D 平台游戏）；机器人数据（RT-1 等 34 万 episode）全部当纯视频用、动作标签扔掉

## 三组件（全是 ST-transformer：空间注意力帧内 + 时间注意力跨帧，线性复杂度）

```
① 潜动作模型 LAM（300M）：(x_1:t, x_{t+1}) → ã_t
   VQ-VAE，词表 |A|=8；encoder 看历史+下一帧，decoder 只看历史+潜动作重建下一帧
   信息瓶颈：画面中"非惯性延续的变化"只能从 8 选 1 的码里来 → 码=按钮
   推理时整个丢弃，换成用户输入
② 视频 tokenizer（200M）：ST-ViViT，1024 码，时间感知（z_t 含全部历史信息）
③ 动力学模型（10.1B）：decoder-only MaskGIT，(z_1:t-1, ã_1:t-1) → ẑ_t
   交叉熵，训练时 0.5~1 随机 mask；bfloat16 + QK norm 稳训练
```

## 关键发现

- **潜动作跨输入语义一致**：平台游戏里某码是"向右"，换手绘草图/照片当初始帧，同码仍"向右"——"学这 8 个键就像拿到新手柄"
- 机器人数据上同样成立：up/down/left 一致（图 6）
- 涌现：视差（从平台游戏学会 3D 深度）、薯片袋形变、OOD 初始帧（Imagen2 图/草图/照片）照样可玩
- Scaling 无饱和：40M→2.7B loss 平滑降，batch 同样受益——"基础世界模型"叙事的证据

## 训 agent（证明潜动作真能用）

冻结 LAM 给 CoinRun 专家视频打标签 → 策略学预测潜动作 → 200 条真实动作样本学映射 → **追平 oracle BC**

## 消融

- LAM 吃**原始像素** > 吃 token（token 化丢动态信息，可控性掉）
- tokenizer：ST-ViViT > spatial-only ViT > C-ViViT（全时空注意过拟合+显存 1.6GB）
- 指标：FVD（保真）+ 自造 PSNR_diff（可控性：真潜动作 vs 随机潜动作的画面差）

## 谱系定位

```
World Models 2018（动作环境给）→ PlaNet/Dreamer（动作交互采）
  → Genie（动作从视频学）★ 本文
  → LAPA（潜动作预训练）→ DreamGen 数据管线 / GR00T 潜动作空间
  → Genie 2/3（产品化；Genie 3 无论文仅博客，2025-08）
```

- 对 Dreamer 系：解放数据（无需环境交互），但 8 键离散不够控 50Hz 机械臂
- 对 Cosmos：同为视频世界模型，Genie 的离散 token+离散动作为交互优化，Cosmos 连续 latent 为生成/模拟优化
- 对 Argon：动作接口的两种形态——Genie 是手柄（离散按钮），Argon+Omni 是语言（文字描述动作）
- 两问定位：预测离散视频 token / 给想玩或训 agent 的人当可交互沙盘 / 动作接口：学出来的
