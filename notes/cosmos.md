# 精读笔记 · Cosmos

> arXiv 2501.03575 · NVIDIA（79 人）· 2025-01 · 代码/模型开源（cosmos-predict1，NVIDIA Open Model License）

## 基本情况

- 路线：**世界模型平台（视频世界基础模型，动作进视频出）**
- 一句话：给 Physical AI 造"物理世界的数字孪生"——20M 小时视频预训练通用 WFM，机器人公司拿去微调
- 定义（图 3）：W(x_{0:t}, c_t) → x̂_{t+1}，c_t 可以是动作/文本/相机轨迹
- 不是策略：**动作是输入，视频是输出**——模拟器不是控制器

## 四大件

1. **数据管线**：TransNetV2 切镜头 + 四道过滤（运动/画质砍 15%/文字/类型剔除游戏动画）+ VILA-13B 标注（FP8 TRT-LLM 10×）+ 语义去重删 30% → 20M 小时 → 1 亿片段
2. **Tokenizer**：因果（图像=视频 T=0）、小波打头、连续版 AE（无 KL）/离散版 FSQ（词表 64000）；光流损失治时间抖动、Gram 治糊；4×8×8 全面 SOTA，每帧 34.8ms（比 CogVideoX 快 12×）
3. **双路线 WFM**：扩散（DiT + EDM + 不确定性加权 u(σ)）和自回归（类 LLM + YaRN + z-loss）并行押注；AdaLN-LoRA 砍 36% 参数
4. **护栏**：输入/输出双向（关键词 + Aegis + 人脸模糊）

## 后训练三线

- 相机控制（Plücker 坐标，COLMAP 验证 3D 一致性）
- 机器人：Cosmos-1X（EVE 人形 200h）指令视频预测 78.3% vs VideoLDM 13%；Bridge 动作条件下一帧（动作 MLP 嵌入进 DiT 时间步）
- 自动驾驶：六相机 2 万小时，view embedding + view-dependent cross-attention

## 谱系定位

- 定位是离线模拟器（10-60s 生成），把实时留给后辈：V-JEPA 2 用它规划要 4 分钟/动作 → DreamZero 38× 加速 → VPP2 0.22s
- §2.1 列了五个用途（评估/初始化/RL 环境/规划/合成数据）但注明"都没做实验"——后来 V-JEPA 2 做了规划、DreamGen 做了合成数据
