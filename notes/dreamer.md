# 精读笔记 · Dreamer (V1)

> arXiv 1912.01603 · ICML 2020 · Hafner, Lillicrap, Ba, Norouzi · 代码开源（danijar/dreamer）

## 基本情况

- 路线：**世界模型（latent 动力学，RL 分支）**
- 一句话：学一个 RSSM 世界模型，在想象里用 actor-critic 训策略，解析梯度回传
- RSSM 三件套：表示模型 p(s_t|s_{t-1},a_{t-1},o_t) / 转移模型 q(s_t|s_{t-1},a_{t-1}) / 奖励模型 q(r_t|s_t)
- latent 空间预测（不生成像素）→ 低内存，可并行想象数千条轨迹

## 关键点

- 想象 rollout H 步 → 算 V_λ(s_τ) → action model 梯度穿过 transition model 回传（解析梯度，非 REINFORCE）
- Actor-critic：action model 最大化 V_λ；value model 回归 V_λ
- 解决"有限想象视界的短视"：value model 估计视界之外的回报
- 对照：PlaNet 每步在线规划（慢）；Dreamer 训出策略直接用
- 消融：无 value model 的 action-only 在 acrobot/hopper 上崩（需要长期信用分配的任务）

## 谱系定位

- World Models 2018 → PlaNet（RSSM 缝合+规划）→ **Dreamer（换 C 的训法为 value 梯度）** → DreamerV2/V3
- 想象里训练 = Ha & Schmidhuber "梦里训练"的稳定化版本
