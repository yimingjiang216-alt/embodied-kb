# 精读笔记 · World Models (2018)

> arXiv 1803.10122 · Ha & Schmidhuber（Google Brain / NNAISENSE）· 2018-03 · 交互式论文 worldmodels.github.io

## 基本情况

- 路线：**世界模型的现代起点（V-M-C 三件套）**
- 一句话：VAE 压视觉 + MDN-RNN 概率预测未来 + 1088 参数线性控制器；策略可以完全在模型自己生成的"梦"里训练再搬回现实
- 参数量：VAE 4.4M + MDN-RNN 1.7M + Controller 1088（小三个数量级）

## V-M-C

- V：VAE，每帧 → 64 维 z（Doom）/ 32 维（赛车）
- M：MDN-RNN（256 LSTM），P(z_{t+1} | a_t, z_t, h_t)——**输出混合高斯分布**，能表达离散随机事件（怪物开火/不开火）
- C：a_t = W_c[z_t h_t] + b_c，CMA-ES 进化策略搜索
- 三步完全分开训；世界模型全程不知道奖励

## 两个实验

- **赛车 CarRacing-v0**：首个达到官方"解决"标准（100 连续试验 900+）；世界模型当特征提取器
- **VizDoom Take Cover**：给 M 加死亡预测头 → 完整"游戏引擎替身" → **梦里训练**（918 分）→ 搬回现实（1092 分 > Gym 榜首 820）
- 温度 τ 控制梦的混沌：τ=0.1 模式崩溃（怪物永不开火，梦里 2086 分现实 193 分）；τ=1.15 最优
- "在嘈杂的噩梦里活下来的 agent，会在干净的现实里茁壮成长"

## 作弊章节（§4.5，最有预见性）

Agent 发现对抗策略：特定移动让梦里的怪物一颗火球都不发。三层分析：
1. 概率近似 ≠ 物理（"孩子知道东西会落地，也会想象飞的超人"）
2. C 能读 M 的隐状态 = 直接改游戏引擎内存
3. 邪道把 agent 带到训练分布外（模型在那里乱预测）

→ 预言了后世全部主线：Cosmos 的 physics alignment 评测、DreamerV2 的 KL balancing、V-JEPA 2 的 teacher-forcing，都在治这三样

## 谱系定位

```
Craik 1943 → 【本文】→ PlaNet（V+M 缝成 RSSM）
                      → Dreamer（C 换 value 梯度）→ V3
                      → LeCun 蓝图 → V-JEPA 2
                      → UniPi → Cosmos → DreamZero/VPP2
```
读过的每篇世界模型论文都在修这篇的某个零件
