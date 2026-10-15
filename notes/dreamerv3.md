# 精读笔记 · DreamerV3

> arXiv 2301.04104 · Nature 2025 · Hafner 等 · 代码开源（danijar/dreamerv3）

## 基本情况

- 路线：**世界模型（latent 动力学，通用化）**
- 一句话：一套超参数征服 150+ 任务（Atari/ProcGen/DMLab/Control Suite/BSuite/Minecraft），首个从零挖到 Minecraft 钻石的算法
- RSSM：encoder 出离散 z（32 类 × 32 维 categorical，straight-through 梯度）+ 序列模型 h_t

## 五大稳定性技巧（核心贡献）

1. **symlog 预测**：symlog(x)=sign(x)·ln(|x|+1)，压大尺度目标，梯度不爆
2. **twohot critic**：回报切成指数间隔 bins，softmax 分类（多峰回报不被平均）——分布 critic
3. **free bits**：KL < 1 nat 时梯度置零，防表示坍塌/爆 spike
4. **回报归一化**：S=EMA(P95−P5)，只压大回报不放大噪声；熵系数固定 η=3e-4
5. **EMA target critic**：向自己的慢速副本回归

## 关键点

- 训练细节全可复用：离散 latent、1% 百分位归一化、categorical 直通估计
- Minecraft：稀疏奖励 + 长链（木镐→石镐→铁镐→钻石），无人类数据无课程
- 新架构很少，全是稳定性——"让一套超参跨域"是工程学的胜利

## 谱系定位

- Dreamer 系顶点；π*0.6 的分布式 value function（201 bins）与 twohot critic 同一思想
- "训练稳定性换通用性"的范式，与 π0.7 的"数据多样性+元数据"是两种通用化路线
