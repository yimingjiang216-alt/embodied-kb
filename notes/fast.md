# 精读笔记 · FAST

> arXiv 2501.09747 · Physical Intelligence · 2025-01 · 代码/HF 开源

## 基本情况

- 路线：**VLA 训练加速（动作 tokenizer）**
- 一句话：DCT 把平滑动作压成稀疏频域系数 + BPE 压成 token，让 VLA 用标准 next-token prediction 训练，快 5 倍
- FAST+ universal tokenizer：1M 真机轨迹训练，即插即用

## 关键点

- 问题：朴素 binning（每维每步 256 桶）在高频控制下失败——相邻 token 几乎相同，自回归模型学会复制就摆烂（800Hz 完全失败）
- 三步：DCT（时域→频域，平滑动作能量集中在前几个系数）→ 量化（×10 取整，后面全零丢弃）→ BPE（词表 1024）
- 900 浮点数 → ~30-50 token（压缩 20-30 倍）
- 交错排列（column-first：所有维度的低频在前）→ 自回归先定整体形状再补细节，rollout 更稳
- 与 flow matching 对比：训练快 5×、推理快（一次前向 vs 10 步去噪）、有损 vs 无损
- π0.5 预训练用 FAST、后训练切 flow matching——各取所长

## 谱系定位

- 解决"连续动作进不了自回归框架"的桥
- 后来成为 π0.6/π0.7 的 KI recipe 组件（离散动作 token 与连续 flow matching 并行预测）
