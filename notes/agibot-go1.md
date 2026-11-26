# 精读笔记 · AgiBot World & GO-1（百万真机轨迹与 ViLLA 隐式动作规划）

> arXiv 2503.06669 · 智元机器人（AgiBot World Colosseo Team / 彭志辉、罗剑岚等）· 2025-03 · 数据集与模型开源（OpenDriveLab/AgiBot-World）

## 基本情况

- 路线：**VLA + 隐式动作规划（ViLLA：Vision-Language-Latent-Action）+ 工业级真机数据基建**
- 两问定位：**① 预测什么空间？** 先由 Latent Planner 预测帧间**隐式动作 Token（Latent Action Tokens）**，再由 Action Expert 预测连续关节/末端控制量；**② 预测出来给谁用？** 隐式动作 Token 作为连接高层 VLM 语义理解与底层毫秒级连续动作头的“中间桥梁”
- 一句话：建成包含 **100 万+ 条真机双臂/移动操作轨迹、217 个长程任务、5 大真实场景（家居/超市/工业/餐厅/办公）** 的开源真机数据集 **AgiBot World**，并提出 **GO-1（Genie Operator-1）** 的 **ViLLA 架构**，让模型既能从海量无动作标注的人类/互联网视频里学操作先验，又能高频输出精细连续动作

## 核心设计（它修了前人的什么）

1. **真机数据质量的代际升级（对比 OXE / DROID）**：
   - Open X-Embodiment（OXE）虽然有 100 万条轨迹，但 90% 是单臂桌面短任务、相机无统一标定、无力控反馈；
   - AgiBot World 采用统一构型的轮式双臂人形（AgiBot G1，带双臂六维力传感器、灵巧手/夹爪、头部双目与腕部 RGB-D），100% 带有多视角时间同步、精细子任务分段标注与人类失误恢复轨迹（Failure-and-Recovery，专门教机器人在抓滑后怎么自己重抓）。
2. **ViLLA 架构（VLM + Latent Planner + Diffusion Action Expert 三级流水线）**：
   - **Stage 1：无监督隐式动作逆动力学模型（Latent Action IDM）**：借鉴 Genie 思想，给定第 $t$ 帧和第 $t+k$ 帧画面，用 VQ-VAE 自编码器无监督提取两帧之间的离散隐式动作码本（Latent Action Tokens），无需任何真机关节标注即可把海量人类视频转化为训练语料；
   - **Stage 2：Latent Planner（隐式规划专家）**：基于 VLM 主干，根据当前图像和语言指令自回归预测未来隐式动作 Token；
   - **Stage 3：Action Expert（连续动作专家）**：以 VLM 特征和 Latent Planner 预测的隐式动作 Token 为联合条件，用流匹配 / 扩散生成高频连续动作块（比纯 VLM 直接回归动作成功率平均高 **+32%**）。

## 谱系定位

- **把 Genie 的「无监督潜动作」真正塞进了工业级 VLA**：此前 Genie 1 只在 2D 游戏里学 8 个潜动作键，GR00T N1 只把潜动作当预训练辅助目标；GO-1 则让 **Latent Planner 在推理时显式输出潜动作 Token 喂给 Action Expert**，搭起了「互联网人类视频 $\to$ 隐式动作先验 $\to$ 真机物理动作」的完整桥梁。
