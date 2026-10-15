# 精读笔记 · Gemini 4 Argon（博客，非论文）

> Google Developers Blog · 2026-10-13 · Kanishka Rao · 无论文，产品发布 · 材料 papers/gemini4-argon/

## 定性（重要）

- **Argon 是通用多模态大模型（gemini-4-argon），不是世界模型，不是机器人策略**
- 博客里的世界模型是 Gemini Omni（视频世界模型）和 MuJoCo（物理引擎）——Argon 是它们的**操纵者**
- 11 个案例全是精选演示，无失败率、无统计；唯一量化：仿真数据把真机 56%→75%、35s→24s（VLA 是谁、多少试验未写）
- ERQA 基准 87% 是"看图答物理题"，不等于能操作

## 机制：Code-as-Policies 的 2026 版

```
图像+指令 → [Argon] → 写代码/curl 命令 → 执行 → 看新图像 → 循环（秒级一轮）
```

- 提示词里给了一套 robot.grasp/place/goto 原语；每轮一句观察+一条命令
- grasp_failed 后自己 look_at 重新观察（闭环自我纠错）
- 和 π0 的对照：端到端毫秒级 vs 写代码多轮秒级；策略是学出来的 vs 现场写的程序

## 11 个案例（按机制分类）

- 控制 3：Spot 踩垃圾桶（MuJoCo 数字孪生预演+2-agent 验证 swarm）、零样本机械臂、**Omni 控制**（写时间戳提示词迭代操纵 Gemini Omni 视频 rollout，33 步拼 ARGON）
- Real2Sim 3：视频/人类第一视角/手绘草图 → MuJoCo 孪生（提示词严禁作弊：不许外力、接触力必须来自求解器）
- 理解 3：事后标注（ALOHA 折纸 136s → 4 层标注+失误降权 0.15-0.25×，即 π0.7 质量元数据）、12 球天平、**稠密 value function**（UMI 视频→28 段进度曲线，即 π*0.6 value function）
- 自我改进 3：MuJoCo 造 4500 episode 增强 VLA、SO-101 迭代自我优化（58.9→22.9→17.4s）、一条提示复现 68 篇论文

## 谱系定位

- Google 押的是"通用大模型当机器人研发全栈助手"，不是"更强的端到端策略"——和 π 系（一个模型包办）两条路线
- 案例 7/9 正好是 π0.7 元数据和 π*0.6 value function 的"通用模型零样本版"——对专用组件路线的挑战（但无对比实验）
