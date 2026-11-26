# 精读笔记 · X-Planner & X2-Turn（自变量机器人：事件级大脑规划与 80ms 实时语音话轮预测）

> X-Planner: arXiv 2609.25187 · 自变量机器人（X Square Robot Team）· 2026-09 · 权重与基准开源（x-square-robot/X-Planner-9B-0916）
> X2-Turn: arXiv 2608.10878 · 自变量机器人（Kaiqi Fu, Rime Wen, Qian Wang et al.）· 2026-08 · 权重与在线 Demo 开源（x-square-robot/X2-Turn-4B-0812）

## 基本情况

- 路线：**具身认知架构的「大脑规划层（X-Planner）」与「实时语音人机交互入口（X2-Turn）」**
- 两问定位：
  - **X-Planner（9B，基于 Qwen3.5-9B）**：**① 预测什么空间？** 结合多视角视频与历史执行状态，预测下一个**动作接地的结构化事件（Action-Grounded Event）或紧凑潜规划状态（Latent Planning State）**；**② 预测出来给谁用？** 喂给下游世界动作模型（如 `Wall-WM 2.0` / `Wall-OSS 1.0`），让长程任务中间的规划状态显式可查、可干预；
  - **X2-Turn（4B，基于 Voxtral-Mini-4B-Realtime）**：**① 预测什么空间？** 每 **80ms** 同步输出中英双语流式语音转写（ASR）+ **6 类话轮状态标签（Turn State）**；**② 预测出来给谁用？** 供语音智能体实时决定“该继续安静听、该立刻回话、该忽略用户的随口附和、还是允许用户打断（Barge-in）”。

---

## 一、X-Planner：解决长程操作中「显式文本规划太慢、隐式端到端不可控」的两难

1. **双模式互补接口（Event Mode vs. Unified Mode）**：
   - **可读事件模式（Readable Event Mode）**：输出结构化自然语言事件状态（含当前阶段完成度判断、下一子目标描述與异常恢复指令），人类工程师与安全监控模块可直接审阅或拦截；
   - **统一潜状态模式（Unified Mode + Staircase Decoding）**：通过**阶梯式解码（Staircase Decoding）** 生成紧凑的连续潜规划向量直接注入下游世界动作模型，消除自然语言逐词自回归带来的几百毫秒停顿。
2. **XPlanner-Benchmark（1,500 集、3,490 段多视角长程视频）**：
   - 专门评测长程多阶段桌面与移动操作中的子目标切分、遮挡恢复与跨阶段状态追踪。

---

## 二、X2-Turn：解决实时语音 Agent「抢话、迟钝、被“嗯/对”误打断」的核心痛点

1. **为什么传统 VAD（静音检测）做不好实时语音交互？**
   - 传统语音助手靠“静音超过 800ms 算说完”：用户思考停顿时会被机器人粗暴抢话；用户说完了却要傻等 800ms 才有反应；而在机器人正在说话时，用户随口说一句“嗯、对、好的”（Backchannel 附和），传统 VAD 会误判为用户打断（Barge-in）而立刻闭嘴中断任务。
2. **80ms 帧同步双头架构（Shared Backbone + ASR Head + Independent Turn Head）**：
   - 在共享音频-文本骨干上并联两个独立输出头：一个头流式输出中英混合 ASR 文本，另一个头（`vad_lm_head`，映射至词表保留 ID 35–40）每 **80ms** 预测一次 6 类话轮状态：
     - `idle`（无有效语音）、`noidle`（有声学活动但意图未明）、`speaking`（语义未完，用户仍在说）、`turn_end`（语义与韵律均完整，Agent 可立即接话）、`backchannel`（“嗯/对/okay”简短附和，Agent 不应中断当前播报或动作）、`uncertain`（推理态低置信度回退，继续等待）。
3. **关键实测数字（EasyTurn 基准）**：
   - 中文话轮结束识别准确率 $\text{ACC}_{\text{comp}} = \mathbf{91.0\%}$（对比基线 SoulX-Duplug `77.67%` 提升 **+13.3 点**），附和词识别率 **96.0%**，端到端响应延迟从 `480ms` 档压至最低 **`120ms`（中文）/ `65ms`（英文）**。

## 谱系定位

- 这两篇与 `Wall-OSS 1.0 Lite`、`Wall-WM 2.0` 合在一起，构成了自变量机器人（X Square Robot）完整的 **「耳（X2-Turn 80ms 实时语音话轮）$\to$ 脑（X-Planner 9B 事件/潜状态规划）$\to$ 梦（Wall-WM 2.0 视频世界模型）$\to$ 手（Wall-OSS 1.0 Lite 32 步残差精炼 VLA）」** 四层全栈开源版图。
