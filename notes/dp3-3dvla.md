# 精读笔记 · DP3 & 3D-VLA（三维空间表征如何赋能具身操作）

> DP3: arXiv 2403.03954 · RSS 2024 · 清华大学 / 上海期智研究院（Yanjie Ze, Gu Zhang, Kangning Zhang, Chenyuan Hu, Muhan Wang, Huazhe Xu）
> 3D-VLA: arXiv 2403.09631 · ICML 2024 · UMass Amherst / MIT-IBM / UCLA（Haoyu Zhen, Xiaowen Qiu, Peihao Chen et al.）

## 基本情况

- 路线：**3D 空间感知 × 生成式操作策略（从 2D 像素策略走向显式 3D 几何策略）**
- 两问定位：**① 预测什么空间？** DP3 基于单视角三维点云预测 3D 连续动作轨迹；3D-VLA 同时预测 **3D 目标点云/RGB-D 目标图（世界模型）** 与 **7-DoF 机器人动作 Token**；**② 预测出来给谁用？** 将三维几何不变性直接注入策略网络，解决 2D 视觉策略“换个相机角度、换块桌布颜色就全崩”的过拟合顽疾
- 一句话：**DP3（3D Diffusion Policy）** 证明只需 **单视角深度图转精简三维点云 + 极简 MLP 点云编码器 + 扩散动作头**，在仅用 **10 条演示数据** 的情况下就比 2D Diffusion Policy 平均成功率高 **24.2%**（训练快 5×），且天然对相机视角旋转、背景纹理干扰具备零样本鲁棒性；**3D-VLA** 则将 3D 场景理解、3D 目标生成扩散模型与动作规划统一进 3D 大语言模型

## 核心设计（为什么 3D 表征比 2D 图像省数据、抗干扰）

1. **DP3 的三个反直觉发现**：
   - **单视角点云足够，无需多相机拼接**：只需一个标定好的第三视角 RGB-D 相机，经工作空间裁剪（Crop 掉背景墙壁）和最远点采样（FPS 降采样至 512~1024 个点），即可提供无歧义的空间尺度与遮挡关系；
   - **去掉点云颜色（只留 XYZ 坐标）泛化反而更强**：2D 策略和带 RGB 的点云策略极易死记训练时的桌面颜色和光照；只输入纯三维坐标 $(x, y, z)$ 让策略被迫只关注物体几何形状与相对位置，换背景/换物体颜色时成功率几乎不掉；
   - **极简三层 PointNet 击败巨型预训练 3D 主干（PointNeXt / PointMAE）**：复杂的 3D 预训练编码器在少样本（10 条演示）模仿学习中反而容易过拟合，轻量级 3 层 MLP + MaxPooling 配合条件扩散头效果最优。
2. **3D-VLA 的「先想象三维未来、再输出动作」闭环**：
   - 在 3D LLM（基于点云特征注入）中引入 `<scene>`、`<loc>`（3D 包围盒标签）与 `<act>` Token，并通过对齐训练接上 RGB-D 与点云扩散解码器：接到指令后先预测操作完成后的未来 3D 场景状态（Embodied Diffusion World Model），再以想象出的 3D 目标状态为条件输出动作。

## 谱系定位与项目衔接

- **连接你的 [`mcp-3dgs`](https://github.com/yimingjiang216-alt/mcp-3dgs) 与 [`pi05-libero-sim`](https://github.com/yimingjiang216-alt/pi05-libero-sim)**：2D VLA（如 OpenVLA、$\pi_0$）强在开放词汇语义但缺显式三维几何与深度鲁棒性；3D 策略（DP3）强在少样本几何泛化但缺开放词汇语义。将 **Semantic 3DGS（同时具备 384 维开放词汇语义场 + 精确度量深度与 ESDF 障碍物场）** 作为具身大脑的空间表征，正好把两派优势合二为一。
