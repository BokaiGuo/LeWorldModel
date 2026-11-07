# 原生相机动作后果对照：结果

协议：`PC_REAL_CAMERA_EFFECT_DEV_PROTOCOL_20261106_CODEX.md`。

## 已完成的真实执行

4 个已暴露 episode（1880–1883），每个取帧 0、5，三个动作、两个原生相机位置，共 48 条轨迹。不是随机背景变化或图像后处理。相机 lookat 从 (3,2.5) 改为 (3.5,2.5)，保持俯视角和距离不变。

- 8 个状态的两视角物理轨迹最大差均为 **0**。
- 相机图像像素 MAE 为 **20.981–21.155 / 255**；图像检查确认场景与物体一起随相机平移。
- 每环境步 0.05 秒，五步链 0.25 秒，逐时刻保存状态、RGB、接触数与物理时间。
- 初始 qpos/qvel 与选定 archive 状态一致；保留原速度，零动作不等于静止。
- 第二个独立进程重放全部 48 条轨迹：保存的状态、图像、接触数和时间全部逐元素一致。render 前后状态与物理时间不变。
- 相机、几何接触与 packet 时钟相关测试合计 **11 passed**。单元测试使用合成日志，仅验证检查器，不作为科学证据。

## 不得扩大解释

这是相机变化与物理后果解耦的认证，不是学习表征跨视角不变性、预测质量或搜索优势。帧对不是左右接触对照；PointMaze 相机实验与 PushT 合法接触实验也不能拼成同环境的完整因子实验。移动相机可能超出训练分布。四个既有 episode 不是独立确认。

本地 PointMaze JEPA 配置包含 proprio 特征（`proprio_emb_dim=16`）；原生数据集将完整 state 作为 proprio。不能为了接通模型，默默把此实验的评价端 qpos/qvel 输入视觉-only 方法。接预测前须显式认证观测权限；不能以补零或新训练适配器冒充原协议。

PC0–PC5 不改判。当前没有建立可进入正式确认的冻结规划候选。

## 产物

- `le-wm-main/experiments/pc_real_camera_effect_dev_20261106_codex.py`
- `le-wm-main/experiments/pc_real_camera_effect_audit_20261106_codex.py`
- `le-wm-main/results/pc_real_camera_effect_dev_20261106_codex.json`
- 同名前缀 `.npz`、`_audit.json`、`_views.png`

预览图首个 episode，行分别为帧 0 与 5，列为两个相机位置；按预定编号选取，未按效果挑选。

![原生双相机视图](../le-wm-main/results/pc_real_camera_effect_dev_20261106_codex_views.png)
