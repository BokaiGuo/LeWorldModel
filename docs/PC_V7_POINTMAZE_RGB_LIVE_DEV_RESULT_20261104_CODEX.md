# v7 PointMaze：单根真实 RGB 反馈与重放

按[开发协议](PC_V7_POINTMAZE_RGB_LIVE_DEV_PROTOCOL_20261104_CODEX.md)，同一 v7 核心在已暴露 PointMaze 根 0 完成三轮真实 RGB 重规划，每轮派发五步，固定终点剩余时域为 15／10／5 步。将暖身和本次实际派发动作送到 native 后端重放，四个边界的目标距离为 **1.745588 → 1.681042 → 1.118418 → 0.405924**。这是单根开发结果，不是独立成功率或规划优势。

| 决策 | 模型代理分数 | 弃权 | 实际 float32 完整链与清零链相同 | 预测 token |
|---:|---:|---|---|---:|
| 0 | 0.00850255 | 是 | 是 | 1920 |
| 1 | 0.00443927 | 否 | 否 | 1919 |
| 2 | 0.00062918 | 否 | 否 | 1920 |

首轮同一实际动作在不同批次的分数有差异；v7 最终同批比较并按相同动作规则弃权，不能要求最终分数与所有跨批历史分数完全一致。模型代理分数和物理目标距离是不同指标。

- [记录审计](../le-wm-main/results/pc_v7_pointmaze_rgb_live_dev_20261104_codex_audit.json)核验三轮因果 RGB／动作历史、carry、配对选择、精修起点、派发时钟及 token。
- [native 动作重放](../le-wm-main/results/pc_v7_pointmaze_rgb_live_dev_20261104_codex_physical.json)提供上表物理距离；没有将每个 native 状态与原 shadow 运行的物理状态逐项比较。
- [同 shadow 后端 RGB 重放](../le-wm-main/results/pc_v7_pointmaze_rgb_live_dev_20261104_codex_rgb_frame_replay.json)为 31/31 帧逐像素一致；[同 checkpoint 显式 packet 预测重算](../le-wm-main/results/pc_v7_pointmaze_rgb_live_dev_20261104_codex_prediction_replay.json)覆盖 305 批、3642 条轨迹，位置最大误差 `2.15e-6`、评分最大误差 `1.52e-6`。这不是独立模型或后端复现。
- [版本绑定检查](../le-wm-main/results/pc_v7_pointmaze_rgb_live_dev_20261104_codex_v5_version_regression.json)显示三轮最终 float32 完整链与该根先前 v5 一致，因此这是版本回归，不是新任务或独立泛化证据。

执行器控制台继承旧“两臂”文案，实际 manifest 只有 response 一臂。PointMaze 的 checkpoint 训练成员和 normalizer 拟合外来源仍未认证；本次没有进行同版本强基线比较或总算力认证，不构成 PC0、PC1、PC3、PC4 或 PC5 通过。
