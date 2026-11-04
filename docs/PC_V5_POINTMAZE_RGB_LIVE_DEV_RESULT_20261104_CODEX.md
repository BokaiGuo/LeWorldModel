# V5 PointMaze 真实 RGB 反馈开发结果

未修改的 v5 核心在已暴露 PointMaze 根 0 完成三轮真实 RGB 重规划，每轮实际执行五步，固定终点 15 步。原生后端重放得到的目标距离依次为 **1.745588 → 1.681042 → 1.118418 → 0.405924**；这是单根开发结果，不是成功率或 PC3 优势。

| 决策 | 剩余时域 | 模型代理分数 | 弃权 | 预测 token |
|---:|---:|---:|---|---:|
| 0 | 15 | 0.00850139 | 是，选择零前缀 | 1920 |
| 1 | 10 | 0.00443917 | 否 | 1919 |
| 2 | 5 | 0.000628869 | 否 | 1920 |

模型代理分数与原生物理距离不是同一指标。旧 v3 同根 response 和 smooth CEM 的最终距离分别为 0.31790 和 0.09485；这是事后开发背景，不构成独立配对确认，也不能把不同版本结果拼成同一方法证据。

## 审计与限制

- [记录审计](../le-wm-main/results/pc_v5_pointmaze_rgb_live_v2_dev_20261104_codex_audit.json)检查三轮图像历史、carry 重评分、动作域、派发时钟、token 及所评分完整链最小值；[native 动作重放](../le-wm-main/results/pc_v5_pointmaze_rgb_live_v2_dev_20261104_codex_physical.json)给出上表物理距离，但未与原运行每个物理状态逐项配对。
- [参考预测重算](../le-wm-main/results/pc_v5_pointmaze_rgb_live_v2_dev_20261104_codex_prediction_replay.json)独立实现 packet 展开，覆盖 302 批、3642 条轨迹；最大位置误差 `2.15e-6`、评分误差 `1.52e-6`，在预定容差内。其依然使用相同 checkpoint、编码器、输入和成本函数。
- [逐帧 RGB 重放](../le-wm-main/results/pc_v5_pointmaze_rgb_live_v2_dev_20261104_codex_rgb_frame_replay.json)用同一 shadow 后端重放 30 条动作，31/31 帧逐像素一致。这不是不同后端或原始物理状态的独立验证。
- 首轮启动时，旧 `mujoco_py` 把 GPU UUID 误按整数解析，环境 reset 前退出；独立检查后以同一物理 GPU 的数字序号运行 v2，首轮 manifest 保留。[兼容性记录](PC_V5_POINTMAZE_RGB_LIVE_V2_DEV_PROTOCOL_20261104_CODEX.md)；未发现 v5 规划器因此存在 bug。

实际 runner 控制台沿用“两臂”文案，manifest 与产物只有 v5 response 一臂。该根已暴露，归一化拟合成员和 checkpoint 训练成员不能视为独立；完整资源成本及任务相关物理响应仍未认证。以上不构成 PC0、PC1、PC3 或 PC4 通过。
