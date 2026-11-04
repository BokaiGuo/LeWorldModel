# V5 PointMaze RGB 设备兼容重试

实验内容沿用 [v1 协议](PC_V5_POINTMAZE_RGB_LIVE_DEV_PROTOCOL_20261104_CODEX.md)。v1 在首个环境 reset、任何规划和派发之前失败，保留其 manifest。将 worker 标准错误显式暴露的独立启动检查显示，旧 `mujoco_py` 把 `CUDA_VISIBLE_DEVICES` 解析为整数，UUID 字符串导致 `ValueError`；将同一物理 GPU 使用数字序号 `0` 后，单次 RGB reset 返回 `224×224×3`。这定位的是旧渲染依赖的设备标识兼容问题，不是 v5 规划器 bug。

v2 使用 `DISPLAY=:0`、`CUDA_VISIBLE_DEVICES=0`，并在运行前确认该设备对应原物理 GPU UUID；新输出前缀，不修改旧 manifest。方法、目标、输入、固定时域和预算均不改变。首次失败及 reset 检查不是反馈实验，不记入性能。
