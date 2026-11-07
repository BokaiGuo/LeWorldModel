# PC1 固定执行环境、同步状态渲染分支

继承受控现场渲染v2，不修改先前失败记录。实际发现：
沙箱内 worker报告1个可见EGL设备，沙箱外4个（不代表CUDA物理卡数量）。
同一旧网格源码、同一状态和相机，沙箱内有约0.027/255初帧MAE，
沙箱外1884初帧MAE=0；后者frame5仍不相同。GPU vendor/renderer
进一步由worker记录，不能由可见设备数量直接推断硬件型号。

本次所有 prepare/predict/replay/analyze 必须在同一外部GPU环境执行；
使用独立的渲染sim对记录的qpos/qvel做forward，得到与该时刻状态同步的
图片，且不修改动力学sim的内部缓存。物理sim仍按原step规则推进。
记录物理sim已有site位置与当前qpos推导位置的差异，检验旧缓存滞后假设。
prepare与replay起始图仍要求逐像素一致；不放宽任何阈值。

官方archive协议依然单独保留；即使本版本某些帧匹配archive，也不能
抹去其余帧差异。当前版本不是官方等价认证或RGB-only结果。原32根、
校准/评价划分、动作、时域、成本与方法均保持。
