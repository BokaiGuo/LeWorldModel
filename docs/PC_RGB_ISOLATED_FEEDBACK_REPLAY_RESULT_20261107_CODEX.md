# RGB 闭环规划进程隔离：开发重放结果

按[事前协议](PC_RGB_ISOLATED_FEEDBACK_REPLAY_PROTOCOL_20261107_CODEX.md)，对已曝光 TwoRoom episode 9000、9013、9022、9032 的 `effect` 和 `cem16` 两臂进行 GPU 闭环重放。环境进程设置物理初态并执行动作；独立规划进程通过消息接收 RGB、目标、时钟和旧尾，使用固定动作 normalizer 与冻结 RGB LeWM。

- 8 条轨迹共 **26 次规划、118 个环境步**，所选索引（effect 还包括来源）、完整计划、实际派发、模型查询和 packet 计数均通过历史记录对照；每一步物理距离、终止及截断标记亦在协议容差内一致。新输出记录每次计划与派发；距离和终止值是在运行时与历史 trace 核对，未重复写入新输出。
- 冻结模型指纹一致，公开 normalizer 与独立从历史数据计算的参数逐值一致；关闭规划进程时再次核验模型指纹。
- 四个既定敏感路径探针均不可读；规划进程未挂载完整结果目录和数据集。测试 **12 项通过**，包括非法物理标签、非法动作尾和时钟字段的拒绝。
- 第一次启动因第三方库在模块导入时向 stdout 写日志，导致协议 JSON 解析失败；增加专用消息前缀后重跑成功。失败没有覆盖历史数据。

限制：本次是既有开发轨迹的复现，不是独立确认，也不是新的方法—基线性能或总成本对比。文件挂载和固定路径探针不等于所有侧信道的安全认证；本机没有建立网络隔离，GPU、系统目录和实验源码仍可见。PC0–PC5 未据此改判。

[实现](../le-wm-main/experiments/pc_rgb_isolated_feedback_replay_20261107_codex.py) · [原始对照记录](../le-wm-main/results/pc_rgb_isolated_feedback_replay_20261107_codex.json) · [接口测试](../le-wm-main/tests/test_pc_rgb_isolated_feedback_replay_20261107_codex.py)
