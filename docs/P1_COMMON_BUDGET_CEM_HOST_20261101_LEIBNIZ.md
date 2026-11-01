# 已知根 CEM 共同预算 host

独占 host 复用已封存 variable CEM host 的真实 RGB 反馈、根与目标校验、
动作派发、物理日志、GPU 合作锁与 v2 CEM 日志锚定。节点使用新的共同
预算 cap 与 replay/selection 实际事件；节点账本在真实动作派发前结算。
目标是已见开发根，不作独立确认、等计算或 P1 admission 结论。

每节点前向 cap 195、replay/selection 各 193；实际预留依次为
195/65、194/97、193/193、193/193，第二个数字同时表示 replay
和 selection。编码与物理开销由既有 host 另行记录。账本仍为
`closed_unverified`；模型内部覆盖、FLOPs 和能耗未知。

本文件、测试、host 和共同预算实现加入新 prepare 的来源表。来源校验、
root 重绑定、身份不符、节点计费错误、失败及进程中断均不得视作成功；
保留唯一输出，不重写历史报告。仅 CPU fixture 检查不构成真实运行。
