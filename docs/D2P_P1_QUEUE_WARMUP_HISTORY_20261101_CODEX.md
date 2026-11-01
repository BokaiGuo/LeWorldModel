# 开发队列：原生 reset 与真实 warmup/history

仅 development，confirmation 不可执行；这不是预注册确认或 P1 admission。
同一原生 PushT 实例先真实随机 reset，再按固定 raw zero 动作执行最多十步；
实际 step 0、5、10 捕获 RGB，记录所有实际返回动作和状态。reset、每步
dispatch/return、帧与最终 source receipt 均单独落盘，不从旧 snapshot
恢复或把预测帧、空动作拼成完整 history。原生成功、截断、异常均停止并
保留实际步数与未完成队列；不补帧、不选替代根。

队列须冻结当前 producer、环境与本文哈希。先为 slot/attempt 独占创建目录，
同步写 reservation 和队列 `reset_attempt`，再执行；不复用该 slot/attempt。
完整 history 持久化后，队列 `reset_return` 的 state/goal/snapshot 哈希绑定
实际 warmup 末端，history 哈希绑定原始 reset 回执、三帧文件/像素哈希、
十次实际动作和实际 source 状态。任一中断可留 pending，绝不自动重试。
文件锁、防重放仅覆盖遵循本入口并使用同一规范队列路径的参与者，复制
队列或绕过入口不能被阻止。无可信外部时间戳和访问控制。

这只演示随机 reset + zero warmup，不含旧 producer 的显式 state placement、
目标重新绑定或通用 policy history，不能冒充旧任务数据。原生 reset 自身
可能含内部状态设置/渲染/物理步骤；本入口计数公开 reset/step/render，
不声称内部 kernel 或完整求解器计数。source 和快照仅是有限范围的实际回执。
训练成员、历史覆盖、freshness 均 unknown，独立确认与 P1 admission 为 false。
