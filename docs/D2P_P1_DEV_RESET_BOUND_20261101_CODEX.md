# 开发样例：原生随机 reset 与源码绑定

这是工程开发样例，不是前瞻确认队列、训练外证明或 P1 admission。两个固定
seed 在测试中已经执行过；DEV01 是相同 seed 的再次开发调用，并非未打开根。

prepare 固定了本 producer、队列历史检查器、协作锁、PushT 原生环境、
Gym core 和相关 Python 环境源码的当前 SHA，以及有界历史清单与条件
quarantine；只有原生 PushT 环境源码另行对照了固定已知 SHA。来源清单是
有限 Python 字节绑定，不是 native 动态库、所有间接依赖、运行时内存或机器
全历史的穷尽证明。运行前后重新核实 prepare；执行持项目协作锁。

调用直接创建 `stable_worldmodel.envs.pusht.env.PushT`，不是原有
`gym.make` 包装链；两次 `reset(seed=...)` 前分别 fsync 事件，返回后记录
原生 state、goal 与刚体位置/角度/速度及其摘要。异常记录失败并停止，
未执行槽保留 `not_started`。这条独占调用路径没有调用模型、public step、
显式 placement reset、RGB-history/warmup 或队列的正式确认角色；reset
内部仍执行环境原生的状态设置、物理和目标渲染。不要将记录的快照误认为
完整 solver 状态。history 标记为未收集，不生成虚假 history hash。

本入口和旧队列工具尚未在同一受监管 producer 中统一：它只使用旧工具的
有界历史 seed 检查，不会把真实调用追加到旧队列的正式 ledger。若进程崩溃，
最后一个已 fsync attempt 可能没有 return；如果输出设备本身失败，无法
保证失败回执可写。也无法阻止本 producer 之外的未记录 reset。未来正式接线
仍须提供 placement/warmup 全流程、完整中断恢复、可信外部保管与 access
control、未知历史和训练来源证据，保持 gate 不变。

固定 prepare SHA
`f10c46e0d3805d907fb6902124600c565a4be657b81d4480f52f75fda74305e1`；
DEV01 report SHA
`e28f3530a3d5c79ac62de0166acf1582623cbad5a7ecd5ee9e5c5fc34cd0694b`。
独立保存核验检查 prepare 源码、计划、日志链、两次调用与 state/goal/snapshot
摘要；它不是另一次物理重放，也不能证明环境内未监控的执行或日志之外无调用。
