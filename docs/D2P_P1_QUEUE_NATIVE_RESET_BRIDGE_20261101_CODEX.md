# 开发队列与原生 reset 接线

此接口只允许队列中 `development` 槽，直接创建原生 PushT，**不授权
confirmation 角色**。队列必须以文件 SHA 输入，且其 producer、环境和本文
protocol 字节摘要必须与当前入口匹配；既有队列 schema 的 model SHA 只是
声明，本接口不加载模型或认证其来源。真实 reset 前先独占写 reservation，
再将对应 slot、attempt、seed 的 `reset_attempt` 同步写入原队列格式账本；
之后创建环境并派发原生 reset；若环境构造失败，attempt 是预派发记录而
不是声称 reset 已实际调用，失败回执的 phase 保留这个区别。返回 state、goal 和刚体快照的独立回执绑定 reservation、
队列、attempt 事件及环境源码。调用没有 RGB/history，成功后账本故意维持
pending：不伪造 `history_sha256` 或提前写 `reset_return`。

入口持项目物理协作锁；其同一规范队列文件路径下的 bridge 专属目录和
advisory lock 串行，不能重复使用已经尝试的 slot/attempt，已创建但中断的
reservation 也不可自动重试。reset 或回执构造错误保存失败回执并尝试写入
`reset_failed`；任一写入或进程崩溃仍可能遗留 pending 或 reservation，
需要人工审查，不得删除/改名后当作新尝试。此保证只覆盖遵循本桥并共享同一
规范队列路径及账本的调用者；复制队列到不同路径、绕过桥直接 reset、
篡改工作区文件或不遵循协作锁均不能由此代码阻止。

冻结队列不是外部可信时间戳或访问控制；成功 attempt 不是有因果 RGB、
action history 的任务。历史穷尽、训练成员/训练外、freshness 均 unknown，
独立确认和 P1 admission 恒 false。source 签名为有限 Python 文件与已知原生
环境源码，不认证完整 native 依赖、内存代码或物理求解器隐藏状态。
正式确认还需要独立可信保管、完整 placement/warmup 生产器及其所有 reset
接口和失败的强制记录、历史排除与 checkpoint 训练来源的独立审验。
