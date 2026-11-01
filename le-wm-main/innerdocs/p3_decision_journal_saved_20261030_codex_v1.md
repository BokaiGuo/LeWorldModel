# P3 四决策六槽：durable callback journal 与 saved 消费

仅为 `p3_decision_equal_work_control_20261030_codex_v1` 新合同实现；旧两块 six-slot consumer 不可认证此 schema。准备合同 SHA 为 `df480bc6d9c55daefbeac9bdb79373bf7d3eba735148cc42f0f850d05d16b860`，原 callback 源 SHA 为 `3629fba1abd9bbbee2dc6ed273f65290d17f88588e30ed863646d3cbb4f83500`。

`p3_decision_journal_20261030_codex_v1.Store` 提供独占目录、追加哈希链、每事件 fsync、目录 fsync、每记录和全日志字节上限、终态预留；写入或上限失败后毒化，禁止继续。调用现有 `execute_roster(..., store)` 后明确调用 `finish(roster)`，末事件保存固定六槽终态及来源 pin。无成功末事件的中断日志保留原字节，但不能由本消费者作为已结算日志消费。上限由调用方制定，不是可用磁盘、显存或 wall 的完整准入；终态过大、磁盘故障或不可中断进程均可能没有末事件，fsync 的硬件持久性不能由 journal 自证。

`p3_decision_saved_consumer_20261030_codex_v1` 要求外部 SHA、正常换行、逐行 canonical JSON、严格索引和哈希链、来源 pin、独占完整六槽顺序和 terminal。它核各槽每决策资源事件的实际尝试、返回、失败与嵌套，重建并对照 ledger、统一 caps、策略决策、winner 前五步、packet 和编码阶段顺序，记录回调数及真实停止；完整、失败、提前停止、setup/cleanup 状态不互相升级，缺槽/未决事件或篡改 fail closed。受限失败槽保留已付账和 callback 计数；callback 已返回但后续验证失败时，返回不表示 ACK 有效。资源辅助 IO 事件只核允许种类与决策阶段，不验证其数组、内容、模型计算或 native census。

该消费者复用原 `core.audit_ledger`、合同与策略，并不是独立实现所有科学数学的 verifier；它没有重建 signal、bank、winner Q 或 RGB/latent 数值，没有采集真实物理计数、模型权重/精度/GPU 证据，不能认证未记录调用或事件真实性。外部哈希和 fsync 调用不是可信时间戳或设备证明；本地报告中的 `source_authenticated`、`physical_certified`、`full_saved_strict` 和 `scientific_pass` 均为 false。相同已测轴不证明等 FLOPs、wall、总计算或物理。

真实主机运行之前，仍须将该 store 安全接至真实 CUDA backend、原生仿真和每步观察的生命周期，绑定和记录 checkpoint、设备、动作、RGB、latent、环境根/goal 与资源；开发独立数值/物理 saved consumer、总资源和存储准入、外部硬超时和 failure cleanup，并经独立审查、事前可信时间戳及主线程授权。原模型 callback source 和已封存 PREP01 不修改。这里不提供 GPU 启动命令，不训练、不运行 GPU/物理，不放宽 P1。

CPU-only 测试用真实文件 fsync 与假 backend；除了可结算完整/失败/早停混合六槽，还测试字节篡改、链重算后删除资源或槽、动作替换、未知记录、错误 SHA、截断、末事件缺失和独占/上限失效。不把 fixture 结果当作生产或科学证据。
