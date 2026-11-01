# P3 六槽来源首尾绑定 producer

新增独立 `p3_six_bound_producer_20261030_v1.py`，不修改原封存、旧 lifecycle、终态或 strict consumer。它只提供注入式接线，不是模型加载器、GPU/物理 launcher、生产授权或科学门。

执行前要求全新独占 `BoundStore` 和全新六槽 native factory；只读重建 strict 来源合同，校验外部提供的来源合同摘要及本 producer 文件摘要，并将 factory 的两个 task、完整 root 数据及 source plan 与重建来源逐项精确对照。随后记 `six_strict_source_begin`（index 0）、调用已有 lifecycle 写固定六槽分母（index 1），store 在其后写入 producer 源码摘要、来源合同摘要及当前完整 store 限额（index 2）。六槽生命周期返回后重新重建并对照来源、factory 输入及 producer 摘要，才写 `six_strict_source_end`，然后使用已有独占 report/completion 发布。首尾事件内容遵循既有 strict consumer 原合同，producer/profile 绑定另由新消费入口核对。

槽内失败由原 lifecycle 按原样保留，六槽返回不代表六槽成功；即使来源首尾核对通过，已有 strict consumer 仍要求六槽都完整、清理成功、无 pending/orphan 等。来源漂移或其他执行异常时不写伪造的 end，尽力追加失败记录并发布部分终态后重新抛出原异常；journal 已损坏或终态写入失败时附加错误说明，但不保证故障后一定能写出记录或终态。

新 `consume` 先调用既有来源/上下文/策略数学保存核验，再独立检查 producer 摘要及 profile 事件。测试包括六槽 CPU 合成 backend、假环境/计量与合成帧，穿过真实 lifecycle、终态与 strict consumer 的完整代码路径；为控制测试成本，该合成链路使用缓冲写，不验证逐次 fsync 或真实来源。另有真实耐久 store 的失败、截断、来源漂移、错误 pin、末事件失败和绑定篡改测试。来源摘要和日志仍只是可复算绑定，不证明可信事前时间戳、硬件运行或真实模型权重。

原任务已有结果之后设计的方案不是事前注册；`full_saved_strict=false`、`physical_certified=false`、`scientific_pass=false`。真实权重/设备运行时身份、可信时间顺序、全作业资源和存储准入、完整 HOST driver 及独立审读仍未闭合，不得依据该接线启动生产或声称科学通过。

禁用 CUDA 可见设备后，联合本次及已有六槽、三策略相关 CPU 测试，67 项通过。真实来源合同及 producer 源码摘要已只读重建；该重建没有创建模型或环境。
