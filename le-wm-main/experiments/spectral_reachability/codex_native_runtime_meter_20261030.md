# Native runtime 接线边界

新增 `codex_native_runtime_meter_20261030.py`，不加载 checkpoint、不创建环境、不提供生产入口。实际调用时：

- `native_rgb_encoder` 将明确形状的三帧原始 uint8 RGB 字节和两包 raw 5×2 历史交给现有 `p4_feedback_matched_20261027_v2.encode_cuda`，其实现负责视觉编码与动作归一化；编码应由已有 online interface 的 encoder 回调调用，该接口预留一次 visual_encoder。
- `sra_online_bank` 将当前在线 context 传入现有 `p2_no_composition_sra_20261027_v2.compile_cuda`，得到 joint-H2 bank。它没有调用 `run_bank`，没有给出选择动作；保存旧 winner 不能替代此缺口。
- `full_packet_online_compile` 从当前在线 context 重算 root nominal 和 Jacobian，再调用 `pusht_full_packet_compiler_20261027.compile_arm(arm="full")`。这是组合式 full-coordinate packet 编译组件，不是已定义或已验证的完整 Full D2P。结果尚未与目标绑定、在线候选选择和独立保存消费完成端到端闭合。

实际 model 的 `predict` 和 `action_encoder.forward` 入口包装在 journal meter 中：每次进入前按实际 batch 预留 calls/samples 或 encoder 次数；超过 cap 不进入原函数，异常留下 pending，最后恢复原方法。不能安全并发或重入地共享同一 model 实例。高层 Jacobian、VJP 方向、SVD 和 solve 是根据当前静态算法预先整阶段预留；阶段成功返回才记录 return，失败保留 pending。它们是**计划工作量的保守预留，不是逐次测量或硬件执行证明**；journal 的对应 completed 数只能解释为该预留阶段已返回，不得解释为独立测得的实际算子数。异常阶段不能退款。视觉编码、物理、初始化、I/O、wall、backward 内部操作和 FLOPs 的完整开销仍未闭合；未列值不等于零。

入口文件固定摘要检查与调用者文件清单的摘要检查不是完整传递依赖闭包，也不证明 checkpoint、内存模型、目标、normalizer 或当前方法参数的真实来源一致。调用者仍须在进入生产前将真实目标/投影、模型权重、normalizer 和全部依赖绑定到同一协议身份，完成 SRA 的在线 `run_bank`/重放/选择并接入有持久消费校验的 receipt，验证两臂相同任务及各自历史，并独立审查完整资源与物理 guard。不得把本接口、源哈希或测试通过当作这些门槛完成。

测试只使用替身 model、encoder 和 native 编译函数；没有真实 CUDA 执行、模型推理或物理执行。没有任何等总预算、公平性、方法效果或 Full D2P 物理结果声明。
