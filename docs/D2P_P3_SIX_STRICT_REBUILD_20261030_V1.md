# P3 六槽来源与上下文数学保存核验

本次新增独立 `p3_six_strict_rebuild_20261030_v1.py` 和 CPU 合成记录测试；没有修改已有合同、已有结果或科学门，没有启动模型、GPU 或物理环境。现有六槽 publisher 和 native saved consumer 的非认证边界不变。

入口先调用已有六槽终态和原生记录消费者，拒绝缺槽、失败、未完成 journal、截断、orphan、未完成 callback 或 store 异常。然后只读重建已有封存来源、已知两根、任务、模型预期哈希、精度合同及相关源码闭包，加上本 verifier 的源码哈希；要求调用者提供该重建合同摘要，且 journal 必须以对应的 `six_strict_source_begin` 开始、在所有六槽返回后以同内容 `six_strict_source_end` 结束。未来 producer 尚未发出这些记录，现有不含记录的产物必须拒绝，不能事后补造并声称预注册。记录与摘要是来源绑定，不是不可伪造的时序证明或硬件证明。

每个完整槽核对真实记录的 warm11 state/snapshot/pixel 与封存来源、root guard，以及 packet return、编码帧和重建上下文的先后关系。消费保存的 latent 数组，从 warm10 和 action normalizer 重算四次完整上下文，并以实际阶段上下文逐项重算两条 signal 的谱、原决策、reverse 首块映射、endpoint bank、四候选 Q 与 winner；对照 native 调用请求/返回，并复用旧无策略假设的 native IO/resource census。原结果需要实际支付的所有动作和调用，不能仅按 receipt 自报。

返回状态只表示**来源绑定的保存数学重建**，`full_saved_strict=false`、`physical_certified=false`、`scientific_pass=false`。模型/设备真实运行、权重运行时身份及完整六槽整体资源/存储认证尚未闭合；事件内容可伪造，不能由哈希本身推断可信时间戳、事前注册、fresh replication、等 FLOPs 或科学结论。还需未来独立实现并审读来源绑定事件的真实 producer 和完整 HOST driver；不能将测试 fixture 升级为物理记录。

测试使用 RAM 的合成 backend，覆盖三策略重建、warm/action/latent、signal、bank 和调用缺失篡改、来源绑定缺失/变化及部分终态拒绝。该测试不构成完整六槽真实保存文件端到端认证；只读真实源码合同重建也不构成真实执行。

禁用 CUDA 可见设备，联合既有六槽相关测试共 60 项通过；另新增的反转 horizon 篡改测试单独通过。真实源码/已知根合同只读重建成功，两个根、`scientific_pass=false`。
