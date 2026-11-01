# P3 K17 原生来源绑定与计数接缝

## 结果和范围

新增 `le-wm-main/experiments/spectral_reachability/p3_k17_live_source_budget_seam_20261029.py`，不改现有 adapter、其他子智能体文件或已有门槛。它包装现有 `NativeStore`，使用现有 `outer.material()` 的 pinned handoff、实际保存 replay 和来源材料，接收现有 `outer.execute_material()` / `Meter.call()` 发出的实时 `emit`，不是新增纯 fixture transcript 检查器。没有 CLI、自动派发、GPU、训练或物理执行。本接缝不授权执行、不验证原生环境身份、不声称可生产或科学通过。

真实保存来源的 CPU 只读预检：使用 `CUDA_VISIBLE_DEVICES=`、临时 `NativeStore` 和 `SourceBoundStore.from_saved()` 成功读取 6 槽及每槽 5 个原始动作；规范化来源快照 SHA256 为 `ee75bb9cd863e700bf0e3d08587e353ea7c21600a0dbd9f11d38c0c4fe86f03a`。该摘要反映本次本地来源快照，不是外部可信签名或真实执行凭据；来源闭包继承上游验证，未重新执行模型或 full strict。临时目录已释放。

## 真实接口契约

未来经独立批准和审查后，调用方可以先创建现有有界 `NativeStore`，调用 `bound = SourceBoundStore.from_saved(store)`，再将 `bound.data, bound.rows, bound` 作为现有 `outer.execute_material` 的 data、source_rows、store 参数。`factory`、环境、审批和执行不由本模块提供。不要将此示例视为当前执行许可。直接构造 `SourceBoundStore(store, data, rows)` 可用于测试，但其来源是调用者自称，不等同 `from_saved` 的 pinned 来源加载。

包装器先向原有持久 journal 写入绑定事件，包括完整规范化来源摘要、六个行摘要、固定顺序、原 meter 资源上限和墙钟，所有授权标志保持 false。run start 和各槽 constructor 前重算完整来源；guard 和计划动作前重算当前槽来源。实际传入的计划动作 raw f32 字节逐步与已加载 paid replay first5 比较；仅在已有 `k17_guard` 接受且 proof 字段符合冻结 gate 契约后接受计划 intent。与 `returned_transition` 的请求及 actual 字节、dtype、终止状态核对；终止后拒绝下一动作。该验证依赖现有 episode 真正调用 `capture_check`，不会独立认证 proof 或环境。

通过 `operation_dispatch_intent`、`operation_returned`、`operation_failed` 对现有物理资源轴独立统计 attempted/completed/failed，支持 public step 内嵌 physics step；逐轴沿用现有 `CAPS`，不退款。intent 验证和持久写入在原 meter 调用 callback 之前；接缝的额外墙钟从其创建时起计，校验和 journal 写入前后检查，较原 meter 更保守。失败时 fail-closed；已有 meter 的 close cleanup 仍可尝试。若写入失败，接缝保守保留可能未持久的 attempted/pending；若 callback 已返回但返回事件写入失败，内存 completed 不能当持久完成证据。底层异常、断电或磁盘故障可能无法保存失败信息。

## 测试与边界

`.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p3_k17_live_source_budget_seam_20261029.py tests/test_agent_p3_c3_k17_native_meter_20261029_v1.py tests/test_agent_p3_c3_k17_candidate_join_20261029_v1.py`：20 passed。测试使用真实 `NativeStore`、真实 `Meter`、真实 `outer.execute_material` 接口，环境和 episode 均为 fake；覆盖六槽接口流、嵌套计数、来源变动、guard 缺失、动作不匹配、资源上限、墙钟、故障不退款、journal 写失败、cleanup 和终止停止。没有真实环境采集或物理执行，测试结果不能视作科学证据。

剩余障碍：正式六槽范围、预算与停止规则须经授权；真实环境构造/reset/goal/warm10/PNG 和 guard 调用方须独立鉴别；调用方可绕过包装器或伪造 `emit`，因此需要不可替换的原生执行边界；这些计数只覆盖接入 meter 的操作，不证明完整物理/计算资源、等实际计算量或全部墙钟；还缺外部签名或锚定的完整 receipt、崩溃恢复和完整 saved consumer、来源与事件的独立回放核验、容量与失败持久化的端到端证明、并发/重入约束和真实环境对抗测试。当前没有自动完整发布与科学准入。
