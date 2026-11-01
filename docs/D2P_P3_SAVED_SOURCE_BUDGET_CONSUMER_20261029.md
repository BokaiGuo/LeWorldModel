# P3 K17 保存日志的独立来源与预算 consumer

## 实现

新增 `le-wm-main/experiments/spectral_reachability/p3_k17_saved_source_budget_consumer_20261029.py`，独立从保存 journal 重放，不调用实时接缝的计数或状态机。`consume(directory, journal_sha256, source_files_sha256, output=None)` 要求调用方预先从独立可信渠道持有完整 journal SHA 和源码摘要字典；验证规范 JSON、序号及链、七个直接源码文件摘要和运行时磁盘源码，再通过现有 `outer.material()` 重建 pinned 来源、六槽和 paid replay 动作，检查 binding 与动作一致。调用方将摘要从同一个不可信目录现场计算，不提供独立真实性。

实时接缝的首条 `source_budget_binding` 新增七个直接源码文件 SHA；新增字段之前的旧日志不满足本 consumer 的新契约，不能静默升级。七文件不是依赖闭包，也不证明加载的 Python 字节码或实际进程使用了这些文件；源码内容与数据来源快照绑定仍依赖接缝诚实产生日志。`from_saved` 继承上游 pinned receipt 的信任边界，不是 full strict 重算。

consumer 从 intent、returned、failed 独立重算各轴 attempted/completed/failed，核对资源上限、序号、槽身份、guard、实际返回动作、事件顺序、各槽关闭与故障。guard 前要求十个 warmup intent/return/transition 和 36 条 PNG 事件，完成分支要求五个计划动作及 51 条 PNG 事件；已登记 PNG 文件检查路径、唯一性及字节 SHA。重链后的语义伪造、缺事件和伪造 completed branch 会拒绝；未完成 journal 的 pending 和活跃分支被报告为未完成，而非填零。报告可选独占写出并 fsync，写出失败不得视作持久凭据。

`six_slot_event_coverage` 只表示本 consumer 所检查的六槽状态、动作和事件数量满足条件；PNG 不解码，36/51 只核对数量及字节 hash，未核验 phase/step/overlay 全部顺序、画面或 guard 的真实性。现有 producer 没有完整 final completion/manifest，故 `complete_receipt_verified` 恒为 false；`native_provenance_verified`、`production_authorized`、`scientific_pass` 也恒为 false。失败或部分记录只能描述可见日志，不能排除未记录的执行或 I/O 错误。墙钟、真实算力、guard 内部重算、完整来源闭包、实时环境身份、外部锚生成与保管、保存消费者的全物理结果审计都不在该 consumer 的认证范围内。

## 测试与真实数据边界

在 `le-wm-main` 执行 `.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p3_k17_saved_source_budget_consumer_20261029.py tests/test_p3_k17_live_source_budget_seam_20261029.py tests/test_agent_p3_c3_k17_native_meter_20261029_v1.py tests/test_agent_p3_c3_k17_candidate_join_20261029_v1.py`：33 passed。测试覆盖真实 store/meter/outer 接口上的 fake 环境和 fake 原始帧文件、六槽计数、持久报告、日志与源码锚不匹配、重链伪造来源/guard/动作/返回、缺 intent/帧、虚假完成、未知事件、故障和 pending、丢失 PNG 与来源变化。假帧文件不是真实 PNG 或原生采集。

本次没有可供消费的外部锚定真实 P3 原生日志，未运行真实环境、物理、GPU 或训练；未声称生产就绪或科学通过。未来实际执行仍需预授权、原生不可绕过的事件采集、完整可持久 completion/manifest 和可信外部锚、完整源码依赖闭包及独立环境/guard/帧/物理结果验证、故障与存储极限审计。
