# MAIN 审读：P2 已知根工程 CUDA 运行

审读对象为 `agent_p2_runtime_guard_20261028_v1.py`，源码 SHA256 `616ec94445681ede449e525a45016b4e7d849ec488181674f570e25f5ca721d3`。MAIN 完整阅读 source/tests/innerdoc，另复核新增独占 metadata writer 与 CLI 回归，不依赖旧前缀 writer。运行前需使用最终源码的 MAIN 测试及 saved verify 返回。

本次仅执行固定四个已知根、8 臂、480 候选的工程验证，不授权独立 P2 确认或物理派发。原 native prepared canonical SHA256 `1d79f5b17002334434454dfb5a94e6d918de1d603afdf84918e5ded6eb980d1b`；wrapper prepare SHA256 `02269ae5d08d1f645e9bcf2473a7c7cbb2a16f3125c3fad62c23859380cc6c48`。不得替换根、重试、借用跨臂预算或修改原数学内核。

真实原生 `compile_cuda/suffix_cuda/run_arm` 与 `guarded_entries` 接入持久化 intent/entry boundary/return。boundary 是 host 调用前边界，不证明 GPU kernel 已启动；observed returned 为下界，失败或 pending 不填完成量。每根最后必须 CUDA synchronize。数组、candidate、日志、首次异常与未执行项全部保留。每根 journal cap 16 MiB、boundary wall 900 秒；它不是全输出硬 cap 或抢占式 watchdog。

模型只从已有本地 checkpoint 加载，prepare 拒绝 encoder.pretrained=true；HOST CUDA、eval、requires_grad=false，参数及 buffer 前后哈希不变。不存在 CPU 神经降级分支。NumPy SVD/solve 为原规划器数值计算，不冒充 GPU 调用或完整 FLOPs。模块调用／数学 VJP 方向不等于 backward engine profiler。

尚不满足的科学前置保持原样：独立 P1、freshness、P2 组合增量、原生物理成功和全资源等预算比较。因此该运行即使工程全部通过，也不证明 C2/C3 或整套方案成立。P2 结果没有新 CEM8 cohort 的可比性，不能混合分母。

该文件冻结作为实际 execute-known 的审读引用，不在运行后追加结果；实际结果在其他新文档记录。
