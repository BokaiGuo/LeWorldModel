# SRA checkpoint / 内存模型生产前检查

独立入口 `python -m experiments.spectral_reachability.codex_sra_checkpoint_identity_20261030 --policy <checkpoint目录> --output <新proof路径> --verification-output <新复核路径>` 仅在 CPU 构造冻结 eval 参考模型，不做推理。输出文件独占创建并 fsync。默认固定已有 PushT checkpoint 的 `weights.pt` 和 `config.json` 摘要及当前兼容加载器源码摘要。加载器通过 `weights_only=True` 读取 checkpoint，重映射旧 ViT 键，拒绝缺失/多余键；此检查再次对重映射后的 checkpoint 与参考模型持久 state 的**名称、dtype、shape、逐字节摘要**逐项比较。proof 同时记录全量 named parameters、persistent 与 nonpersistent buffers、module 类结构以及 `interpolate_pos_encoding`。

`verify_proof` 以外部提供的 proof 文件摘要为锚，重新核查权重、配置、加载器、检查器和 journal `model_sha256`，安全读取 CPU checkpoint、重新映射并对实际内存对象的持久状态逐项比对；还与 CPU 参考的参数、全部 buffers、模块结构和标志比较，要求所有模块 eval、参数冻结，以及 `replay.model is model`。`guarded_online_select` 可选地把该检查接到现有 online journal：方法 manifest 必须包含 proof、weights、config、loader 和本检查文件的路径及摘要，encoder 与 planner 前后均检查；回调内发生模型漂移时拒绝选择并保留既有 journal 的失败状态。旧入口仍可绕过此 guard，因此它**不是强制生产门禁**；调用方必须实际使用 guarded 入口。

CPU02 结果使用本机真实 checkpoint，构造一个参考和一个另行加载的 CPU 模型：303 个持久 state 项相符，其中 192 个键被重映射；参考模型有 297 个参数项和 6 个 buffer 项。两次加载源于同一 checkpoint、同一加载器及环境，第二次只验证重复实例一致，不是独立训练来源、独立科学确认或 GPU 运行验证。CPU01 是早期格式的中间产物，不包含最终检查器源码摘要，不应交给最终 guard；CPU02 proof 和 recheck 以各自文件摘要为准。

边界：调用方提供的 encoder/planner 可以在内部使用其他对象，本 guard 无法证明回调闭包或原生代码真实执行；也不能防止检查间并发修改、动态 monkeypatch、未登记依赖或模型 Python 非 tensor 状态变化，须独占持有模型并另行绑定完整运行时。nonpersistent buffer 只与同一加载器生成的参考比较，不是来自 checkpoint 的直接证据。配置和权重哈希也不证明训练数据、独立根、目标/normalizer 的来源或科学授权。每次检查会重新读取 checkpoint 并复制/哈希模型张量；若未来对象在 GPU，这些检查本身可能触发设备到 CPU 传输，其时间、I/O 和计算未纳入现有 SRA 预算账本。不得把检查通过描述为等预算、完整 D2P 或物理结果。
