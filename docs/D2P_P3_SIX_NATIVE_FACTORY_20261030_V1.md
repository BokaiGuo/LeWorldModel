# P3 六槽原生 factory 接线：仅准备

本次新增 `p3_six_native_factory_20261030_v1.py`，不修改旧四槽或六槽合同。它可由既有六槽注入式生命周期调用，复用原生 meter、reset wrapper、warm/root/goal setup 和 packet/encode session；本身不是 launcher，构造 factory 不加载模型或创建环境，调用 factory 会按注入的原生 factory 创建环境。六槽固定顺序、单次尝试、每次最多一个活动环境；setup、wrapper、session、ready 事件失败时，若已取得环境句柄则尝试关闭；关闭失败或构造阶段无句柄的异常阻止后续槽。setup 抛错前未返回句柄的内部资源无法由本 factory 恢复：尤其旧 `native_meter.native_factory` 内部构造成功后若记账返回失败，句柄可能不可见，必须在未来原生 runner/构造实现中处理，不能当作已解决。

新 capture 限制每槽 32 帧、全局 192 帧，按原始 capture 接口保存 PNG 和记录；逐次写入继续由注入 store 执行其配额检查。预检只检查两个 task/root 哈希一致、固定存储 ABI、初始帧数，以及 PNG 分类能否容纳 192 × 256 KiB 的上界；**并非六槽总存储、剩余磁盘、失败终态预留或 wall/显存的完整准入**。现场 root/warm/goal 一致性仍依赖复用的 native setup，输入哈希一致本身不认证来源。`source_contract()` 重建已有六槽源码封存，再附加本文件哈希；它不是外部可信时间戳或注册。

严格 saved consumer 和 journal 缺口保持不变：尚无六槽专用 durable store、部分 journal 恢复、独立完整原生资源/动作/帧/encoder/模型核查或实际 HOST 全作业 driver。factory 的来源、store 标签、计数及 fixture 结果都不授权或证明原生运行。现有生产路径仍未就绪，科学门不变；已知根不是 fresh replication，不声称事前注册、科学通过、等算力或泛化。

测试只注入 CPU 假 store、meter、环境、session 和 callback，不运行 GPU、模型或物理。覆盖六槽顺序、跨槽生命周期、预检拒绝、重试拒绝、帧限额、setup/包装/ready 失败清理及清理失败后锁死；这不验证真实环境或 durable journal。

禁用 CUDA 可见设备，联合执行本次 factory、既有六槽准备和三策略 callback 测试：34 项通过。只读重建并验证 factory 扩展源码合同：共 25 个源码、全局帧上限 192，`host_runner_ready=false`。
