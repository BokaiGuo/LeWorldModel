# P3 四决策：native CUDA 与反馈接口

## 资产和接口

旧四槽 factory 固定 fixed/signal roster；旧六槽 factory 固定两块策略和不同 policy 标签。新适配器不修改它们：按新合同固定的六槽顺序，将 `fixed_h1`、`signal`、`reverse_signal` 分别显式映射到旧 factory 的 `fixed_order`、`signal_order`、`reverse_signal_order` 作为 session 标签。映射只用于构造相同根的 native 环境；四决策策略仍由新合同执行，不能继承旧六槽 receipt。

`p3_decision_native_cuda_adapter_20261030_codex_v1.Adapter` 接收**调用方已加载**的冻结 CUDA 模型、torch、显式 CUDA device、物理 UUID、两个不同 store、峰值限额和源码 pin；它不加载 checkpoint、不创建服务、不分配 GPU，也没有 CLI 或生产授权。对象初始化校验新 PREP01 与旧源码闭包、两已知根、模型权重与源码、eval/frozen、所有参数/缓冲设备、精度和实际 runtime UUID。源只读重建得到 native source contract 摘要 `60f3b0c05f89f1ba51df79529b99e3fb9ce50cf6b971600be8b5b6ee7bbac537`；旧 prepared 的 policy 是 `/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm`，预期 `original_state_dict` 哈希 `4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546`。此为合同中的预期模型哈希及来源重建，**本次未加载或计算实际模型状态哈希**。

接口为 `adapter.session_factory(slot, root, policy)` 和 `adapter.backend_factory(task, decision_store)`，供现有 `control.execute_roster` 调用；最后先调用 `adapter.finish()`、再按各自协议结束两个 store。session 利用已存在的 guarded native factory 做 reset、warm10、root/goal 检查、观察 capture、packet ACK 和逐点 RGB encoder；数值接口实际创建 `NativeTree` 和 `RecordingBackend`，绑定同一 model/device、当前槽 root 派生 task、持久数值 sink 和 guard。调用前强制 `device.type == "cuda"`，不提供 CPU 回退。

每完整槽四个决策各实际调用两次 native signal、18 次 native replay，共 8 signal、72 replay；四次实际反馈 packet 及 encoder 边界。四决策的主账在新 decision journal；native 调用请求/返回/失败和 tensor hook IO 使用旧 numeric bridge 分块写入独立 native store，单槽序号跨多个 decision 持续递增。native store 同时保存原有 physical meter、帧和 encoding 记录；适配器在槽边界将决策 journal 的当前索引/链尾及新旧 policy 映射写到 native store。该单向局部锚点尚没有独立跨日志 saved consumer，不能将两条日志凭这些记录宣称已认证闭合。

## 守卫和计量范围

沿用 `NativeTree` 的 signal/replay：H1/H3 状态预测、nominal 与 AD、每个信号 192 个 VJP 方向；观测 model 的 action_encoder、predictor、pred_proj，缓存信号路径禁止视觉 encoder/projector，实际 RGB 编码走原 encoder v2 和自己的记录。每决策的数值账及 caps 仍由新合同的 ledger 收取；物理 step/render、图像和观测编码由旧 native meter/store 分别记录，不把不同轴加总。

数值调用前后、session packet/encode 前后及原 meter 非 close 调用边界检查精度、设备 UUID、model eval/frozen 与参数/缓冲设备、CUDA allocator 峰值、native store 状态/时间。出错锁死适配器和 factory，禁止后续派发；清理不受 guard 阻挡，旧 session/backend 在槽关闭后不能复用。入口/结束会比较模型状态及源码、精度快照和 hook 恢复；这些是调用方上下文中的自报告检查，不证明运行时每个瞬间的状态或硬件真实性。`finish` 是身份检查，不代表六槽科学成功。

护栏不覆盖所有 GPU kernel、全部 GPU 显存、进程 RSS、模型加载、checkpoint IO、存储写入之外的文件、FLOPs 或全部物理动作的独立真实计量。`max_memory_*` 是进程 allocator 采样而不是显存预留；外部硬超时、任务间资源竞争、环境构造在取得句柄前的资源泄漏仍不能由本接口解决。store 的类型和 token 不构成可信授权或时间戳。

## CPU 测试与生产前置

CPU 替身测试核六槽映射、真实 RecordingBackend/分块桥事件顺序、同槽跨决策 call/sequence、失效 session/backend、UUID/显存/精度边界、错误 root/sink、后端故障锁存、桥构造失败清理及日志失败后先关闭环境。联合旧原生 factory/guard 测试执行；fixture 不启动 GPU、神经或环境，也没有证明真实 native 端到端兼容。

真正运行前仍需独立实现和审查：新合同的 HOST 生产驱动，两个日志/帧/latent 的原子性与独立跨日志消费，signal/bank/Q/上下文重算与 native tensor IO census，结束/失败终态、前后模型与精度快照的持久绑定；总存储、内存、显存、wall/外部 systemd 超时与设备/仿真准入；并将新适配器、消费者和完整来源重新冻结到经独立可信时间戳的前瞻合同。再由主线程单独协调和授权。`PREP01` 早于本适配器，不把它解释为对新接口的预注册。P1 未通过不变；无生产命令、无科学通过。
