# SRA 强制 guard 入口与预算

`codex_sra_guarded_budget_20261030.select` 是新的独立入口：它自行构造既有原生 RGB encoder 和在线 SRA `run_bank` planner，不接受调用者指定 planner、encoder 或旧 winner。它在 encoder 前后和 planner 前后，各执行一次 checkpoint guard；proof、权重、配置、加载器、guard、入口和 PushT 图像转换源码须出现在方法 manifest 中，manifest 摘要绑定已有 selection receipt。目标及原生 SRA 来源仍由既有在线接口核查。模型和 replay 由调用者提供，生产路径要求既有 planner 核查二者是同一原生 replay 模型对象。

每次检查进入前先在 durable journal 中预留 `setup_steps=1` 和调用者指定的正整数 `wall_ms`；检查完成且观测单调时长不超预留才记录 return。失败或超时留下 pending，拒绝 selection。四个检查合计预留 4 个 setup step；独占写入 guard receipt 后，再以第 5 个 setup step 将其摘要绑定 journal，成功 selection 必须在该标记之后。guard receipt 保存四段实际单调时长与预留，保存消费者 `consume` 验证来源、SRA audit、各项预留与返回以及其相对编码/规划/选择的顺序。发生错误时已写入的 SRA audit 或 guard receipt 可能是孤立产物，不代表选择成功。

墙钟预留是**事后超限拒绝**，不是抢占式硬超时；journal 的 completed wall 记录预留量，不是实测时间，实测纳秒在 guard receipt 中。每次检查的计时包括身份检查内的读取及模型状态比较，但不覆盖其前后 journal fsync、receipt 写入、既有在线接口及 planner 的其他文件读取、模型装载、proof 构造或全部设备工作。I/O 字节、设备传输、FLOPs 等仍为未知，不得将未知记为零；现有资源账本不构成等总成本证明。

测试专用 `_select` fixture 分支在 audit 和 guard receipt 中标明 fixture；公开 `select` 不暴露替身参数。CPU 测试使用替身 encoder、编译器、model 和 replay，实际运行原有 `run_bank` 的候选选择逻辑；未运行 GPU、真实推理或物理。检查不能提供硬件证明，也不能排除同进程 monkeypatch、并发修改、隐藏依赖或 Python 非 tensor 状态差异。新入口的 guard 是该入口的强制路径，旧 SRA 和旧 online API 仍存在，**不能声称整个仓库的其他入口都被强制保护**。本接线不代表完整 D2P、物理效果、独立确认或等预算通过。
