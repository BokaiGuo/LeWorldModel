# P2 双根真实 native 回调与启动前认证接缝

`Session` 使用已重建双根桥接，以原生 `NativeTree(model, device, task, sink)` 为每根构造真实回调，并分配独立的原有 H15 固定资源账本。模型 state-dict 与全 buffer 哈希、历史完整文件清单、MATH/no-TF32/no-autocast、模型 eval/frozen/device 和类源码在构造及调用前后复核。非 fixture 模式要求历史 P2 HOST 的完整源文件清单、预期 state-dict 和实际模块类来源；调用处始终拒绝派发。它不加载模型，也不创建设备。CPU 测试通过无张量的占位模型和设备类型替身构造原生对象，没有调用原生 signal/replay，不能成为 HOST 证据。

独立预检报告对历史清单 398 个文件逐一重哈希，并记录本次接缝额外文件哈希、双根桥接摘要和计划预算。报告的外部 SHA 可用于重建验证；其中运行时模型、精度、预算认证均为 false。文件源清单匹配不证明进程真实载入那些文件，也不能证明无隐藏调用。Session 的输入可由调用方提供；即使检查通过也不是可防伪远程证明。

当前没有将这组真实回调派发进双根 lifecycle 或保存其 HOST 成功/失败 transcript：此前双根生命周期只接受明确 RAM fixture，不能通过伪装 backend 类型来接入。开放派发前还需实现真实单槽执行和独立 saved consumer（不把 HOST 标成 fixture）、双根中断/故障与资源预留、运行时前后 model/precision/source 证据持久化及验证、调用覆盖/未记录调用边界、CUDA 设备与进程资源限制、授权和调度。物理仍完全隔离。任何人传入的 flag 都不能开启当前派发。
