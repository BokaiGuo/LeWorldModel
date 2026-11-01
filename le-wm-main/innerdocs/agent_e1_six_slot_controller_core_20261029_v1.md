# 六槽 controller 算术核心（部分实现，非 CUDA runner）

本新文件是 RAM/fixture 可运行核心：沿原 3×24 H2 CEM 数学支持全部六槽，保留根 seed、原目标和候选，winner 后从保存 raw10 调用 whole 与 1+1 两种 replay，最后必须对完整六槽结果给出 seal 才允许比较。逻辑 packet attempt 在回调之前扣除，失败锁死、不可退费或重试。只比较保存 context 的 JSON 字节，不称浮点 ULP 验证、神经 parity 或科学通过。

**这不是完整交付**：此核心未实现 CUDA model loader/权限、真实 module 计量、精度和 loaded bytes 前后核查、耐久 Store/journal、全量 strict saved consumer、内存/磁盘/wall 实证证明、raw token 的显式保存重载和真正 module replay IO。Attempts 仅 fixture 逻辑计数，绝不能当实际模块调用 receipt。原 beam.SavedDispatch 的每资源 144 上限不容本协议 replay 追加4，必须独立新实现；不可 monkeypatch 扩容。代码目前无生产 CLI、不能授权生产/发布动作。科学门保持 false。

本协议仍应先由 MAIN 实际 prepare，提供路径及其 SHA；不预填尚不存在的 authority。额外资源以新实际 writer 和逐字节 worst-case census 证明，不能把旧四槽64MiB外推六槽。旧文件不改。下一实现需源闭包、真实 loaded model、所有 target 持久 barrier、144+4每组真实 AE/API/predictor/proj hooks、全候选/动作/context/随机数/journal 重建以及 replay byte/ULP 和完成写盘的全量 saved consumer。worker 未运行 NN/physics/production/fullstrict。
