# P3 六槽 HOST 安全准备：未授权执行

本次新增六槽准备、注入式生命周期、数值分块传输和 callback 记录核查；不包含可启动的 HOST runner，也没有完整原生 saved strict consumer。所有测试使用 RAM 替身，未运行 GPU、模型或物理环境。方案形成于已有反馈结果之后，不声称事前注册、科学通过、等 FLOPs 或泛化，也不改变既有科学门。

准备代码重建已有四槽 source closure，追加三策略 callback、六槽传输、自身和新消费者源码的哈希，固定两个已知根、六槽顺序、每槽原完整数值轴 caps、四次 packet/encode、20 计划动作。按旧完整路径每槽 warm11 加 plan21 为 32 帧，六槽需要 192 帧；旧四槽 128 帧限制不能复用。六槽存储、wall 和显存总上限尚未审定，均保持未知，`storage_admitted=false`、`host_runner_ready=false`；继承的四槽限额不是六槽许可。

分块传输使用独立资源名、0–5 槽位、固定分片长度与摘要检查；其本身不认证来源或行为。注入式生命周期固定顺序、保留 setup/cleanup 错误，setup 内部未返回的对象由 factory 清理；它不是 HOST 驱动。callback 消费器检查独立保存的终态与事件、数值账本 caps/attempted/completed、调用尝试/返回/失败、已付费用和完整槽的策略及 packet 边界，并可消费完整六槽生命周期记录。缺槽保留空值，所有输出都拒绝原生认证和科学通过。该消费者接收内存记录，尚未接持久 journal、原生物理帧、source root、encoder、设备/模型和原生调用独立重建；因此不应将它称作完整 saved strict。未知物理步数、encoder kernel 和 FLOPs 均保持 null。截断或有未结 callback 记录的生命周期由该入口拒绝，不声称已经实现部分 journal 的恢复。

后续真实执行前仍需独立实现并审读：六槽实际 factory/lifecycle 及异常清理，192 帧容量和全类别存储预留、磁盘余量、总 wall/显存 cap，独占耐久 journal 和失败终态，原生 meter、真实逐步动作与 warm/root/goal/帧/encoder、记录模型 hook 与上下文数学的完整独立消费校验。还需独立审读、可信不可变时间戳和单独执行授权；不得借用旧四槽 receipt 或 fixture 证明实际运行。

验证：禁用 CUDA 可见设备后运行新测试与三策略 callback 测试，22 项通过；真实源码闭包只读重建和验证为 24 个源码、6 槽、每槽 27 轴。测试中的数值 backend 使用独立 RAM 记录，不代表原生调用记录完整性。
