# P5 同输入接口与递归敏感性预固定合同

四枝是固定 panel 的四个 baseline：索引 0、41、82、123，即两个已知 root × off/on。它们由既有合同预定，不按误差筛选；不是全部 164 枝，也不是独立或新鲜 root。动作均来自各枝固定五步原始 float32 命令，初始 previous/current 是各枝 plan4/5 的已公开 RGB。无物理目标、goal、位置速度或未开放 holdout 进入模型。

每枝先在完全相同输入上运行一次 `model.forward(image, action0, previous)`，再分别运行 `encode_state` 和 `predict_next`；记录两组六维 normalized state 的逐元素差。差值报告而不预设阈值或宣称相等。其后以显式 encode 输出为初态，运行原始轨迹和六个坐标各 ±0.001 的 12 条初态扰动轨迹，各使用同样五步动作并递归；不刷新真实帧。每个扰动后 float32 初态都记录并核验，中心差商分母为实际存储的 plus-minus 坐标差，不假定它精确等于 0.002。对所有四枝、五 horizon 保存完整 6×6 列向量，不做模型选择或事后挑选。

预算：4 次 forward（内部各包含编码和一步预测）、4 次显式 encode、264 次显式 predict、52 次非神经状态 reset；总计 272 次显式神经 API 调用，以及 forward 内部调用。只是调用数，不是 FLOPs、延时或显存上界。输入图像每次包含两张 float32 3×64×64、动作五个 2D float32；checkpoint 93,842 参数。实际 GPU 内存/耗时、日志总大小和 host 可用容量未实测。

运行器记录每次 attempt/returned 或 failed 的哈希链、输入/动作身份、每次 predict 的实际输入状态与输出 float32 bytes SHA、前后模块/参数/归一化/精度快照；独立 review 命令以外部 journal SHA 校验完整固定调用数、次序、扰动、递归状态衔接及快照。这里的 saved 审核仍是进程自报告，不是独立硬件证明。超时、失败或未完成的记录不被当成成功，保留 partial，不自动重试。

## 主线程生产前提

本代码尚无本协议专用的外部监督器、显存准入接线和 systemd cgroup 认证。`--max-seconds` 仅在调用间检查，无法中断卡住的 CUDA；**不得直接作为已具备硬超时保护的生产命令执行**。主线程须先独立审查源码与合同 SHA、制定 GPU UUID/显存下限与磁盘/日志限制、实现或批准适配本新协议的外部进程组硬超时和 no-retry 监督，核验 systemd watchdog/cgroup、驱动与实际 GPU 可见、精度及审计接线，且明确授权后才可安排生产。现有五步或刷新监督/验收不能直接视为已适配本 journal。指定 CUDA 0 必须通过 UUID 映射到选定物理设备，不允许 CPU 推理回退。

有限差分是该 checkpoint、输入和有限步长下的局部数值敏感性，不是全局稳定性定理、理论因果证明、环境响应或新鲜确认。模型历史使用物理标签监督；off/on 外观不同，不应混为同分布。
