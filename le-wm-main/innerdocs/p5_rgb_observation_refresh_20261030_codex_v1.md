# P5 已知面板逐步观测刷新：固定运行合同

范围固定为先前已公开 physical receipt 与 panel 的 164 枝、80 对、两个已知 root；off/on 各 82 枝、40 对，四个 baseline 不删。并非 fresh、holdout 或科学准入。本合同不授权执行 CUDA；生产须主线程独立审核、调度。

每枝五次独立的一步预测。第 h 次的 previous/current/target 分别为 plan `(h+3, h+4, h+5)`，h=1…5；动作是该枝原始命令 `commands[h+4]`（零基下标）。对于 h>1，前一步预测的目标帧现作为真实当前观测输入，这是显式 teacher-forcing / 观测刷新，不是五步开环。每步均重新 encode 相邻两张真实 RGB，然后仅调用一次 predict_next；不反馈预测状态。不使用目标时刻图像、物理状态、goal、位置或速度作为输入。物理 receipt 是先前已开放的历史记录，prepare 解析它以获取绑定元数据，但不把状态或目标标签写入输入合同；不访问未开放 holdout 或保留根。

每帧复核原 PNG SHA、RGB 像素 SHA 以及 224→64 OpenCV INTER_AREA、CHW float32/255 的模型输入 SHA。时间、overlay、branch、task、selection、原始十动作及每步动作均固定并校验；H1 与旧五步合同图像和动作绑定。每个准备文件记录既有 checkpoint、源码和固定 panel 哈希；运行前以用户提供的准备文件 SHA 验证，再从原固定来源逐字节重建，并重新 materialize 所有输入。无动态筛选、补行、分母缩减或自动重试。

预期 820 次 encode 和 820 次 predict；journal 对每次调用在执行前持久写 attempt，成功后写 returned 及六维 float32 输出和哈希，失败保留 failed/partial。start 记录 checkpoint SHA、CUDA 设备、自报告 strict load/frozen/eval、版本和本 runner/输入合同 SHA。进程内时间上限只在调用间检查，不能中断卡住的 CUDA 调用；生产前仍需专门的外部硬超时、设备显存准入、前后精度/模块状态采集和针对本 journal 格式的独立验收。现有五步监督/验收工具不能直接当成本新协议的验收。

观察权限与开环 MAIN02 不同，输出可用于描述性诊断，但不支持称两者严格同条件，也不能将接收额外状态的 JEPA 标成 RGB-only。该 RGB 模型原训练使用物理状态标签；本准备无任何训练、CUDA、神经前向或新物理。
