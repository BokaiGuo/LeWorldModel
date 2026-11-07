# RGB 梯度规划：进程分离与回归结果

按[协议](PC_RGB_GRADIENT_PROCESS_SEPARATION_PROTOCOL_20261107_CODEX.md)，
将旧开发运行拆为评价端公开输入准备、独立 RGB 规划和独立物理评价。
公开文件只含 episode ID、当前 RGB、目标 RGB、名义动作以及动作
normalizer 均值／尺度；物理状态、成功标签和真实后果不进入该文件。
规划过程不再调用会打开原始数据集的 normalizer 拟合函数。

在同四个已曝光任务、相同冻结模型与名义链上：

| 检查 | 结果 |
| --- | ---: |
| 与旧计划的候选动作逐元素一致 | 4/4 任务 |
| 选中索引一致 | 4/4 |
| 所有候选评分最大绝对差 | 0 |
| 全部五候选物理轨迹与旧评价逐项一致 | 4/4 任务 |

跟踪规划进程及其子进程的 `open/openat/openat2` 后，观察到公开输入、
checkpoint 权重和配置的打开；在跟踪日志中未发现对预先列出的原始
`tworoom.h5`、旧谱计划／物理记录和旧梯度计划／物理记录的打开尝试。
第一次在沙箱内启动 `strace` 因 ptrace 不允许失败；经授权在沙箱外
重新运行后获得日志。这是本次运行的指定路径检查，**不是访问控制
沙箱**：它不能排除未列出的敏感路径、别名、继承文件描述符或其他
数据通道，也不能认证完整 MPC 执行过程。

公开输入白名单、非法字段拒绝、动作归一化梯度及跟踪审计的反例测试，
连同候选生成测试共 **8 项通过**。评价端对公开初始 RGB 与物理重置
渲染逐像素核验。运行模型指纹与旧计划一致，运行前后无变化。

这修复了本实验路径中已确认的进程数据流问题，并复现了旧负结果；
四任务仍是开放环、已曝光开发数据，梯度方法不因此获得新性能或机制
证据，PC0–PC5 不改判。

- [公开规划器](../le-wm-main/experiments/pc_rgb_gradient_public_plan_20261107_codex.py)
- [独立评价器](../le-wm-main/experiments/pc_rgb_gradient_public_evaluate_20261107_codex.py)
- [规划结果](../le-wm-main/results/pc_rgb_gradient_public_plan_20261107_codex.json)
- [物理重放结果](../le-wm-main/results/pc_rgb_gradient_public_evaluation_20261107_codex.json)
- [文件访问审计](../le-wm-main/results/pc_rgb_gradient_public_filetrace_audit_20261107_codex.json)
