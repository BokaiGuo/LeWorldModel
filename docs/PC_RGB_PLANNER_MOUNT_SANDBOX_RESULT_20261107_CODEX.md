# RGB 规划器文件系统隔离：开发结果

按照[事前协议](PC_RGB_PLANNER_MOUNT_SANDBOX_PROTOCOL_20261107_CODEX.md)，在 bubblewrap 私有挂载和 PID 命名空间中运行冻结 GPU RGB 梯度规划器。规划进程只挂载指定公开输入、checkpoint、代码与运行依赖，不挂入完整研究结果目录、TwoRoom 数据集或用户主目录。结果由命名空间外的评价进程与此前公开计划比较。

- 进程内四个指定路径探针均不可读，包括 `tworoom.h5`、已有规划结果和两个物理评价文件。
- 四个已曝光任务的候选动作、梯度、方向和分数与先前公开输入计划逐元素相同，选中索引均相同；模型和公开输入摘要一致。
- 相关测试 **4 项通过**。本次不重新进行物理评价，原计划此前的真实执行结果不能因此升级为新实验。

首次在外层受限环境运行时，CUDA 初始化报错 304；直接的命名空间 GPU 检查也不能与驱动通信。经批准，在外层受限环境之外运行该命名空间，并只读挂入 `/sys` 和 NVIDIA 驱动信息后，GPU 规划完成。失败阶段的临时目录保留在 `/tmp/pc-rgb-sandbox-xqd50oi0`，未覆盖历史结果。

**范围限制：**本机不支持所需的网络命名空间，因此本实验没有隔离网络；GPU 设备、系统目录和整个实验源码目录仍可见。固定路径探针与命名空间清单说明指定数据未挂入，不等于对恶意代码、设备侧通道、所有系统资源或所有别名的全面安全审计。现有运行是四个开放环梯度开发任务，不是整个 RGB MPC 闭环，也没有建立方法增量或改变 PC0–PC5。

[启动与比较代码](../le-wm-main/experiments/pc_rgb_planner_mount_sandbox_20261107_codex.py) · [命名空间及比较记录](../le-wm-main/results/pc_rgb_planner_mount_sandbox_20261107_codex.json) · [封存计划](../le-wm-main/results/pc_rgb_planner_mount_sandbox_plan_20261107_codex.json) · [测试](../le-wm-main/tests/test_pc_rgb_planner_mount_sandbox_20261107_codex.py)
