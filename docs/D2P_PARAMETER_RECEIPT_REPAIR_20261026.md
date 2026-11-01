# PushT 参数见证补全与正则合同核验

## 本轮完成

`pusht_d2p_full_metric_suffix.refine` 新增原始最终 atanh 参数、参数能量、终态 SSE、正则坐标与正则中心字段。优化、动作、选择规则和旧报告不变。

此前仅存 tanh 后动作：float32 参数 10 会映射为精确的 1，无法由动作反解有限原参数，因而完整目标无法独立复核。这是证据可观测性缺口，不是已证明的动作执行错误。

主机 NVIDIA GeForce RTX 5060 Ti 上，显式断言 CUDA 可用后执行 `tests/test_pusht_d2p_consistent_metric_suffix.py`：**5 passed**。未进行 CPU 神经推理或更新权重。

新增核验包括：

- 饱和动作仍保留原参数 `[10,-10]` 与能量 200；零终态 SSE、阻尼 .01 下目标为 2。
- 非零 nominal 为 .5、目标 1、阻尼 1 的仿射问题，当前 absolute ridge 最优仍为 .5，不是 delta ridge 的 .75。
- 原有 identity-GN 逐位等价、投影度量接线及 CPU 输入拒绝测试均通过。

## 不能推出什么

这证明所测路径满足声明的绝对 atanh 参数正则合同，不证明该正则符合最佳物理控制目标，也不证明它解释历史 P1 失败。PushT suffix-only 的实验没有变成完整 compiler，P1 仍未通过。

未来新实验可以保存更完整的参数见证；旧实验的饱和参数缺失不能事后补造。正则改变须另立协议，以同一批预定校准根进行单因素验证，不得借物理 outcome 挑根、改阈值或打开后续阶段。

## 后续推进：解析合同与采样基线

- H3/H5/H7 独立 NumPy 合同测试已完成；连同报告值注册器及源码封存相关测试，本轮现场运行 93 项通过。H7 使用实际 helper 的 AST 函数体及标号 fixture，绕过归一化，不是神经或物理 fidelity 证据。
- 新增 `d2p_cuda_sampling_cost.py`，只将数值候选搬入 CUDA 并回传逐候选成本，神经模型不降级 CPU。主机 CUDA 2 项接线测试通过；这里使用明确的数值 fixture，不是实际 checkpoint 性能。
- `d2p_sampling_baselines.py` 的 57 项测试通过。uniform random shooting 保留 initial 与 best-ever；另一个内核明确是 `clipped_gaussian_mppi_heuristic`，无 importance correction，不能冒充官方或完整路径积分 MPPI。
- 真实冻结 Reacher 校准 runner 已交由独立任务实现；截至本段写入尚未运行，不计作已完成真实对比实验。预测样本账需包括 initial 和更新后的 nominal，而不仅是 proposal 数。

## 真实采样校准启动状态

`run_reacher_d2p_sampling_calibration.py` 已落盘并经主线程通读，编排 fixture 核对为：启发式 133 个候选成本、随机 33 个，不能忽略初始和更新点评分。新增独立收据审计 `audit_reacher_d2p_sampling_calibration.py`，按 horizon=2 预期各根 predict 样本 266/66/256，4 项 synthetic 审计测试通过。阶段准入与 P0 相关回归 103 项通过。

两次主机 CUDA 执行审批均因 automatic approval review deadline 超时而未创建进程（首次及工具允许的一次重试）。目标目录 `results/reacher_d2p_sampling_calibration2_v1_20261026_cuda` 未出现。**没有新真实校准结果，也没有存活实验句柄。** 不改为 CPU、不转用其它工具绕过审批；需要有效主机执行批准后启动。此项是执行权限等待，不是代码或理论负结果。

## 评分桥准入加固

后续审查发现 `CudaSamplingCost` 原来只检查顶层 `model.training` 和构造时 `requires_grad`，没有拒绝“顶层 eval、子模块 train”，也没有阻止构造后的参数设备或可训练标志漂移。现增加 `_require_frozen_eval`，构造和每次成本评分前逐子模块检查 eval，逐参数检查冻结与同 CUDA 设备。

metadata fake 反例覆盖部分训练模式、可训练标志漂移和 CPU 参数；没有 CPU 神经推理。连同采样和审计测试，本轮 **63 passed, 1 deselected**（明确排除需要主机 CUDA 的桥接测试，不将其计作本轮通过）。这是一项准入防护修复，不是历史实验中发生该漂移的证据，也不能据此解释既有 P1 负结果。

同时已分派固定候选库 PushT caller 的实际接线，要求在每个 prefix 自己的预测状态重新计算 suffix nominal/Jacobian，并用同一度量形成 stage/own-V/非线性终态回执。当前仍为任务执行中，不是已完成完整谱前缀生成、真实物理实验或 P1。

## 切回主线程后的集成修复

用户要求回到主线程后，已停止六个子智能体并保留其现有文件；未完成的 PushT preflight/独立数组审计入口不计作已交付。

主线程修复成本导出失败时仍提前增加 completed 计数的问题：只有 CUDA 成本校验、导出 NumPy 并复核有限值全部成功，才增加 completed；导出异常计入 failed 并中断。原失败刻画测试改为修复回归，相关组合 230 项通过。

继续加强独立采样审计：不再只看动作位于边界内，逐根检查所有方法共用 normalized bounds，独立用 NumPy 从参数重建 tanh、affine normalized action、time-major 原始动作，覆盖 CEM 与两种采样方法。允许与 float32 映射相符的有限舍入；不允许更换动作或时间坐标。新增“plan/report 同时改成另一组合法动作”的反例也会被拒绝，非零时变动作正例通过。最新相关组合 **138 passed**。均为保存产物/数值合同测试，不是新物理成功证据。
