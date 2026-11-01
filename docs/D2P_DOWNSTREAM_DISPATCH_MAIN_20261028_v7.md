# P2–P5 后续推进派发

本轮用户明确要求后续 P 阶段同步推进。以下为已经投递的任务，不代表实现或科学验证已经完成。保留当前独占任务，完成后续接，避免打断造成半成品；MAIN 负责最终审计和真实实验。

| 阶段 | 实际续接工作 | 负责人/独占输出 |
| --- | --- | --- |
| P2 | 从已有真实组合动作库接入同根组合/无组合执行准备，复用已有 runtime，不重写文档合同 | Jason，`agent_p2_execution_assets_20261028_v1`；先完成当前 kinodynamic runner |
| P3 | 固定请求5步的 CUDA 生成与物理执行严格分工，原生早停保留，不 padding | Bohr，`agent_p3_fixed_execution_cuda_20261028_v1`；Hegel，`agent_p3_fixed_execution_physical_20261028_v1` |
| P4 | 对已有 RGB-only 反馈控制器补真实执行输入、回调/预算与原生指标接线，避免复制既有闭环 | Nash，`agent_p4_runtime_assets_20261028_v1`；先完成三臂 assets |
| P5 | 找到现存 training-free source-native 冻结复制入口，精确检查 checkpoint/normalizer/许可，分开同模型多任务与跨架构 | Pauli，`agent_p5_frozen_replication_assets_20261028_v1`；先完成四基线 readiness |

Poincare 继续 P1 真实能力探针配套实现；MAIN 此刻实际运行 probe v2 的 saved-only prepare（session62369），检验此前8MiB工程容量拒绝的完整闭包修复；这不是新的神经或物理实验。

科学依赖不变：P1 尚未建立，不能以实现、prepare或单测打开依赖它的 P2–P5 正式结论。下游当前允许实现/校准准备，任何真实执行及证据解释由 MAIN 另行审读。神经只用主机 CUDA，不训练/适配冻结模型，不放宽门槛、不按结果挑根，保留全部旧源码和失败证据。
