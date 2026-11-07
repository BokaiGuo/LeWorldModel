# TwoRoom 精修评分路径交叉核查结果

按[事前协议](PC_TWOROOM_REFINEMENT_SCORE_CROSSCHECK_PROTOCOL_20261107_CODEX.md)，使用冻结 RGB LeWM、相同动作归一化与 math SDPA 设置，对上一轮 40 个已曝光任务的全部 720 条候选，用原生批量 RGB 接口重算评分，与原带历史逐条接口比较。

| 指标 | Adam 候选 | 随机候选 |
| --- | ---: | ---: |
| 最大绝对评分差 | 0.000889 | 0.000889 |
| 超过预定 0.005 容差的任务 | 0/40 | 0/40 |
| 最低分索引改变的任务 | 0/40 | 0/40 |
| 最终动作改变的任务 | 0/40 | 0/40 |

因此，在本候选集、同模型及此后端下，未发现评分路径或批量计算改变上轮选链的证据。这不证明其他输入、后端或 checkpoint 的接口等价，也不能排除训练分布、动作时钟、物理模型或目标评分存在其他问题。批量重算使用同一模型实现，不是独立模型复现；未使用真实后继重新选择。

测试覆盖近并列评分发生索引交换时的检测、动作完全相同的情况、非法输入、原始结果绑定及逐任务汇总复算；与带历史和上轮精修测试合计 **12 项通过**。`code-debugging` 用于核对归一化、时间维度与候选选择，并分别报告数值误差和实际动作变化。

本结果不改变上轮开发结论，也不证明行动后果描述的独立价值或 PC0–PC5 通过。逐任务评分、最小分差、候选身份见[原始记录](../le-wm-main/results/pc_tworoom_refinement_score_crosscheck_20261107_codex.json)，实现及测试见[核查程序](../le-wm-main/experiments/pc_tworoom_refinement_score_crosscheck_20261107_codex.py)与[测试](../le-wm-main/tests/test_pc_tworoom_refinement_score_crosscheck_20261107_codex.py)。
