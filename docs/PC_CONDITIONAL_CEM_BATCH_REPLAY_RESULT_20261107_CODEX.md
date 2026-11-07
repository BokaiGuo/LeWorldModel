# 条件尾部 CEM 批大小敏感性审计结果

对已封存 TwoRoom roots 20–49、两臂各三轮的全部 11,520 条候选及每个 root 的两个 incumbent，在同一冻结 RGB LeWM 上按[预定审计协议](PC_CONDITIONAL_CEM_BATCH_REPLAY_PROTOCOL_20261107_CODEX.md)重新评分。每条链记录五个预测时刻，incumbent 单独评分。

| 重算批大小 | 相对原始逐时刻分数最大绝对差 | elite 集合发生变化的轮数 | 最终选中动作改变次数 |
| --- | ---: | ---: | ---: |
| 16 | 0.0148011 | 0/180 | 0/60 |
| 32 | 0 | 0/180 | 0/60 |
| 64 | 0.00650025 | 0/180 | 0/60 |

32 的分批方式与原运行一致；本机本次逐项复算完全一致。16 和 64 有数值差异，但本固定池中均没有改变 elite 集合、最终选择索引或实际动作，包括两臂曾成功结果不同的 root 38。

实验对已曝光、固定的动作池进行评分，不是重新运行自适应 CEM。精确结论限于此次 checkpoint、设备、动作池、前处理和批次分组；不能外推到其他运行，也不能据此认证环境时钟或排除其他代码问题。本次没有发现足以解释上一轮退化的批大小选择错误，不能据此断言理论或模型必然存在问题。

新增固定池决策审计测试与原条件 CEM 测试共 **6 项通过**，覆盖 elite 和动作改变、相同动作不同索引及非有限分数拒绝。审计计时仅为离线复算，不代表规划器整任务成本；本次不改变历史结果或 gate 判定。

原始记录：[逐 root 与汇总](../le-wm-main/results/pc_conditional_cem_batch_replay_20261107_codex.json) · [全部重算分数](../le-wm-main/results/pc_conditional_cem_batch_replay_20261107_codex.npz) · [审计代码](../le-wm-main/experiments/pc_conditional_cem_batch_replay_20261107_codex.py)。
