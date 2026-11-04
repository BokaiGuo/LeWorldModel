# v7 PushT 同候选库物理排序：开发结果

## 结果和边界

在已暴露 episode 0 的第三轮同一物理边界，事后重放预先保存的九条候选、实际 selected 和 carry，共 11 条；所有分支通过 RGB guard 并执行五步，没有终止或截断。每条分支都重放相同 15 步暖身及前两轮实际动作，重放边界的 RGB、动作历史、目标和物理状态与原记录一致。

| 原模型分数排序与下列量的排序 | 一致／有效对数 | 比例 |
|---|---:|---:|
| 实际终点 RGB 的目标 latent MSE | 43/55 | 78.2% |
| 物体位置误差 | 10/34 | 29.4% |
| 物体角度误差 | 12/34 | 35.3% |

仅排除分数或所比较指标的精确并列；候选在同一个状态且高度相关，这些对数不是独立样本，也不是统计显著性或泛化证据。候选库、selected 和 carry 的分数来自不同批次，极小差值可能受浮点批次影响。

实际 selected 的模型分数为 1.3150、实际 RGB latent MSE 为 1.9387，物体位置误差 110.9681、角误差 2.3008；其物体姿态与 zero 候选相同。一个不同候选的实际 latent MSE 为 1.8013，角误差降至 2.1635，但位置误差升至 115.8754。由此可见该根上 latent 排序、预测误差与具体物理指标并不等价；不能据此断言唯一原因、证明接口 bug，或据后验物理结果替代原来的选择。

## 可复核材料

- [事前诊断协议](PC_V7_PUSHT_SAME_BANK_PHYSICAL_RANK_DEV_PROTOCOL_20261104_CODEX.md)
- [运行器](../le-wm-main/experiments/pc_v7_pusht_same_bank_physical_rank_dev_20261104_codex.py)
- [完整结果及 660 条逐步日志](../le-wm-main/results/pc_v7_pusht_same_bank_physical_rank_dev_20261104_codex_result.json)
- [封存清单](../le-wm-main/results/pc_v7_pusht_same_bank_physical_rank_dev_20261104_codex_manifest.json)；[末帧数组](../le-wm-main/results/pc_v7_pusht_same_bank_physical_rank_dev_20261104_codex_arrays.npz)
- [独立 CPU 记录审计](../le-wm-main/results/pc_v7_pusht_same_bank_physical_rank_dev_20261104_codex_audit.json)及[审计脚本](../le-wm-main/experiments/pc_v7_pusht_same_bank_rank_audit_20261104_codex.py)：核对依赖与文件哈希、11 条来源和评分、660 条事件、候选动作、排序，以及 selected 的状态与 RGB 与原运行一致。此审计没有独立重跑物理或重编码 RGB。

诊断不是 PC2 机制干预，不是 PC3 基线确认，也不能宣告 PC4 通过。模型评分与物理姿态错位值得进一步在预先固定、来源合格的多根设计中检验；这次结果本身不授权更改冻结评分或挑选候选。
