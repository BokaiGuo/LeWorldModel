# PointMaze V7：task-contrast 与 goal-balanced candidate-bank 修复

## 完成状态

已完成一轮全新的、GPU 执行的 sealed replication：先构建 root/action
disjoint panel，再冻结 calibration 选择，之后才物化 outcome。结果位于
`le-wm-main/results/pointmaze_v7_goal_balanced_task_contrast_replication_20261021_cuda/`。

这不是旧 V6/V7 root 的重跑。原始 2,000-root pool 的历史 panel 已排除
1,916 个 root，无法同时容纳 144 个 evaluation root 与 64 个 design root。
因此新增了两个独立 MuJoCo trajectory extension；本次 panel 的 state/action
pool 有 2,640 roots，并显式排除了第一轮 V7 的 evaluation/design roots 和
candidate hashes。

## 修复对象

V6 的 effect-MSE 模型将 successor effect 再转换为 goal distance，可能在
action ranking 中丢失 task-relevant ordering。V7 直接以 source-only realised
successor 的 task cost 做 scalar regression + within-root pairwise ranking，比较：

- history-conditioned FiLM task ranker；
- 参数预算匹配的 action-only task ranker；
- calibration-only margin-gated residual policy；
- action-only fallback。

候选库也从 task-agnostic state-interaction screening 改为在 64 个独立 design
roots 上、对预先声明的 `[0.8, 0.8]` target 最大化 oracle-winner entropy，同时
保留等能 pulse-order interaction pairs。design roots 不与 train/calibration/outcome
roots 重叠，也不读取其 outcome。

## 协议修复

发现并修复了一项真实时序漏洞：首次 V7 评分后，calibrate 仍可能覆盖 frozen
selection/model。现在只要 frozen JSON、model 或 held outcome 的任一文件存在，
calibrate 就会 fail closed。`tests/test_pointmaze_temporal_ood_v7_task_contrast.py`
覆盖该回归。

## 新 replication 的结果

候选设计已经改变：64 个 design roots 的 winner 覆盖 12 个 held candidates，
最大单一 winner 占比为 `0.21875`，normalized winner entropy 为 `0.75098`。
因此该测试不再主要奖励一个固定的全局 action prior。

但 V7 仍未通过 authorization：

| quantity | value |
| --- | ---: |
| selected policy | recurrent task ranker, hidden 64 |
| calibration authorization gain vs action-only | `+0.00667` |
| authorization 95% bootstrap CI | `[-0.01382, +0.02664]` |
| held recurrent mean regret | `0.04973` |
| held action-only mean regret | `0.04504` |
| held recurrent gain vs action-only | `-0.00469` |
| held 95% bootstrap CI | `[-0.03838, +0.02135]` |
| deployed policy | action-only fallback |

`debug_opened_outcomes=false` 且
`outcome_labels_generated_after_frozen_selection=true`，所以这是有效的 offline
held-out consequence；它不是 live RGB closed-loop 或 universal-world-model
evidence。

## 结论

截至该面板，负结果不再可归因于固定 bank 的全局 dominant candidate、action
support shift、旧 root 复用、提前打开 outcome，或 V7 selection overwrite。
在这个 frozen visual representation 和 20-step PointMaze bank 下，直接优化
task contrast 仍没有稳定把 history 转为 transfer gain。正式控制器应保持
action-only fallback；V7 ranker 仅保留为机制诊断，而不能被声称为已解决的
规划修复。
