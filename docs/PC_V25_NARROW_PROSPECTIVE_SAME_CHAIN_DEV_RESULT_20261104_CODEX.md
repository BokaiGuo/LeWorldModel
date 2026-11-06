# 窄束 v25 最终同链机制：开发结果

按[事前协议](PC_V25_NARROW_PROSPECTIVE_SAME_CHAIN_DEV_PROTOCOL_20261104_CODEX.md)，16 根计划全部完成、封存并经计划审计后，才对全部最终选中链实施机制干预。16/16 干预完成；计划均未弃权。终距 `<0.05` 的计划为 **0/16**，最小终距为 0.08362。

原有机制条件下，16/16 前缀非零，**3/16 满足非即时进展条件；基础机制交集和全对照机制交集均为 0/16**。在两个差分尺度上，selected 分支稳定条件为 1/16。组成条件不能单独替代交集，也未因结果改变阈值。

对照维度的单项计数如下；后缀收益比较使用各前缀各自的零尾作为基准，重规划比较为每前缀三次相同预算优化的均值。各项存在重叠，不能相加或当独立样本：

| 对照 | selected 的原选中尾相对收益更好 | selected 的等预算重规划均值更好 |
| --- | ---: | ---: |
| zero | 10/16 | 16/16 |
| same-bank greedy | 13/16 | 11/16 |
| lookahead | 14/16 | 10/16 |
| random bridge | 15/16 | 12/16 |

每根规划上限为 2,880 prediction-token；每根机制干预另记录 854 次端点调用、9,870 个样本、29,592 token，局部分支复算是额外工作。这些干预成本不含在规划预算内。计划及干预的持久化动作与费用、选中链绑定、同代码确定性干预重建和另一同环境几何终点核对均已完成；这些不是独立复现，也不能证明代码没有其他缺陷。

这是同一窄束开发参照的事前全根机制结果，不能与旧版反馈拼成 Full D2P；任务来源独立性、PC0 声明预算可解性、视觉响应、强基线和真实控制仍未认证。结果未达到所用机制合同，不构成 PC2 或 PC3 通过，也不证明理论失败。面板停止，不替换根、追溯改阈值或选择有利组成条件。

原始计划、逐批预测、干预数组、结果及 `plans_audit.json`、`interventions_audit.json`、`interventions_geometry_audit.json` 在 `le-wm-main/results/pc_v25_narrow_prospective_same_chain_dev_20261104_codex/`。在 `le-wm-main` 依次运行 `.venv/bin/python -m experiments.pc_v25_narrow_prospective_same_chain_dev_20261104_codex plan`、`audit_plans`、`intervene`、`audit_interventions`；另运行 `.venv/bin/python -m experiments.pc_v25_narrow_same_chain_geometry_audit_20261104_codex`。所有阶段拒绝覆盖已存在结果。
