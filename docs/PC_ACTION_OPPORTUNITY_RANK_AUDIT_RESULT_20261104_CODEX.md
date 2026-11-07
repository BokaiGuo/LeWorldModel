# Action-opportunity descriptor ranking audit

日期：2026-11-04。性质：对既有精确动力学 Stage-1/Stage-1b 输出的独立统计重算；不重跑模型查询，不修改历史 `result.json`，不是独立确认，也不是 PC1–PC3 通过证据。

审计入口：

`le-wm-main/experiments/audit_pc_action_opportunity_rankings_20261104_codex.py`

审计输出：

`le-wm-main/results/pc_action_opportunity_rank_audit_20261104_codex/panel.json`

来源哈希、每个面板的完整 root×arm 单元和候选级重算结果均保存在输出 JSON 中。

随机探针已用修正后的运行器在 `.venv` 中重跑为
`results/pc_action_opportunity_random_probe_dev_v2_20261104_codex/result.json`；
v2 补写了 `baseline_score`，旧 v1 结果仍保留。

## 1. 修正内容

历史两个运行器使用稳定排序为并列值分配不同秩。审计改用平均并列秩，并同时报告：

- 候选级描述性 Spearman；
- 按 `root × arm` 聚合后的任务单元统计；
- 控制即时分数、剩余时域以及可用时的零尾代价后的 rank-residual 诊断；
- 来源文件哈希与缺失字段状态。

候选行不是独立任务，因此任务单元统计用于限制过强的 pooled-candidate 解释。它仍是开发诊断，不是预注册确认统计。

## 2. 结果

| 面板 | 候选数 | 候选级 `A` vs `B` | 候选级 `A` vs `-best` | root×arm 级 `A` vs `B` | root×arm 级控制后 `A` vs `-best` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 固定/常值-交替探针 | 2476 | 0.1126 | -0.2255 | 0.5591 | -0.0075 |
| 随机探针 | 2476 | 0.6833 | -0.2667 | 0.4126 | -0.0607 |

固定探针和修正后的随机探针 v2 均包含 `baseline_score`、`immediate_score` 和剩余时域；原随机 v1 没有记录 `baseline_score`，因此旧 v1 审计只控制 `immediate_score` 和剩余时域，并明确标记字段缺失。正式比较使用 v2 审计输出，不能把 v1 与 v2 的偏相关视为同一统计合同下的直接比较。

## 3. 当前能支持的结论

1. 原运行器的唯一秩实现是代码问题，未来运行器已改用平均并列秩；旧结果保留，不被静默重写。
2. 随机探针在候选级仍显示较高的 A/B 相关，但在 root×arm 聚合并控制可用协变量后没有正向质量关联。该结果不能支持把 A 描述作为 retention 依据。
3. 固定探针的候选级与任务单元级结果差异较大，说明把 2476 个候选当作独立样本会夸大证据。
4. 当前统计仍不能回答完整后果向量是否提供超出 probe-minimum 的信息，也不能证明真实物理后果或视觉评分有效。

## 4. 下一项必要实验

在不修改正式 planner 的前提下，新增同成本诊断：

- 保存完整后果向量、后缀动作身份和顺序；
- 比较完整向量、probe-minimum、普通 lookahead 及动作身份打乱；
- 使用未参与构造的后缀和独立重规划随机流；
- 统一记录零尾代价、剩余时域、失败状态和全部查询成本；
- 以 root/task 为统计单位，保留候选级结果仅作描述；
- 随机探针运行器必须补写 `baseline_score` 或明确采用同一可用基线后重新封存。

在该实验完成并显示样本外预测性前，不接入 consequence-aware retention，不据此宣称 PC2 或 PC3。
