# 行动后果保留：精修与最终链来源审计

本审计对既有 ranked 端到端开发面板做事后来源重建，不新增模型预测、不改变原规划结果。它使用已通过动作与预测核对的调用日志，按共享多起点 CEM 的种子批次和各起点 generation 边界定位实际 incumbent；检查 seed 分数、generation 样本数和最终配对调用，并对动作使用 float32 精确匹配。

## 结果

| 候选方向 | min/effect 返回完全相同动作的配对 root | 两臂合计弃权 |
| --- | ---: | ---: |
| random | 11/16 | 0/32 |
| spectral | 15/16 | 0/32 |

谱方向虽在 16/16 根的终距于 `1e-12` 内相同，但只有 15/16 是相同动作；不能把等分数当成相同轨迹。不存在通过弃权产生相同结果的情况。

对动作完全相同的配对，按两臂中与 incumbent 匹配的精修起点标签统计：

| 方向 | carry | global_warm_start | retained_final_beam | 其他已评分来源 |
| --- | ---: | ---: | ---: | ---: |
| random（11 根） | 4 | 2 | 5 | 0 |
| spectral（15 根） | 4 | 3 | 4 | 4 |

“其他已评分来源”包括 `depth2/candidates` 和 `depth1/successor_reachability`，表示进入精修时的 incumbent 在这些阶段已被评分。`carry` 来自共享全局候选库的选择；`global_warm_start` 是另一全局起点。`retained_final_beam` 是末层保留的精修起点，**不自动意味着响应组件造成了最终发现**。

在全部 64 次递归运行中，记录的 `incumbent_source` 为 `full_refine` 的有 62 次，其余 2 次为 `depth2/probes`。在日志中精确匹配最终 incumbent 的精修起点标签数如下（同一动作若出现在多个起点可多次计数，不能解释成互斥因果归因）：

| 方向 | carry | global_warm_start | retained_final_beam | 其他 |
| --- | ---: | ---: | ---: | ---: |
| random（32 次运行） | 9 | 6 | 15 | 2 |
| spectral（32 次运行） | 8 | 6 | 8 | 10 |

## 解释和下一步

共享的全局起点能解释一部分相同最终链的**来源路径**，但不是全部：多根相同动作来自 `retained_final_beam` 或其他已评分来源。更重要的是，共享起点、全局 incumbent、随机流、精修预算和先前候选之间存在依赖；该事后日志不能证明“全局保底抹平了保留增益”，也不能证明某条链依赖行动后果组件。

下一项有识别力的比较应在事前冻结的候选与优化机会下，直接比较不同保留起点的后续可实现质量，并保留相同全局库作为正式端到端保底；不能为了制造差异删除 global CEM。之前的有限 CEM 漏检限制和完整成本限制仍然适用。新开发面板未建立来源独立，也不能作为 PC2/PC3 通过证据。

入口：[来源重建脚本](../le-wm-main/experiments/audit_pc_action_effect_refinement_lineage_20261104_codex.py)；完整逐起点记录及源文件哈希位于 `le-wm-main/results/pc_action_effect_ranked_end_to_end_dev_20261104_codex/refinement_lineage_audit.json`。这是同工作区、同优化器的事后审计，不是独立复现。
