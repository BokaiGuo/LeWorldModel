# 同预算候选替换的事后优化诊断

按[协议](PC_FEEDBACK_COMMON_TAIL_BUDGET_REUSE_DIAGNOSTIC_PROTOCOL_20261103_CODEX.md)，
对每个已暴露任务和五个既有前缀，保留旧 CEM 的前 119 条候选，并以反馈
实际执行尾部替换第 120 个名额；每个前缀获得同一尾部，共比较 120 条。
所有旧候选及新增尾部的精确动力学评分均重新计算。

| 前缀 | 旧 120 次 CEM 平均最优末距 | 替换后平均最优末距 | 共享尾部胜出根数 |
|---|---:|---:|---:|
| selected | 0.54048 | 0.48155 | 19/24 |
| zero | 0.57935 | 0.57483 | 2/24 |
| greedy | 0.51991 | 0.51298 | 5/24 |
| random bridge | 0.51842 | 0.51868 | 0/24 |
| lookahead | 0.50649 | 0.50674 | 0/24 |

4 个非即时根（2、8、9、15）中，selected 的反馈尾部均优于该前缀原先
120 次 CEM 搜索的最好结果。替换后，非即时、共同尾部相对变化及候选
最低分同时优于 zero/greedy 的交集从原诊断的 0/24 变为 4/24。
另以**全部原 120 条加共享尾部**做 121 条事后核对，这四根仍满足相同
比较，因此交集变化不只是删去第 120 条旧样本的结果；该附加核对不等
预算，不用于正式比较。

这一结果显示原有限 CEM 未找到已知更优尾部，不能把原干预的重规划
负结果直接归结为前缀没有可利用收益。但共享尾部来自 selected 前缀下
后续两轮反馈，生成本身使用额外计算及观测，并非对各前缀公平可得的
事前候选；重用前 119 条也不是让 CEM 接收该尾部后重新迭代。故 4/24
不是事前同链机制见证、充分后缀优化或 PC2 通过。原实验和负结果均
保留。下一步应预先固定各前缀同初始化、同实际总资源与同反馈机会的
优化充分性诊断，不能对 selected 单独注入后验尾部。

[原始记录](../le-wm-main/results/pc_feedback_common_tail_budget_reuse_20261103_codex.json)
SHA-256：`cb1383802cf8df67e14543b11d7bbb50d596ddc6c17a64adbef2992a8efa5a50`。
以下回归在 `le-wm-main` 运行，结果 **4 passed**：

```bash
.venv/bin/pytest -q -p no:cacheprovider \
 tests/test_pc_feedback_common_tail_budget_reuse_20261103_codex.py \
 tests/test_pc_response_feedback_executed_tail_intervention_20261103_codex.py \
 tests/test_pc_response_diverse_beam_same_chain_diag_20261103_codex.py \
 tests/test_pc_response_diverse_fixed_terminal_exact_dev_20261103_codex.py
```
