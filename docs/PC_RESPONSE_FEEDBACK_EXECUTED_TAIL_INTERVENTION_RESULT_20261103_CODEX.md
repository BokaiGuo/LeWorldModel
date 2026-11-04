# 固定终点反馈实际执行尾部：事后机制诊断

按[事前诊断协议](PC_RESPONSE_FEEDBACK_EXECUTED_TAIL_INTERVENTION_PROTOCOL_20261103_CODEX.md)，
对 24 个已暴露任务，将 V1 反馈的三个实际派发段拼接；对五个已有前缀分别
重放同一个零尾部和该根实际执行尾部。重规划数值复用原先已封存的同前缀、
同预算干预，未重新采样。

| 前缀 | 零尾部平均末距 | 实际执行尾部平均末距 | 相对零尾部变化 | 原同预算重规划平均末距 |
|---|---:|---:|---:|---:|
| selected | 0.85154 | 0.48165 | -0.36988 | 0.54048 |
| zero | 1.10586 | 1.14585 | +0.03998 | 0.57935 |
| 同库 greedy | 0.73785 | 0.66451 | -0.07333 | 0.51991 |
| random bridge | 0.76480 | 0.81816 | +0.05336 | 0.51842 |
| lookahead | 0.74358 | 0.79373 | +0.05014 | 0.50649 |

selected 前缀非即时进展为 4/24；它的实际执行尾部相对变化分别优于
zero、greedy、random bridge、lookahead 的根数为 17、19、22、22。
非即时、相对变化及各自同预算重规划同时优于 zero 和 greedy 的交集为
0/24；同时优于全部四个控制的交集也为 0/24。

这是反馈后拼接的实际执行尾部，不是第一次决策事前选定的完整链。对其他
前缀使用该尾部是开环反事实，不能称为其他前缀下的闭环行为。该结果不能
证明 PC2 通过、PC3 优势、RGB 控制或独立复现；它也不证明理论失败。
首先应继续排查候选覆盖、保留与资源分配，或按明确新假设另立开发版本，
不能重开已停止的旧构造。

[原始结果](../le-wm-main/results/pc_response_feedback_executed_tail_intervention_20261103_codex.json)
SHA-256 为 `03d8f5047f7f2d06ccaf07e287916dca2e1af29e1683e6d9a147f205a3c156a0`，
包含逐根任务、实际链、对照前缀、终距与判定，并绑定协议和源结果哈希。
以下命令从 `le-wm-main` 执行，逐条重放本诊断、原同链干预与固定终点
反馈，结果 **3 passed**：

```bash
.venv/bin/pytest -q -p no:cacheprovider \
 tests/test_pc_response_feedback_executed_tail_intervention_20261103_codex.py \
 tests/test_pc_response_diverse_beam_same_chain_diag_20261103_codex.py \
 tests/test_pc_response_diverse_fixed_terminal_exact_dev_20261103_codex.py
```
