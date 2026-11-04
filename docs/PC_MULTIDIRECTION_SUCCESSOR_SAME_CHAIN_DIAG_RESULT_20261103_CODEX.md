# 多方向后继编译器：最终链事后机制诊断结果

按[诊断协议](PC_MULTIDIRECTION_SUCCESSOR_SAME_CHAIN_DIAG_PROTOCOL_20261103_CODEX.md)，
固定 24 根编译器精修与弃权后的最终链，以 selected、zero、同库 greedy
及历史 random bridge、lookahead 前缀做共同后缀和相同 120 条 CEM 重规划。
后两类是事后历史对照，不能称为同方法事前控制。

| 前缀 | 平均五步末距 | 零后缀末距 | selected 尾末距 | 相对零后缀变化 | 自行重规划末距 |
|---|---:|---:|---:|---:|---:|
| selected | 0.95369 | 0.95369 | 0.51053 | -0.44316 | 0.54205 |
| zero | 1.10586 | 1.10586 | 1.09398 | -0.01188 | 0.57935 |
| 同库 greedy | 0.73785 | 0.73785 | 0.67465 | -0.06319 | 0.51991 |
| random bridge | 0.76480 | 0.76480 | 0.81615 | +0.05135 | 0.51842 |
| lookahead | 0.74358 | 0.74358 | 0.82948 | +0.08590 | 0.50649 |

7/24 个 selected 前缀满足预定义非即时条件；其中根 20 同时满足相对
zero 和 greedy 的共同 selected 尾部变化及自行重规划严格优势。根 20
起始距 1.28062、selected 五步末距 1.45074；自行重规划末距 selected、
zero、greedy 分别为 0.55012、0.57984、0.62613。它并未同时优于历史
random bridge 和 lookahead 的自行重规划：后者为 0.51691、0.52826。
同时优于四类控制的交集为 **0/24**。

在该根上，selected 与 zero、greedy 的后继零名义有限差分响应矩阵的
Frobenius 差分别为 0.74044、0.74044；整个面板两项差的均值分别为
0.57911、0.21854。这是具体有限差分设置下的响应改变，不单独证明因果
机制或全尺度有效性。共同后缀及重规划给出有限的开发线索，但原 120 次
CEM 已知可能遗漏更好尾部；历史对照、暴露任务及后验 selected 尾部也
限制结论。根 20 是**事后有限见证**，不是事前确认或 PC2 自动通过，
更不是 PC3 规划优势。

共计 720 条共同后缀链、14,400 条自行重规划链和 2,400 条响应差分链；
这些是诊断成本，不计入原 478–480 token 的编译器成绩。
[原始数据](../le-wm-main/results/pc_multidirection_successor_exact_dev_20261103_codex_same_chain_diagnostic.json)
SHA-256：`ab1c5982507199fe018950766f22a17211d691dca537966fb211028c9a2cc739`。
预测、前缀来源、响应、机制判定和最终编译器决策经独立重放，以下命令
结果 **2 passed**：

```bash
.venv/bin/pytest -q -p no:cacheprovider \
 tests/test_pc_multidirection_successor_same_chain_diag_20261103_codex.py \
 tests/test_pc_multidirection_successor_exact_dev_20261103_codex.py
```

下一步须事前冻结同一版本及机制规则，用来源可认证的数据检验，且给各前缀
共同充分的优化机会；不能将此根的后验结果提升为独立复现。
