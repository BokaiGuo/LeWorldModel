# 多方向 RGB 零前缀比较的物理诊断结果

按[事前诊断协议](PC_MULTIDIRECTION_RGB_ABSTENTION_PHYSICAL_DIAG_PROTOCOL_20261103_CODEX.md)，
在四个暴露开发根的 12 个已封存决策上，将弃权比较前完整链及相同后缀的
zero-prefix 对照，从相同热身和实际动作历史重放至固定终点。

| 诊断 | 结果 |
|---|---:|
| 模型与物理重放的成对排序同向 | 12/12 |
| 反向或平局 | 0/12 |
| 事前弃权选中 zero-prefix | 0/12 |
| 实际前五步末距不小于决策起点末距 | 0/12 |

每对都满足模型与物理的 `zero - before > 0`：在这个具体零前缀比较
中，模型和物理重放均偏好原链。全部分支的决策前状态和时钟与既有原生
重放逐位一致。每次实际选中前五步均缩短到目标的欧氏距离；按“不产生
即时进展”的条件，这些前缀不能作为非即时机制见证。反馈三次决策也不
是一条首次事前选定的完整物理执行链。

该结果排除不了其他候选间的排序反转、模型轨迹误差、搜索覆盖不足或
目标投影不可靠；仅对本面板这 12 个原链/零前缀对照成立。每一对使用
同一个后缀，但这是事后执行固定对照，不是覆盖 selected、same-bank
greedy、随机桥接和等预算自行规划的完整 PC2 干预。不能推断 PC2–PC5
通过，也不能由此声称不存在实现问题或证明理论无效。

[逐对模型与物理结果及完整重放轨迹](../le-wm-main/results/pc_multidirection_rgb_fixed_terminal_dev_20261103_codex_abstention_physical_diag.json)
SHA-256 `7d87e937e9978d56051fbade1f9091a9450561480a0aac40aaa49c223a17c715`；
[诊断代码](../le-wm-main/experiments/pc_multidirection_rgb_abstention_physical_diag_20261103_codex.py)。
结果由固定决策记录生成；运行命令：

```bash
.venv/bin/python -m experiments.pc_multidirection_rgb_abstention_physical_diag_20261103_codex
```

输出已存在时命令会安全拒绝覆盖；复现应使用隔离副本和原始依赖。
