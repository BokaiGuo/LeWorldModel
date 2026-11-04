# 多方向后继多起点：配对预算开发结果

按[事前协议](PC_MULTIDIRECTION_MULTISTART_BUDGET_DEV_PROTOCOL_20261103_CODEX.md)，
在 24 个已暴露 Wall 任务上，对新多起点编译器和四起点 full-CEM 分别
运行 480、1920 五步预测 token 上限；数值是首次 15 步规划的精确动力学
终距，越小越好。

| token 上限 | 多方向平均终距 | full-CEM 平均终距 | 多方向逐根更低 | 多方向 / CEM 成功 `<0.05` |
|---:|---:|---:|---:|---:|
| 480 | 0.52053 | 0.55249 | 18/24 | 0/24 · 0/24 |
| 1920 | 0.47612 | 0.50588 | 18/24 | 1/24 · 0/24 |

1920 点唯一成功为根 6，其前五步已立即降低终距，不是非即时机制
见证。两个预算点复用同 24 根，不是独立样本；CEM 只是一个四起点
配置，未证明强基线优势或谱增量。多方向实际消耗 478–480、
1918–1920 token，对照均为上限；多方向每根 3 次 SVD，对照不做
SVD。单次平均规划墙钟时间在本机分别约 0.036/0.064 秒，而对照约
0.018/0.028 秒；仅为当次串行测量，不能据此声称总成本匹配。
480 点每个前缀按可用份额使用 1 或 2 个起点，1920 点均为 4 个。

随后按[机制诊断协议](PC_MULTIDIRECTION_MULTISTART_BUDGET_MECHANISM_DIAG_PROTOCOL_20261103_CODEX.md)
对全部 48 个最终选中链进行相同后缀、各前缀 120 链重规划及后继
响应干预：

| token 上限 | 非即时 selected 根数 | 同时满足 zero/greedy 共同尾部与重规划比较 | 四控制交集 |
|---:|---:|---:|---:|
| 480 | 3/24 | 0/24 | 0/24 |
| 1920 | 6/24 | 0/24 | 0/24 |

机制干预是结果封存后的开发诊断，其历史 random/lookahead 对照和
selected 尾部均有事后局限；120 条后缀搜索不是充分搜索上界。它
没有给出符合该有限比较定义的见证，也不证明方法或理论不可能有效。
由于协议规定无必要机制即停止该单配置，不在此暴露面板继续追加预算
或调参。新版本尚未接入真实 RGB，不能与旧版 RGB 成绩拼成同一方法。
PC2、PC3 及后续 gate 不由此通过。

96 次规划的每批预测、浮点动作、token、完整链及最终全局最优评分
已逐批重算；48 组机制干预以相同输入重新执行并逐项一致。这是
确定性回放，不是独立实验。

- [原始规划](../le-wm-main/results/pc_multidirection_multistart_budget_dev_20261103_codex_plans.json)，SHA-256 `67723fa118f556877a10e9e9402e7092fe1edbe05a6663efe05c90e6fc44370e`
- [预测审计](../le-wm-main/results/pc_multidirection_multistart_budget_dev_20261103_codex_audit.json)，SHA-256 `de1e376c9fdf06af9acdd9280465709b378ab8e6fa37713a93fa41430ed5848b`
- [机制干预](../le-wm-main/results/pc_multidirection_multistart_budget_dev_20261103_codex_mechanism_diag.json)，SHA-256 `508f2c45edd145e492470d7980b3777230f1f67620917ab16efb3fd22379c78a`
