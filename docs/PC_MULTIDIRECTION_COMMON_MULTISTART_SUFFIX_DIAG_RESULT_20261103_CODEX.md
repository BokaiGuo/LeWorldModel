# 已封存前缀的共同多起点后缀诊断

按照[事前协议](PC_MULTIDIRECTION_COMMON_MULTISTART_SUFFIX_DIAG_PROTOCOL_20261103_CODEX.md)，
对 24 个暴露 Wall 根的五类固定前缀，分别运行同一 CEM、四个起点、
每起点 160 条完整链。每个前缀的诊断预算为 640 条链、1920 个五步
token；它是原方法预算之外的后验诊断。

| 前缀 | 1 起点平均最佳终距 | 4 起点平均最佳终距 | 4 起点成功 `<0.05` |
|---|---:|---:|---:|
| selected | 0.49446 | 0.46579 | 2/24 |
| zero | 0.53033 | 0.50652 | 0/24 |
| same-bank greedy | 0.52072 | 0.51121 | 0/24 |
| 历史 random bridge | 0.50386 | 0.50090 | 0/24 |
| 历史 lookahead | 0.50702 | 0.50103 | 0/24 |

selected 的两个成功根均不是非即时前缀，不能当作机制见证；这些结果
也不是原编译器 480-token 成绩。对于 7 个非即时 selected 根，四起点
后分别有 2、4、1、1 根严格优于 zero、greedy、random bridge、
lookahead 的各自重规划结果；同时优于四者的是根 12。根 12 的 selected
终距为 0.500237，greedy 为 0.500307，差约 `0.000070`，且两者均未
接近成功阈值。历史 random/lookahead 非事前同库控制；不能将该后验
差异宣称为独立机制或 PC3 优势。

根 20 的 selected 终距为 0.546233，zero 与 greedy 分别为 0.507440、
0.542136：此前 120 次搜索中 selected 对这两个控制的优势，在本次更大
预算配置下未保持。不同预算和随机起点可能改变排序；这支持继续检查
后缀搜索充分性，但不证明哪一种前缀不可达或理论无效。四起点也不是
搜索上界。四组中第一个至第四个起点的平均 incumbent 仍有改善，不能
将该配置认定为搜索充分。

逐根结果及每起点评分、随机种子、最优动作：
[原始结果](../le-wm-main/results/pc_multidirection_successor_exact_dev_20261103_codex_common_multistart_suffix_diag.json)
SHA-256 `fabddb303d0a24f1dafdbea17700616bbbc761036964f0b3ee417b46bf864d58`。
另外用相同种子重新生成全部 480 次搜索，逐项核对得分和最优链一致；
这是确定性重放而非独立复现。PC2、PC3 和来源独立性状态不变。
