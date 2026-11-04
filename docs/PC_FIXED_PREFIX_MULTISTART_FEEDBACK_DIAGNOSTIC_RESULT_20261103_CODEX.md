# 固定前缀等预算多起点反馈结果

按[协议](PC_FIXED_PREFIX_MULTISTART_FEEDBACK_DIAGNOSTIC_PROTOCOL_20261103_CODEX.md)，
在相同 24 根、五类前缀及每阶段 480 token 下，将单个零初始化 CEM 改为
四个独立起点；每根每阶段的起点、随机种子在五类前缀间相同。执行并
封存 120 条固定前缀反馈轨迹。

| 前缀 | 四起点平均末距 | 成功 `<0.05` | 比单起点末距更低的根数 |
|---|---:|---:|---:|
| selected | 0.50249 | 0/24 | 2/24 |
| zero | 0.50581 | 0/24 | 6/24 |
| 同库 greedy | 0.50117 | 0/24 | 5/24 |
| random bridge | 0.50171 | 0/24 | 4/24 |
| lookahead | 0.50059 | 0/24 | 10/24 |

selected 逐根优于 zero、greedy 分别为 13/24、9/24；四个已定义的非即时
根中，只有根 9 同时优于这两个前缀。该比较不自动满足共同后缀机制或
事前同链要求。根 13 的 selected 末距为 0.50093，仍未找到历史反馈的
0.001087 可行尾部。

这一固定四起点配置未改善搜索覆盖；预算均分减少了每个起点的迭代次数，
因而不能据此否定多起点的所有其他配置。单起点与四起点使用不同随机流；
都属于已暴露开发任务，不是独立统计确认。原单起点及历史结果保留，
PC2、PC3、RGB 和独立复现状态均不由此提升。

## 审计与复现

计划 SHA-256 为 `1d076c118513e7ab2e83dffbcd8cb9fd28f704117ac11575f3f15eb5a6074252`，
[审计](../le-wm-main/results/pc_fixed_prefix_multistart_feedback_diagnostic_20261103_codex_audit.json)
SHA-256 为 `8e8507f17440439d4095c7fd9a37bc25475855eddd1defb65e7df8536cf52707`。
实验文件自带的 `audit` 子命令存在依赖函数替换时递归错误，发生在读取
封存计划之后、写审计之前；规划结果未修改。独立适配入口先验证原封存
依赖，再调用既有逐批物理审计，成功重放 120 条轨迹：

```bash
.venv/bin/python -m experiments.pc_fixed_prefix_multistart_feedback_independent_audit_20261103_codex
```

独立审计入口现已更新：审计文件存在时，它将重放结果写到临时目录，并与
封存审计逐项比较，不覆盖已有产物；若封存依赖或审计内容发生变化则报错。
原实验脚本的 `audit` 子命令仍有递归错误，原实验源码及其哈希未修改。
独立测试重新构造四起点及全部采样、精英更新，并检查重放不覆盖和篡改
检测；与单起点测试合计 **4 passed**：

```bash
.venv/bin/pytest -q -p no:cacheprovider \
 tests/test_pc_fixed_prefix_multistart_independent_audit_20261103_codex.py \
 tests/test_pc_fixed_prefix_multistart_feedback_diagnostic_20261103_codex.py \
 tests/test_pc_fixed_prefix_equal_feedback_optimizer_20261103_codex.py
```

剩余问题包括结构化动作搜索及在合法输入下跨后继组合的覆盖，不能通过
继续反复调整已暴露根上的这个固定配置解决。
