# v7 E0 特权前缀后缀优化：事后开发结果

按照[诊断协议](PC_V7_E0_ORACLE_PREFIX_SUFFIX_DEV_PROTOCOL_20261104_CODEX.md)，在已暴露 18 根上取首条解析见证的五步前缀，不向优化器提供解析后缀。每根使用三条固定随机流，各流评分一个零尾初始链及 480 个 CEM 样本；zero 和 same-bank greedy 使用已封存的相同后缀优化协议结果。

| 前缀 | 平均终距 | 成功流数（共 54） | 至少一流成功的根（共 18） |
|---|---:|---:|---:|
| 特权可行见证前缀 | 0.438288 | 7 | 4 |
| zero | 0.500900 | 0 | 0 |
| same-bank greedy | 0.500841 | 0 | 0 |

特权前缀按每根三流平均终距比较，对 zero 与 greedy 均为 **9 胜、0 平、9 负**。解析完整见证在所有 18 根上重新验证达到 `<0.05`；该完整链仅是可行性对照，未用于 CEM 初始化或选择。

这说明给定已知可行前缀后，当前有限预算、四结点参数化的 CEM 在部分根上可以重新找到成功后缀，但在另 14 根三次均未成功。结合前述前缀注入经常被裁剪的诊断，候选覆盖／保留和后缀搜索均可能限制结果；实验不能将 14 根未成功解释为后缀不存在，也不能唯一归因于预算、参数化或实现问题。各随机流共享根，不是独立任务。

[原始结果](../le-wm-main/results/pc_v7_e0_oracle_prefix_suffix_dev_20261104_codex_result.json)保存每流全部动作、分数与 CEM 代信息；[封存清单](../le-wm-main/results/pc_v7_e0_oracle_prefix_suffix_dev_20261104_codex_manifest.json)绑定历史见证、干预和源码。[审计](../le-wm-main/results/pc_v7_e0_oracle_prefix_suffix_dev_20261104_codex_audit.json)以同一精确动力学后端重算 25,974 条动作的评分、选择和摘要；它不是独立模型复现。运行器和审计入口为[脚本](../le-wm-main/experiments/pc_v7_e0_oracle_prefix_suffix_dev_20261104_codex.py)的 `run` 与 `audit` 参数。

这是事后特权分析，不是 v7 自主发现程序，也不构成 PC2、PC3 或泛化通过。结果不支持在这些已暴露根上继续事后挑前缀或调参来制造正式正结果。
