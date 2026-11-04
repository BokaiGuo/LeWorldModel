# E0 四结点后缀表达：特权投影结果

按[事后协议](PC_V7_E0_SUFFIX_KNOT_PROJECTION_DEV_PROTOCOL_20261104_CODEX.md)，对 18 根的首条解析见证保留其前五步，并将后十步投影到四结点分段线性动作族；该投影按动作空间最小二乘拟合，再逐结点裁剪，不按终距优化。

| 固定前缀后的后缀 | `<0.05` 成功根数 | 平均终距 |
|---|---:|---:|
| 原解析后缀 | 18/18 | 约 0.000000088 |
| 四结点动作投影 | 0/18 | 0.299870 |
| 全零后缀 | 0/18 | 1.025317 |

动作拟合不保证保留绕墙动作的切换时刻或动力学分支；因此该结果提示存在值得检验的时序压缩风险，但既不是四结点下的物理性能上界，也不证明不存在其他四结点可行解。事实上，[共同预算 CEM 诊断](PC_V7_E0_ORACLE_PREFIX_SUFFIX_DEV_RESULT_20261104_CODEX.md)中相同四结点后缀在 4/18 根至少一次成功。这里研究的是特权固定前缀与零基底诊断，并不等同于 v7 完整方法使用的所有后缀基底。

[结果](../le-wm-main/results/pc_v7_e0_suffix_knot_projection_dev_20261104_codex_result.json)包含矩阵、结点、动作、端点及误差；[审计](../le-wm-main/results/pc_v7_e0_suffix_knot_projection_dev_20261104_codex_audit.json)重新构造投影并用同一后端核对 18 根。可通过[运行器](../le-wm-main/experiments/pc_v7_e0_suffix_knot_projection_dev_20261104_codex.py)的 `run` 和 `audit` 参数复核；结果排他写入。该分析不构成自动程序发现、规划优势或泛化证据。
