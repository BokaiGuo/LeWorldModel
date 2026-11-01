# Teacher library40：固定槽 Q 排序误差归因

这是独占 saved-only 分析器 `agent_p1_teacher_library40_rank_error_20261028_v1`，不修改冻结的 RGB40 v1/v2、MAIN report/consume、compiler/physical结果或共享文档，不执行模型、CUDA、physics、网络，也不生成新的神经数组。输入绑定当前实际 RGB40 report `cb08d580b3ee2e8d15d2cc83065a8dc68a74672df0e2b602479a836469cb15f3`、prepare `499f1d310b1844bb05ba78c88d3eecc4285c7041a6482fd7c68fc2fa39ce8676`、consume `9dfb1081711505cef281f1ff547b1dad46b59cb0ea207e985497ecbf9cecf0a9`，并重新验证 RGB40 strict consume、原compiler goal/prediction数组、actual latent数组及三层SHA。

## 固定40槽与合法匹配

固定顺序为四task `(root0,off),(root0,on),(root1,off),(root1,on)` × 两role `Z,S` × 五word `zero,uZ,-uZ,uS,-uS`。报告始终输出40行；index12（root0/on、Z、`-uZ`）的 actual latent/Q/误差为null。其所属固定五word组的完整五成员排序不能计算，因此该组五个 `rank_error_full5` 全unknown，不把缺失项伪装成零或“失败”。

两种匹配分栏：

- `within_role_library`：同一task/condition/role/anchor内五word的10个无序对；8组共80对，index12所在组的4对unknown，76对可用。
- `same_word_ZS`：同一task/condition、同一word、相同suffix raw bits的Z/S成对比较；4task×5word共20对，涉及index12的1对unknown，19对可用。它保留prefix/context差异，不宣称隔离因果。

跨task、跨condition、跨root、goal不完全相等或suffix不匹配的配对直接fail-closed。不会跨指标加权、择winner、改goal/Q、按结果筛根或输出动作选择。

## 算术与分类

从原保存数组以float64重算：`predicted_Q=mean((predicted[-1]-goal192)^2)`、`actual_Q=mean((actual-terminal-goal192)^2)`、`Q_error=predicted_Q-actual_Q`。在数学可定义时保留精确分解：

`Q_error = 2 mean((actual-goal) * (predicted-actual)) + mean((predicted-actual)^2)`。

同时列 `latent_MSE=mean((predicted-actual)^2)`、cross term、quadratic term及identity residual。它们是latent/Q指标，不是physical benefit或native success。

比较只使用精确符号：positive/negative/tie；零tie不计反转。每个pair分类为 `concordant`、`discordant`（排序反转）、`both_tie`、`predicted_tie_only`、`actual_tie_only` 或 `unknown_missing_terminal`。固定pair分母、available分子、unknown分母和 observed-only rate 分开报告；只要unknown存在，`full_fixed_library_reversal_rate`、完整tau-b保持null并注明`missing_terminal`。不把unknown计为失败或反转。

排序误差行只写固定组内1-based ascending midrank；不重排rows，不输出argmin、winner或选根结论。缺失组的五行rank误差均unknown，是完整五成员rank向量不可定义的精确边界。

## 接口、测试与限制

```bash
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_teacher_library40_rank_error_20261028_v1 analyze --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_teacher_library40_rank_error_20261028_v1_analysis_MAIN01.json

ANALYSIS_SHA=$(sha256sum /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_teacher_library40_rank_error_20261028_v1_analysis_MAIN01.json | cut -d ' ' -f1)
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_teacher_library40_rank_error_20261028_v1 verify --analysis /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_teacher_library40_rank_error_20261028_v1_analysis_MAIN01.json --analysis-sha "$ANALYSIS_SHA"
```

测试：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p1_teacher_library40_rank_error_20261028_v1.py
```

单测使用RAM NumPy fixture；saved `analyze --summary`读取并复核实际39 latent、40 goal/prediction和RGB40 strict receipt，不调用run。当前没有完整40 actual Q，不能报告完整库反转率、winner或P1科学结论；known roots=2、independent roots=0、P1=false。任何后续修复只允许新stem。
