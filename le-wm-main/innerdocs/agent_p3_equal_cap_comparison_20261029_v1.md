# P3 equal-cap vs old-parent：独立 saved 描述比较

仅新增本stem source/tests/doc。冻结 equal-cap driver 三件套、旧parent及全部产物不改。使用experiment-code技能组织独立保存与RAM回归；用户范围优先，不生产prepare/NN/physics/fullstrict、不导入producer/driver/torch、不训练/拟合/挑根/调参。此任务不是追加审计runner或科学gate。

## 输入ABI及认证边界

MAIN提供两套各三个文件：`--old-prepare/--old-report/--old-receipt`、`--new-prepare/--new-report/--new-receipt`，每个均有对应`-sha`实际SHA参数，不能以预期未来SHA代替。prepare/report/receipt分别须为真实`agent_p3_parent_native_cuda_20261029_v1`与`agent_p3_equal_cap_parent_20261029_v1`自有schema。新输入尚待MAIN生产，worker不读取新运行中结果。

接受完整strict consumer出具的settled或可验证partial receipt，不接受未验证/未知status。receipt SHA由MAIN外部认证；本脚本不再调用consume、verify、build或重算journal/native AD/ridge。这是六份文件的绑定和描述性数值差值，不是独立重签fullstrict。源文件最多256MiB，重复JSON字段/NaN拒绝；报告和receipt的prepare路径/SHA与report SHA精确join、四槽身份/status、receipt每个已观察arm的完整resource_receipt、depth/width/core-plan/unknown逐项join；前后核六输入SHA不变。相同两完整task（含goal/context/raw/normalizer）和模型identity，否则fail closed。

## 固定输出

两known roots × fixed/adaptive，顺序root0 fixed/adaptive、root1 fixed/adaptive。每槽完整144个请求位置，总576个**配对位置**，不是576个已执行候选、独立样本或同动作见证：

- 每个位置保存双方原status/error、H、候选宽度/索引、program/extension type、parent context/binding SHA、prefix action摘要、完整request摘要、node depth/indices/各edge H/type/cost/context/node摘要、extension raw action摘要。不复制巨大latent/J/local_response正文。
- 顺序索引只是位置匹配。`same_request`要求摘要和相应字段一致；双方未请求或缺失返回null。分叉树上的同序号不能当相同parent或同动作误差对照。
- 每arm保留全部queue decision/status/depth/parent/request_count/used_before、realized depth/width、diagnostic_best、core_model_plan与实际released_plan分栏。equal-cap最终guard失败时，即使core已有plan，也不可视作release。
- 原raw JSON、float64与float32 little-endian位摘要分开；正负零不会被Python普通数值相等吞掉。`same_extension_action_bits/same_released_action_bits`只指float32位相同，不表示整request/上下文相同，不重归一化/clip。
- 全27轴caps/attempted/completed逐轴`new−old`，失败attempt不退款；缺失整arm全144行null，资源及差值null，不补0；not_requested/合法R拒绝/失败status全部保留census。完整资源wall/inclusive scope/FLOPs/cuda/backward未知量原样保留，scalar_backward/AD-primal/J元素不由predict推算。
- 保存两个原profile、whole wall与scope；old core resource wall不是新同步arm timer。新timing只保留scope/cap/elapsed/sync计数/failure/max overrun与失败probe数，不重复完整probe。墙时无需相等、不据此推断预算效益。

共同cap不等于相同实际资源。即使27轴实际数相同也不推导equalFLOPs、效率或因果预算收益；旧/新没有新增真实物理任务评价。本输出P1/P3/C3/freshness/new_independent_roots/equal_actual/equalFLOPs/budget_effectiveness全部false。unknown不填零，不做CI，不选最优根/时间/指标。

## MAIN CLI

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。所有SHA来自MAIN实际冻结文件；不在此编造新report/receipt值。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p3_equal_cap_comparison_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p3_equal_cap_comparison_20261029_v1 analyze \
  --old-prepare results/agent_p3_parent_native_cuda_20261029_v1_prepare_MAIN01.json --old-prepare-sha ACTUAL_OLD_PREPARE_SHA \
  --old-report results/agent_p3_parent_native_cuda_20261029_v1_cuda_MAIN01/report.json --old-report-sha ACTUAL_OLD_REPORT_SHA \
  --old-receipt results/agent_p3_parent_native_cuda_20261029_v1_consume_MAIN01.json --old-receipt-sha ACTUAL_OLD_RECEIPT_SHA \
  --new-prepare results/agent_p3_equal_cap_parent_20261029_v1_prepare_MAIN01.json --new-prepare-sha ACTUAL_NEW_PREPARE_SHA \
  --new-report results/agent_p3_equal_cap_parent_20261029_v1_cuda_MAIN01/report.json --new-report-sha ACTUAL_NEW_REPORT_SHA \
  --new-receipt results/agent_p3_equal_cap_parent_20261029_v1_consume_MAIN01.json --new-receipt-sha ACTUAL_NEW_RECEIPT_SHA \
  --output results/agent_p3_equal_cap_comparison_20261029_v1_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p3_equal_cap_comparison_20261029_v1 verify \
  --summary results/agent_p3_equal_cap_comparison_20261029_v1_MAIN01.json --summary-sha ACTUAL_SUMMARY_SHA
```

analyze新路径独占xb、32MiB上限、flush/fsync；verify重读六份输入，独立重建整份summary，逐字节规范JSON比较，不调用fullstrict。相同摘要source SHA亦是冻结绑定的一部分。未执行生产消费，不把RAM合成数据作为实际证据。

## RAM测试

最终回归23 passed/0.36s（包含六文件末尾稳定性检查）；三文件git diff --check通过。覆盖576位置/27轴、整arm missing与unknown、core有plan但禁止release、signedzero/float32舍入、任务/模型/receipt/resource/status/顺序/漏槽/unknown/release篡改、R/H3缺返回、真实writer→CLI verify→整份数值篡改拒绝、禁止覆盖、无producer或torch导入。experiment-code技能仅用于独立保存与回归组织，未引入训练/图/择优。剩余MAIN独立审读与提供新实际report/完整strict receipt，再运行saved分析；无需worker生产。
