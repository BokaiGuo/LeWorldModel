# 标准 PushT 开发面板：参考重放、执行轨迹与多 seed

2026-11-05。开发诊断，不是官方实现认证、独立确认或 PC0 通过。协议：[参考重放合同](PC_STANDARD_PUSHT_REFERENCE_REPLAY_PROTOCOL_20261105_CODEX.md)；此前结果不修改。

## 参考动作

原 seed 42 面板的 50 个固定起点、每根 25 个归档动作均执行完成；50/50 在执行过程中至少一次、50/50 在第 25 步满足本地成功定义。记录包含每步动作、归档状态、重放状态、误差和终止标记：[原始报告](../le-wm-main/results/pc_standard_pusht_reference_replay_20261105_codex/report.json)。

执行后继续至第 25 步，即使此前已成功，因此这是参考重放诊断，不是原生早停策略。动作按归档原值执行，没有裁剪；row 2005033 的第 1 个动作是 `[1.188823938369751, 0.11257454007863998]`，超出声明动作空间。全部 1250 个动作中共 1 个命令越界。不可把这个结果称为 50 条严格合法参考链。

首步状态最大绝对误差的中位数约 0.758、最大值约 5.232；末步物体位置误差中位数约 0.494、最大约 2.942。reset 本身的最大状态偏差最高约 5.232；状态不包括全部物理隐变量，本次未认定偏差的具体原因。重放说明这些任务在当前设置下有较强的可达性线索，不保证数据来源独立、完整物理状态逐步一致或严格动作域内可重放。

## 原生 CEM 轨迹及种子

记录器在原生调用外部捕获动作、状态、活动掩码和终止值；seed 42 原配置 50 任务重跑与先前结果的逐任务布尔值完全一致（47/50）。共记录 50 个时刻、1370 个活动环境步，活动步动作与状态均无非有限值：[执行轨迹](../le-wm-main/results/pc_native_trace_full50_v2_seed42_20261105_codex/trace.json)、[指标](../le-wm-main/results/pc_native_trace_full50_v2_seed42_20261105_codex/eval_metrics.json)。非有限值被编码为 `null`，必须结合活动掩码解释；1130 个非活动环境步的动作含 `null`。该一致性不证明每个动作都与没有记录器的旧运行逐位相同，旧运行没有动作记录。

追加本地 PushT 原预算：

| seed | 成功 | 原始记录 |
|---|---:|---|
| 42 | 47/50 | [指标](../le-wm-main/results/pc_standard_pusht_native50_seed42_20261105_codex/eval_metrics.json) |
| 43 | 45/50 | [指标](../le-wm-main/results/pc_standard_pusht_native50_seed43_20261105_codex/eval_metrics.json) |
| 44 | 42/50 | [指标](../le-wm-main/results/pc_standard_pusht_native50_seed44_20261105_codex/eval_metrics.json) |

不同 seed 同时改变任务及规划随机流；面板可能重叠，且均来自同一数据和 checkpoint，不构成独立任务来源或多 checkpoint 证据。

## 错误、差异及未完成项

- 记录器第一次启动因 Hydra 配置路径错误而在规划前失败；第二次在受限环境中因缺少 GPU 节点而失败；首次 50 根因严格 JSON 不接受非有限占位值而在序列化时失败。均保留产物。修复后用 `null` 表示非有限值、保留活动掩码；2 个针对采样与序列化的测试通过。已检查活动步动作和状态均有限，但不能仅凭此宣称轨迹与权威实现一致。
- 原生 `eval.py` 的有效索引采样排除最后一个有效行；本次重放故意复刻此行为。未做修复后配对效应实验。
- JEPA 本地 checkpoint 和训练／验证数据可见；原仓库 PushT 模板 `configs/online_plan_evals/pt/pt_L2_cem_sourcedset_H6_nas6_ctxt2.yaml` 列 `eval_episodes=96`、300×30、10 elite、horizon 6、执行 6、goal_H 6。后续进一步找到包含模型配置的 `configs/evals/simu_env_planning/pt/jepa-wm/pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml`，需核对 checkpoint、数据和 decoder 路径及该配置与论文结果的对应关系。仓库当前有 5 个已修改文件，不能直接称为原版复现；验证集目录可见 21 个视频，也不能假设已满足原协议需要。尚未执行 JEPA 官方基线。
- 本次未完成 TwoRoom 参考重放、多 seed 或逐步轨迹，未执行 Reacher、JEPA 基线和 Planning Compiler 的等资源公平比较。未完成与作者权威代码、依赖、数据和 checkpoint 的逐项来源／等价认证；保留原压力集和评分失配结果。

本次新增实现：[参考重放](../le-wm-main/experiments/pc_standard_pusht_reference_replay_20261105_codex.py)、[原生记录包装器](../le-wm-main/experiments/pc_standard_native_trace_wrapper_20261105_codex.py)及[测试](../le-wm-main/tests/test_pc_standard_pusht_reference_replay_20261105_codex.py)。
