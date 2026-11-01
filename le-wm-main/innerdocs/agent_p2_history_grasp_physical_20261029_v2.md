# GRASP history-joint30 原生物理接入 v2（仅 ABI delta）

保留 physical_v1 三件套冻结。本版 source 仅改 NAME 与 fix_policy 精确字符串，执行、容差、资源、动作与原运行均未改变。MAIN 已给最终 consumer source SHA `20863c44843ef76434067f3241c5b310e044160a6d21132ee8d722dae63bf34f`；仍须实际 receipt 路径/SHA，本文不虚填。

独占 source/test/本说明三件套，不修改冻结 planner、consumer、旧物理源和结果。worker 仅实现与 RAM 测试；MAIN 完整审读后才能运行真实 physics。不是官方 GRASP 复现、等预算比较或 P1/P2 科学准入，也不认证 fresh roots。无模型加载、NN、重规划、动作选择或 outcome 调参。

## 实际 authority / ABI

原 planner `agent_p2_history_joint30_grasp_cuda_20261029_v1` source SHA 为 `6c44630f75259ce107b5b001a8dfc7f50b60c4f7b6ab920371e2994cde1ca9a1`。MAIN 已提供实际 report：
`results/agent_p2_history_joint30_grasp_cuda_20261029_v1_cuda_MAIN01/report.json`，SHA `d92e24d9807a20f296811d4321b64cc87f2c8c73c04677a350f136debc1ce14a`。

旧 v1 strict 在 step0 Adam snapshot 的负零初始化处失败；不能用它签发物理输入。本接入显式使用 **新的** `agent_p2_history_joint30_grasp_saved_consumer_20261029_v2`：外部传 source 绝对路径、实际冻结 SHA、schema、完整 receipt 绝对路径及实际 SHA；先验 source seal，再调用真实 `consume(output, reportSHA)`，逐字典重建并比较整个实际 receipt。原 report/run 保留，未改位、未改旧 consumer。要求 fix_policy 精确为 `exact empty parameter-times-zero; installed beta complements and RN-FMA lerp; unchanged bounds`，绑定其 `consumer_bindings`。拒绝旧 `.consume` / `.model_plan`，不是把 v2 计划伪装成原 v1 或 wallcapped schema。

v2 `.consume` 的两条原 `.model_plan` 原封嵌入 own `.execution_plan.planner_plan`。逐条认证 producer compact+LF selection digest、source/model/task/blind pointer、original_context、goal/Q/normalizer、raw float32 SHA、完整末态 context 与 global30 logit chart。自己的执行计划 hash 仍用原物理 spaced JSON codec，绝不混用。输入没有 wallcapped `edges/contexts/warm_raw10` 字段假设；warm_raw10 来自真实 GRASP prepare slot，terminal 来自实际 v2 plan。v2 receipt 的资源原样保存，并非本 driver 重新声称 planner 算力公平。

`material/verify` 重建原 planner prepare，重建 native source authority，核 source blind exports、history3/goal RGB pixel SHA、原 warm raw10。source/model/task/renderer/consumer/三件套全部入 bindings，执行前后及消费末尾复核。MAIN 已报告 v2 full consume 成功，actual receipt `/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_history_joint30_grasp_saved_consumer_20261029_v2_consume_MAIN01.json`，SHA `55c192a908835e6f8c05160c92c4d5a2d46a0ed81ce4985dbf1de3c3c2f4fe2e`。worker 未重复生产 consume/prepare/物理；实际 prepare 仍须完整重建该 receipt。

## 真实执行与资源

固定顺序 root0/off、root1/off，两条 arm 均为 `history_joint30_grasp_variant`。每根 fresh env，从原 seed/placement 重置、原实际 warm10（非零），核三帧实际 RGB、root native7 hash、task goal/pose/renderer，然后执行选定 raw15 的 float32 原字节；无 clip、padding、二次 /100、tanh 重算替代或 imagined-context restore。logits 只用于事前认证，不作为执行动作源。

自有 full15 loop / Store / Meter / chain / strict saved consumer。复用已审 `agent_mixedmode_physical_20261028_v1` 原生 placement/hooks/dispatch，以及 allmode 逐帧 off/on/off capture/参考检查；不 monkeypatch 冻结 globals，不把新 report 投给旧 schema 验证器。每 public returned action 验实际 bits/dtype、全 native7 goal/state/reward/termination/original success；物理 state 仅作真实性/评价，不进 planner。

cp0/5/10/15 含实际 PNG/state/snapshot。原生早停保留固定两条分母及 missing_checkpoints，不补步、不替换根。warmup 早停和 root 已 terminal 同样留存。prefix51 是实际 warmup0..10 加本动作 plan0..5 三覆盖绘图，不宣称这五步与其他 selector 相同。原 source 未封存的中间 on 帧只记 unknown，不伪造参考。失败保留 operation intent/failure、错误、行状态和可写的 partial artifacts；违反动作/renderer/PNG 不变量的失败不能被 strict 当作有效数据放行。

无早停上限：2 constructor / 2 close / 4 reset / 50 public / 508 physics / 8 set_state / 6 goal_setter / 4 goal_render / 2 goal_bind / **162 PNG/explicit_render**。嵌套资源不相加为“总调用”。逐 callback intent/returned/failed 实计，失败不退款；wall 900s、输出64MiB（最终8MiB reserve）、磁盘 floor200MiB、metadata128MiB。cleanup close 可在超时/故障后尝试，但不能新 dispatch physics。返回 actual counts，不将 caps 冒充实际工作量。

## MAIN CLI

在 `/home/gbk/code/LeWorldModel/le-wm-main`，下列输入已填 MAIN 提供的实际冻结值；后续物理 prepare/report SHA 仍须取真实输出，不偷填。

```bash
MOD=experiments.spectral_reachability.agent_p2_history_grasp_physical_20261029_v2
.venv/bin/python -m "$MOD" prepare \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_history_grasp_physical_20261029_v2_prepare_MAIN01.json \
  --planner-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_history_joint30_grasp_cuda_20261029_v1_cuda_MAIN01 \
  --planner-report-sha d92e24d9807a20f296811d4321b64cc87f2c8c73c04677a350f136debc1ce14a \
  --planner-source-sha 6c44630f75259ce107b5b001a8dfc7f50b60c4f7b6ab920371e2994cde1ca9a1 \
  --planner-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_history_joint30_grasp_saved_consumer_20261029_v2_consume_MAIN01.json \
  --planner-receipt-sha 55c192a908835e6f8c05160c92c4d5a2d46a0ed81ce4985dbf1de3c3c2f4fe2e \
  --consumer-source /home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/agent_p2_history_joint30_grasp_saved_consumer_20261029_v2.py \
  --consumer-source-sha 20863c44843ef76434067f3241c5b310e044160a6d21132ee8d722dae63bf34f \
  --consumer-schema agent_p2_history_joint30_grasp_saved_consumer_20261029_v2.consume

.venv/bin/python -m "$MOD" verify \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_history_grasp_physical_20261029_v2_prepare_MAIN01.json \
  --prepare-sha "${GRASP_PHYSICAL_PREPARE_SHA:?actual prepare SHA from preceding command}"

.venv/bin/python -m "$MOD" run \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_history_grasp_physical_20261029_v2_prepare_MAIN01.json \
  --prepare-sha "${GRASP_PHYSICAL_PREPARE_SHA:?actual prepare SHA required}" \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_history_grasp_physical_20261029_v2_MAIN01 \
  --authorization MAIN_GRASP_HISTORY2_FULL15_PHYSICS_ONLY

.venv/bin/python -m "$MOD" consume \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_history_grasp_physical_20261029_v2_MAIN01 \
  --report-sha "${GRASP_PHYSICAL_REPORT_SHA:?actual physical report SHA required}" \
  --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_history_grasp_physical_20261029_v2_consume_MAIN01.json
```

全部路径必须规范绝对路径，新输出专属 stem 且不存在。消费返回 own `.consume`：`rows[2]`、`plans[2]`、`counts/errors`、`saved_PNG_verified`、真实 `planner_authority/bindings`、`metric_breakdown`。每 row 的 checkpoints（含 cp15 缺项）是后续 RGB companion 的接缝；不提供旧 wallcapped 伪 schema/receipt。

## 测试记录（不是物理证据）

`.venv/bin/python -m pytest -q tests/test_agent_p2_history_grasp_physical_20261029_v2.py`

v1 历史记录为52 passed /12.24s。本版只迁移同一套52项 RAM 回归，最终运行 **52 passed /12.29s**。RAM complete lifecycle 完整检查两 env 的 raw25、所有资源、162 PNG 与 saved consume；覆盖 warmup/plan native stop、throw/guard/substitution/render 失败、重新 hashchain 后动作/目标/renderer/reward/late return 篡改、新旧 schema/digest codec、consumer source/receipt/body/self binding/fix-policy 拒绝，以及真实 Store→disk→两 event journal reader 的 LF/hashchain 回归。没有调用真实环境、CUDA、模型或生产 prepare；未读物理结果挑根。
