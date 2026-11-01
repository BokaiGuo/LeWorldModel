# Actual-cp5 commit physical v2：真实 flags ABI 修复

仅新增本 stem 的 source/tests/本文；v1 三件套和旧结果保持冻结。MAIN 实际 v1 prepare 在 `material→assemble` 遇到 `KeyError: C3`，未生成 prepare、未构造 env、未执行 physics。它是 **consumer/producer schema 接线错误**，不是 R 模型负结果；实际 R 四槽仍各自 no_witness，不能用本修复改变该事实或派发物理。

## 最小差异

v2 保留原 reset/warm10/prefix5、control6 intent 前 durable causal guard、FIRST5 commit、H3 lookahead 不执行、资源/原生 label/PNG/hashchain/partial 消费逻辑与预算。没有阈值、normalizer、Q、选择或物理动作改动。

|位置|v1 错误/遗漏|v2 与真实 producer ABI|
|---|---|---|
|prepare/rules、consume、arms[i]、delivery|fixture 造 `C3`；实际 receipt/delivery 无此字段；arms 其他 flags 未逐项检查|要求真实13个 flags 全部存在且 `is False`；明确拒绝假的 `C3` 别名|
|C3 声明|读取不存在的 `C3`|读取并严格检查 `full_C3_demonstrated`，不使用 `get` 默认值、不补字段|
|own v2 输出|v1 own schema 的 `C3` 命名|v2 own FLAGS 同样明确命名 `full_C3_demonstrated=false`，不伪装 v1 输出|
|cp5 元数据|已有 task/context 精确连接|另核真实 rules/receipt 的 `root_context_role` 与 `cp5_tasks` 描述符一致|
|fixture|手写缩减 receipt/delivery 导致假 ABI|以实际保存 prepare/receipt 的完整键集合制模；非空 delivery 用冻结 producer 的**纯字典构造器**，不调用其 group/consume/verify/NN|

真实13个 flags 为 `P1_admission, P3_admission, equal_FLOPs, equal_cap_tree, execution_authorized, freshness, full_C3_demonstrated, full_policy_executed, full_tree_executed, original_R_equivalence, physical_release, policy_tree_executed, scientific_admission`。与当前 producer `FLAGS` 完整集合回归比对；任意键缺失、true、0、null 或字符串 false 均拒绝。额外提供 `C3=false` 也不替代真实字段。

其他实际 schema 接缝核对：固定 `arms[i].slot/root/horizon/status/complete/model_delivery_verified/no_witness/delivery/resources`；prepare 的 tasks/provenance/sources/rules；consume 外部 report/prepare SHA、prepared 路径、protocol、role、cp5_tasks；bridge 的 roots/anchor/history/actual_cp5_task；原 physical prepare 的 source_roots/source_slots/warm_traces。真实 no-witness 四槽的 saved-only assemble 已完整通过，资源 ledger 未填零、未重建/删减。未发现第二个已访问字段缺失。

实际本批所有 delivery=null，所以**没有实际非空 delivery 的兼容性实证**。该分支仅通过 producer 真实构造器的 RAM 回归，完整 raw/path/verification(call_id/segments/status/path)/flags 结构在 fixture 中保留，不能假称本批验证了一个可执行 R witness。

## 绑定的已保存输入

- R report `results/agent_p3_actual_cp5_r_delivery_20261029_v1_cuda_MAIN01/report.json`：`deb69d78579b4cdc0ffb3ae4c671dd71f9ecd9b9e02729484085cb3d0c0dd19d`。
- R consume `results/agent_p3_actual_cp5_r_delivery_20261029_v1_consume_MAIN01.json`：`8d55a55e108aca1695a550861d125a0b0baac2b225cba914d0af4b47d30159a5`。
- R prepare `results/agent_p3_actual_cp5_r_delivery_20261029_v1_prepare_MAIN01.json`：`b4b2f9d3ca1867a86edbb84c753cef3e52d757582299b6f851c3e8c9244a042d`。
- R frozen source：`09af91981549547c5f770ad3d0ffa42ad92b48e37daefd612191d18ec9755a26`。
- 原 bridge `results/agent_p3_physical_commit_authority_bridge_20261029_v1_MAIN01.json`：`9c47743431166ddb2564bf5dea5d5335e794d98ec0873a8b90417e632eefc97d`。

RAM/saved ABI 测试只读取这些已保存文件及原 H3 physical prepare（SHA `0e31243a683a7383066ea3b495840610a4b1f2df811fcae318764d3b5acbac4e`），调用纯 `assemble`；不调用 material/prepare/producer.consume/verify/fullstrict/source_material，不写生产产物。actual native metadata 从已保存 physical prepare 提供给显式 fixture seam，不能算真实 reset/restore 验证。producer 构造器只处理 RAM 字典，不产生预测。

## MAIN 完整 CLI

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。仅 MAIN 审读后实际 prepare/verify；**本批四槽 no_witness，不运行物理**。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p3_actual_cp5_commit_physical_20261029_v2.py

.venv/bin/python -m experiments.spectral_reachability.agent_p3_actual_cp5_commit_physical_20261029_v2 prepare \
  --delivery-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_actual_cp5_r_delivery_20261029_v1_cuda_MAIN01 \
  --delivery-report-sha deb69d78579b4cdc0ffb3ae4c671dd71f9ecd9b9e02729484085cb3d0c0dd19d \
  --delivery-prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_actual_cp5_r_delivery_20261029_v1_prepare_MAIN01.json \
  --delivery-prepare-sha b4b2f9d3ca1867a86edbb84c753cef3e52d757582299b6f851c3e8c9244a042d \
  --delivery-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_actual_cp5_r_delivery_20261029_v1_consume_MAIN01.json \
  --delivery-receipt-sha 8d55a55e108aca1695a550861d125a0b0baac2b225cba914d0af4b47d30159a5 \
  --delivery-source-sha 09af91981549547c5f770ad3d0ffa42ad92b48e37daefd612191d18ec9755a26 \
  --bridge /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_physical_commit_authority_bridge_20261029_v1_MAIN01.json \
  --bridge-sha 9c47743431166ddb2564bf5dea5d5335e794d98ec0873a8b90417e632eefc97d \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_actual_cp5_commit_physical_20261029_v2_prepare_MAIN01.json

.venv/bin/python -m experiments.spectral_reachability.agent_p3_actual_cp5_commit_physical_20261029_v2 verify \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_actual_cp5_commit_physical_20261029_v2_prepare_MAIN01.json \
  --prepare-sha "$P3_COMMIT_V2_PREPARE_SHA"
```

保留 run/consume CLI，以下仅说明接口，**不是本批执行指令**。未来必须有真实 witness、MAIN 另行审读/授权，参数全部为实际产物；无 witness 代码也不构造 env/不派发任何动作。

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p3_actual_cp5_commit_physical_20261029_v2 run \
  --prepared "$P3_COMMIT_V2_PREPARED" --prepare-sha "$P3_COMMIT_V2_PREPARE_SHA" \
  --output "$P3_COMMIT_V2_NEW_OUTPUT" --authorization MAIN_P3_ACTUAL_CP5_R_COMMIT5_PHYSICS_ONLY
.venv/bin/python -m experiments.spectral_reachability.agent_p3_actual_cp5_commit_physical_20261029_v2 consume \
  --output "$P3_COMMIT_V2_NEW_OUTPUT" --report-sha "$P3_COMMIT_V2_REPORT_SHA" \
  --receipt "$P3_COMMIT_V2_NEW_RECEIPT"
```

## 验证状态

定向新增回归73 passed/9.85s（44 deselected），其中真实 saved ABI 先复现冻结 v1 `KeyError: C3`，然后 v2 纯 assemble 保留四个 no-witness/null。最终完整回归 **117 passed/72.82s**，包括 RAM 生命周期/真实 Store+PNG 临时文件回归。没有生产 prepare、NN、physics 或 fullstrict；不签 physical release、P1/P3/C3/freshness。技能 experiment-code 仅用于窄修复、保留失败记录和 RAM 回归，不训练/调参。

冻结 source SHA `a17ee3512ebb3289fdc862c9651641fa0e61a1670689518786701f3e64021776`；tests SHA `67ee7fc3bfff635ef882ac1b025bb25aa383daaac8054bbf7d1a1be74a6d91f0`。本文 SHA 由交付消息提供。复核 v1 source/tests/doc 分别仍为 `32e485063b98e5da27599e91115e92e3238ec39e761e9860a92de15390cd815d` / `b3503782c83471aff5be19b3fb9eebbd6933f403e454fdb9e00683af1a9048b4` / `6edf798f64f3a8fdc35aa01bcd30401e732080918811e4dfd9e8208a42be7489`。
