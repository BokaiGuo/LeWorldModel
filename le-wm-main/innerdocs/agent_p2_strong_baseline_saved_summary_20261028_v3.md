# Matched baseline 参数/资产合同 v3

本版本只新增独占 source/tests/innerdoc，不改v1/v2或结果。使用 experiment-code 的事前定义与记录原则，按用户约束不执行神经/物理/网络，不生成研究曲线。它为 kinodynamic、no-composition、sampling 三类已封存资产实现可调用参数准备与saved校验，而非授权运行。固定全已有 root/arm 与真实资源对象/SHA；不按 outcome 筛选、不 padding、不改变 checkpoint/Q/normalizer。

## 比较规则

`family` 与 `algorithm` 只作为报告元数据，不是匹配前提。arm名称也不要求跨算法相同。必须逐项已知且相等的字段为 `task_identity/checkpoint_identity/action_support/observation_spec/eval_metric/horizon/resource_axes/root_ids/goal_identity/source_authority`。任一缺失/unknown 直接拒绝；已知不等则 `control_mismatch_no_direct_comparison`。不填0、不猜测、不生成delta。空值、嵌套null、unknown/unavailable/TBD、非有限值都不能算已知；真实显式计数0可保留。

每个方法自己的完整root×arm分母固定为 `fixed_rows`，与封存报告逐行相等；删根、删臂、替换臂、重排都拒绝。库存为kinodynamic2行、no-composition2行、sampling4行（2根×CEM/IID）。root_slot不能默认为另一协议root SHA；真正跨协议identity mapping需MAIN提供有证据的绑定，不能按文件名推断。

prospective horizon证据应明确planning packets、packet steps、execution steps；checkpoint/action support应包含冻结权重、原normalizer、原动作映射/合法域；observation应包含完整历史及goal身份；eval metric需含定义、单位及native/model区别。通用JSON绑定检查不替代MAIN对这些值科学含义的审读。

当前报告层抽取仍缺checkpoint/observation/goal等精确控制身份，inventory为 `unknown_control_refuse`。不是说全仓库无该资产，而是当前尚未证明对应关系，不能填默认值。全根全臂仅表示盘点范围，不表示协议匹配。

## 真实资源及可行停点

- kinodynamic两根共288 predict_API/action_encoder/predictor/pred_proj，144/root；beam20→retain6→120 terminals，加whole2/segmented2。原schedule/seed参数在hash-bound报告contract，未改动。
- no-composition两根176 predict/action_encoder、2J、384VJP、4SVD、2bounded solve，88predict/root（含nominal、AD forward、40候选和replay）。不是144/root。
- sampling搜索504predict/AE、12CEM update；继承direct120predict、12J、2304VJP、6solve/6SVD及8encoder/8projector分别保留。实际物理另40public steps、8reset、44render；历史是causal padding，不是moving observed history。

`native_resource_evidence` 分planner/sampling_search/physical/inherited/wall/predictor_samples，缺轴仍null；不相加API、samples、AD和物理。模型指标不能直接比较native物理指标。现有成本及观测不同，因此没有真实matched comparison；不能从这些报告诚实推导统一运行参数，更不能称等FLOPs或公平。

## 可调用prepare与证据格式

`template`生成固定全根臂的request，缺参数/控制/预算均null，故原样prepare必须拒绝。MAIN先事前封存共同协议JSON，再逐方法填写 `parameters/controls/resource_budget/evidence`。每个控制字段、整个参数对象、整个预算对象都需精确 `{path:absolute_JSON,sha256:actual_SHA,pointer:JSON_pointer}`。prepare实际读取、核SHA及字段相等；没有未来SHA或默认身份。

参数对象需MAIN绑定算法真实seed/candidate/beam/tie-break/selection/replay/normalizer；本模块不改40谱候选或20×6beam凑144。若runner无法消费新共同参数，具体停点是缺执行adapter，不得以通过本合同称已运行。evidence只证明字节/字段对应，不证明证据作者的科学结论。

资源预算强制14轴：predict_API、predictor_samples、action_encoder、predictor、pred_proj、encoder、projector、AD_forward、VJP、J、SVD、solve、wall_seconds、physics_steps。每轴有 `unit/scope/limit/inherited`；scope固定 `per_root_per_arm_including_signal_search_replay`，单位分别calls/samples/seconds/steps，计数非负整数，wall允许实数秒。继承成本不得缺失；controls.resource_axes必须等于完整预算。实际执行未来仍需attempt/return逐轴实账；本模块不伪造returns或padding。

prepare永远返回 `run_authorized=false/fairness_claim=false/scientific_admission=false/P1_admission=false/freshness_certified=false`。比较键匹配不代表运行授权或科学通过。verify重建全部saved证据及source/tests/doc三文件SHA，漂移拒绝。

## 文件与命令

- source: `experiments/spectral_reachability/agent_p2_strong_baseline_saved_summary_20261028_v3.py`
- tests: `tests/test_agent_p2_strong_baseline_saved_summary_20261028_v3.py`
- doc: `innerdocs/agent_p2_strong_baseline_saved_summary_20261028_v3.md`

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_strong_baseline_saved_summary_20261028_v2.py tests/test_agent_p2_strong_baseline_saved_summary_20261028_v3.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_strong_baseline_saved_summary_20261028_v3 inventory --output results/agent_p2_strong_baseline_saved_summary_20261028_v3_inventory_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_strong_baseline_saved_summary_20261028_v3 template --output results/agent_p2_strong_baseline_saved_summary_20261028_v3_request_MAIN01.json
# 以下依赖MAIN补齐并封存request，不能填unknown/null；ACTUAL_SHA取真实输出SHA。
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_strong_baseline_saved_summary_20261028_v3 prepare --request results/agent_p2_strong_baseline_saved_summary_20261028_v3_request_MAIN01.json --output results/agent_p2_strong_baseline_saved_summary_20261028_v3_prepared_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_strong_baseline_saved_summary_20261028_v3 verify --contract results/agent_p2_strong_baseline_saved_summary_20261028_v3_prepared_MAIN01.json --sha ACTUAL_SHA --request results/agent_p2_strong_baseline_saved_summary_20261028_v3_request_MAIN01.json
```

实际RAM/saved回归：v3 48项，联合v2共60项通过（0.95s）；包括算法不同可匹配、每控制未知/不等、嵌套缺失、全方法/根臂分母、真实资源分scope、API/sample分轴、证据SHA/字段漂移、RAM prepare/verify、CLI inventory/verify、独占写入与冷导入拦截。只读现有三份known资产和临时RAM fixtures，不打开reserved fresh roots，不运行神经/物理。测试不是科学证据；P1独立机制证据及freshness仍未通过，不能由此解锁。
