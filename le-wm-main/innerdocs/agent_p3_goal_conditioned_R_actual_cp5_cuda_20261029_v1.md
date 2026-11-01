# actual-cp5 goal-conditioned R HOST integration v1

独占 source/tests/doc；不改任何冻结库/结果。事前协议：
`innerdocs/main_p3_goal_conditioned_R_actual_cp5_protocol_20261029_v1.md`
SHA `24012da9b83f80491bb8fd6c9ac158d01d6752eb3f4254fd32902da44cbd84f0`。
这是探索性 known2 单parent模型诊断/首包delivery，不是原spectral8等价、盲确认、物理执行或full-C3。
worker仅RAM测试，不运行生产prepare/NN/physics/fullstrict。MAIN审读三件套并实际prepare/verify后才可授权CUDA。

## 固定来源与核接入

固定 actual-cp5 builder SHA `ce6670d84f6f0f712bdfe5e1b7471be65b0441338c18010f79b3967ef28397c7`。
顺序 root0H1/root0H3/root1H1/root1H3；完整双任务身份、raw10/prefix5、三latent/两token、原goal/Q/normalizer来源沿冻结 `agent_p3_actual_cp5_r_delivery_20261029_v1.build/input_pair` 重建；不读outcome。
Nash核source `367cdba6fd97888504b5005b1bd72512c111e522e783aaaca164671e0625b6c8`，tests/doc三SHA均硬pin。
核的 `build_bank` 只有独立内部非durable ledger，无原27轴回调注入口。此接缝不猴补：直接调用冻结 `validate_inputs/direction_data/slots/contract`，将其 literal generate 方程放入原 `core.Ledger`，分别真实派发1 SVD、1 solve、1 program_compile。
RAM逐位核对完整 intermediates/directions/candidates 与原冻结 build_bank，非另一算法/允许偏差。

原 `ParentNativeAdapter` 的root/local映射保持完整；bank输入严格使用 `mapping.parent_task/native_response`。
使用原 `_mapping/_evidence` 核对同次付费raw-zero signal：原生origin/nominal/J数组、索引、完整调用、嵌套费用和local binding。
HOST另外读取本次新持久capsule，核对完整JSON/字节SHA；saved consumer再次认证全部文件和capsule。
保持核contract的 `source_artifact_authenticated_here=false`、核flags `C3=false`；不改它们。
生产own-schema单独给出 `source_capsule_bytes_verified_here`（仅该真实调用字节认证），不能据此签模型/R科学certificate。
生产层沿真实ABI使用 `full_C3_demonstrated=false`，不补 `C3` 别名。

## 分母、模型资格与动作

四槽固定，每槽zero+八候选。八个方向槽均保留，退化slot为 `blocked_degenerate_direction`、actions/eligible=null、保存reason，不替换/去重/补零、不派发其NN。
每组真实另付zero edge→H1 postcontext J→原bounded_ridge→suffix诊断；有效候选逐一同样执行。
原lambda=.002、原两半径、全经济SVD/目标投影+切向/64eps/方向顺序/tanh64→cast32均不变。
资格仍严格保存值 `immediate>zero.immediate && reach<zero.reach`，无epsilon。
全部eligible按(immediate,index)排序，选择首项。无eligible保留no_witness，不fallback。
选中edge另付whole和segmented；H1 [5]/[5]，H3 [15]/[5,10]，完整路径/动作bits必须精确一致。
复用冻结 `NativeDelivery` 的已审原生实现，不运行预测回退；diagnostic suffix不进计划。
delivery为新 `.delivery` schema，复用旧field builder后显式新producer/protocol/bank SHA；从不将它作为旧producer receipt传入。
字段 task/base_task、provenance/sourceclosure、raw_edge/commit_raw5/lookahead_raw及三套f32字节SHA、whole/segmented验证均保留。
仅首5可作为未来模型commit候选；physical_release/execution_authorized/full_C3/P1/P3/freshness/equalFLOPs全部false。

## 资源与失败

旧actual-cp5 caps(h)仅solve_calls+1。最大H1/H3 predict40/66、J10、replay20；27轴均保存，未知scalar backward/FLOPs为null。
bank独立计1compile+1SVD+1solve；每实际item计原edge/post-J/suffix/两metric/solve/SVD。parent原H signal另付；有witness再付2replay/2h predict。
派发bank前原子预检完整zero+8+bank+双验证reserve；退化只减少后续真实派发，不改cap、不填费、不refund。
完整实际费用 = parent signal + bank + (有效候选数+1)×原item + (有delivery则双验证)。native calls=1+3×(有效数+1)+(有delivery则2)。
2GiB总output含512MiB收尾reserve、每组512MiB输出、disk floor200MiB；全作业3600s、每组1800s；CUDA allocated3GiB/reserved4GiB。
原MATH/no matmulTF32/no cudnnTF32/no CPU/CUDAautocast、冻结模型和installedclass/source闭包前后核查不变。
durable intent/return/failed、完整nativeIO/AD/arrays、hashchain与capsule保存。任何运行异常保留付费prefix、全部四槽，停止后续派发不重试。partial可保存但不能成为完整strict receipt。

## Saved consumer

own report `.report` / `four_slots_settled_not_admitted`；receipt `.consume` / `actual_cp5_goal_R_model_deliveries_verified_NOT_physical`。
consumer完整重建prepare和source；全文件census/SHA、binding、journal hashchain、全部nativeIO/原native数值校验和27轴事件census；仅从saved callbacks重建完整bank、zero、八槽null、资格/排序、动作/whole-segmented路径、费用、分组失败停止语义。
不重跑NN/encoder/physics。consumer中的SVD/solve为saved数值验证，不假装原生产零费用或新神经结果。RAMfixture必须显式opt-in，生产拒绝RAM backend。

## MAIN CLI

在 `/home/gbk/code/LeWorldModel/le-wm-main` 执行；所有SHA取实际prepare/run输出，不填未来值：

```bash
.venv/bin/python -m pytest -q tests/test_agent_p3_goal_conditioned_R_actual_cp5_cuda_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p3_goal_conditioned_R_actual_cp5_cuda_20261029_v1 prepare --output results/agent_p3_goal_conditioned_R_actual_cp5_cuda_20261029_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p3_goal_conditioned_R_actual_cp5_cuda_20261029_v1 verify --prepared results/agent_p3_goal_conditioned_R_actual_cp5_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p3_goal_conditioned_R_actual_cp5_cuda_20261029_v1 run --prepared results/agent_p3_goal_conditioned_R_actual_cp5_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_p3_goal_conditioned_R_actual_cp5_cuda_20261029_v1_cuda_MAIN01 --authorization MAIN_APPROVED_HOST_CUDA
.venv/bin/python -m experiments.spectral_reachability.agent_p3_goal_conditioned_R_actual_cp5_cuda_20261029_v1 consume --prepared results/agent_p3_goal_conditioned_R_actual_cp5_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_p3_goal_conditioned_R_actual_cp5_cuda_20261029_v1_cuda_MAIN01 --report-sha ACTUAL_REPORT_SHA --receipt results/agent_p3_goal_conditioned_R_actual_cp5_cuda_20261029_v1_consume_MAIN01.json
```

未闭合：MAIN实际prepare/verify、HOSTCUDA、完整consume及任何后续科学评估。实现/RAM通过不称production已验证。
