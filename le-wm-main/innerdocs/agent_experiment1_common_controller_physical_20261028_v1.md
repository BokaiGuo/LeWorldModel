# E1 common-controller physical v1

本stem三新文件独占实现。原common-controller三文件冻结不改；使用experiment-code的固定输入、
回执/错误保留及RAM验证流程，不训练、不运行模型/物理/网络、不派后代。
旧blocked个人goal不覆盖/虚标完成。MAIN审读及实际调度；本交付不是完整E1/P1。

## 接线与未来实际报告

固定`agent_experiment1_common_controller_20261028_v1`最终source/tests/doc三SHA为
`c34a303c5e2df56a2002f195118cc2272a7b4d10afb79f8ed63d730af1a732a9`、
`2a160139a19cc8c802e5eac8aeea48093f16f60d8948ef129ac1edbcb3589544`、
`c58f9a491f1674da5f8cd82e76dc83bf22a2da42a3ed26e1dc61252dbac072bc`。
已收到MAIN真实report/prepare/strict字节SHA，固定为
`bc3cd74c85f878cdfc7b170d0b5b555827e0dc4a59cdaea6fb5781e71f03767f`、
`8c472485d44c40ce3d95f8043d2c58f9ec2694fef74786a840d1699d5a1a86bd`、
`b4b72dea9d782c6d2e1b93f50523a1089e7f3d894e2e2c1171fc13f9b8cd425d`。
authority仍必填这些外部SHA与实际路径，并逐字核MAIN strict与重算consume相等；不支持未来占位SHA。
先调用producer纯saved strict
consume/verify，绑定其全部input/model/source/array/journal/artifact。要求576API/samples/各模块
完整、固定四plans和六rows含learned两unknown。不能用producer rows直接猜commands字段。

selection JSON精确字段（路径必须canonical绝对路径，无symlink components）：

```text
schema = agent_experiment1_common_controller_physical_20261028_v1.selection
sourcepack / sourcepack_sha256 = 原moving sourcepack路径/实际SHA（必须原SOURCE_SHA）
source_prepared / source_prepare_sha256 = 原moving source prepare路径/实际SHA
roots = [{root_slot:0,root_sha256:原root0},{root_slot:1,root_sha256:原root1}]
arms = [arbitrary_latent_waypoint,dynamics_compiled_witness]
authority = {authority:agent_experiment1_common_controller_20261028_v1,
             output:MAIN实际CUDA output绝对目录,report_sha256:实际SHA,prepare_sha256:实际SHA,
             strict_receipt:MAIN实际strict JSON绝对路径,strict_sha256:实际SHA}
selection_fixed_before_physics = true
physical_outcome_selection = false
wall_seconds = 600
```

prepare从strict plans完整投影raw10、target、两predicted_contexts、prefix/terminal/tracking cost、
原task_id/blind/task_contract/model/history/normalizer及原selection_sha；selection_digest用原
source.digest序列化，不混compact JSON摘要。不能重选目标、重解controller或改prefix。
以原moving encoder load_task严格核三帧history、goalRGB、原非零raw10 warmup、identity192 Q、
原logit球/tanh→raw。实际commoncontroller target只是latent，不当native goal、goal_pose或setter。

## 四工程分支与六科学槽

root-major×arbitrary/compiled共四工程rows，另`scientific_rows`为root-major×
arbitrary/learned/compiled六槽；learned保持unknown/null资源及commands，不dispatch，不记失败率。
失败/未派发/合法native stop仍留四工程分母；全10步不是完成recorded的唯一合法路径，
native终止或truncated可合法earlystop，但不能补checkpoint或降低科学分母。

每工程branch复用原moving准确reset recipe：原seed reset→sample_state placement→seed0
带原requested state reset→10真实非零warmup。严核原0/5/10 RGB/state/snapshot与raw逐位
相等，再以evaluator权限attach原goal7/goalRGB，无physics/setter/render/goal_pose改动。
环境with_target=False，controller不可读native goal来重选动作；仅执行该arm own raw10。
复用sealed kernel.branch/native_hooks/capture这些小内核，自有execute/authority/schema/rows，
不调用旧physical execute或hardwired witness/source.episode，不monkeypatch旧module globals。

每warmup0..10、plan0..10均显式render并PNG保存，共完整88PNG；checkpoint为plan0/5/10
真实收据，不补帧。每plan步新增tracking_observation，保存native7全状态与原生指标、实际
PNG引用/像素SHA、target SHA以及原模型成本；5/10附各原predicted context，其余步不插值。
没有新encoder，因此`observed_latent192`和`latent_tracking_error`始终null；后续MAIN RGB
消费才可量化192维实现误差，native7与latent192不能混用。该tracking事件先保存观测，
随后原guard_receipt可能拒绝；不能以“有tracking事件”视为guard通过。

guard失败先保存实际RGB/state/snapshot，再记录guard_receipt后停止当前branch。
step内部抛异常不保证能取到事后frame；row.last_observation只是最后可用帧，不能冒充故障现场。
原native angle可能在abs delta>2pi时变负，本版honor native终止，不改环境；同时保存wrapped
诊断区别。native_distance是全7维norm含velocity；success按原agent+object XY及原angle，
不能以object-only或模型Q宣称任务成功。

## 实际操作预算、失败与权限

完整四branch上界：construct/close各4、reset8、set_state16、goal_render8、goal_bind4、
public_step80、physics_step816、explicit_render88（88PNG）。physics计原space.step实际hook，
reset内部16及public每次10substeps；与public/goalrender是重叠视角，不可相加当成本。
合法早停减少实际count，不减少4/6分母。构造失败内部工作unknown，不伪造0physics证据。
原576模型/AE/proj预算和source encoding/proposal继承成本另列，不宣称总算力匹配。
无新neural/physics预测重放/训练/Qfit/adapter学习。

output NEW、32MiB cap含全部partial/final，512KiB final reserve，200MiB free floor；prepare
NEW且≤256KiB。wall≤600s从Store创建起dispatch/write检查，不抢占阻塞native调用。
actual operation_dispatch_intent/operation_returned/operation_failed收据，actual attempts/returns
分开、失败不退款。cleanup close即使日志/wall失败仍尽力；首错与secondary分开；各final
binding/journal/report尽力保留，不保证磁盘故障/SIGKILL后最终report存在。工程异常CLI exit2。

## MAIN CLI

在`/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_experiment1_common_controller_physical_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_experiment1_common_controller_physical_20261028_v1 prepare --manifest ABS_SELECTION --manifest-sha ACTUAL_SELECTION_SHA --output ABS_NEW_PREPARE
.venv/bin/python -B -m experiments.spectral_reachability.agent_experiment1_common_controller_physical_20261028_v1 verify --prepared ABS_NEW_PREPARE --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.agent_experiment1_common_controller_physical_20261028_v1 run --prepared ABS_NEW_PREPARE --prepare-sha ACTUAL_PREPARE_SHA --output ABS_NEW_OUTPUT --authorize-physics
```

prepare/verify仅saved CPU数组，不调用模型/环境；run必须MAIN显式authorization。
本版没有独立saved physical consume；报告/journal/PNG为后续独立consumer/RGB薄桥输入，
不能把producer strict或此物理runner自计账冒充独立物理验证。
actual报告为`results/agent_experiment1_common_controller_20261028_v1_cuda_MAIN01/report.json`，
strict为`results/agent_experiment1_common_controller_20261028_v1_consume_MAIN01.json`。
worker只核saved接口，不执行模型/物理或创建prepare；MAIN仍须独立manifest、审读及实际prepare。
RAM验证包含四臂完整fake-state transitions、88数组frame、8/80/816操作账、11个plan
tracking收据/臂、native7/velocity/角度discrepancy、原raw/context/goal不符拒绝、先保失败图、
native早停、step/close/report故障、四/六分母与外部authority拒绝；测试不是研究成功。
