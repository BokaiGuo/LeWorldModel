# P2 contextual full10 原生物理 handoff（工程，非科学准入）

独占新source/tests/doc；不改producer、双LF consumer、原paired source或任何MAIN结果。
只复用原 `p1_moving_overlay_recompile_physical_20261027_v1` 的真实episode/native_hooks/Meter/Store，
独立8臂dispatch loop；不调用旧arms限定的execute/validate_plans，不patch globals、不执行旧结果替代。

## 真实authority与接口

planner report: `results/agent_p2_contextual_composition_runtime_20261028_v1_cuda_MAIN01/report.json`

SHA `498b45ed9f4c580064d8f2a5dfd00ac8542b7ce9d0b86e6587d9ab6a9639d49c`

v2 strict receipt: `results/agent_p2_contextual_composition_saved_consume_20261028_v2_MAIN01.json`

SHA `578c5bd93394ae9c3863cb420afe4cfd0c59633fbbea503b4012c029211e063f`

`material`调用v2 verify_receipt，完整重建native_strict_receipt，不仅核status/hash。
确认4slot/8plans/480候选/1328predict、FLOPs=null与科学门false。原prepare/source/model/全部arrays/
module buffers/journal/原task/原blind与新source/tests/doc递归导入闭包完整绑定SHA。
四槽顺序root0/off,root0/on,root1/off,root1/on，每槽joint80再conditional40。
编译选中digest与物理新plan seal分栏；原full10 rawfloat32原bits与SHA必须精确，不clip、不round、不再次归一化。

精确核对同condition的3RGB root0/5/10、原warm10 rawhistory、goalRGB、taskid、identity192与root latent context。
真实物理每臂由原resetseed/placement recipe恢复，warm10后才绑定原taskgoal，原完整native7+goal root hash精确一致。
条件臂suffix只在编译时用自身imagined中间context生成；执行时只派发原选中full10，不把imagined state注入物理，
不把teacher15 observed selector或旧equal144结果混接。无新模型调用。

## 真实dispatch与预算

MAIN独立授权后才I/O/构造env；constructor AST需匹配既有native源，且实例在hooks前space/render_buffer/latest_action为空。
复用原episode中真实10warm+10plan steps，native success/truncation立即早停，不补动作、不重试；8失败分母始终保留。
每次观测off/on/off，物理step一直with_target=False。off/on是输入RGB条件，默认goal_pose outline并不是taskgoal，
不改状态/goal_pose制造图；taskgoal bind不改变动态snapshot/renderer/goal_pose。

900s总wall、64MiB输出、2MiB final reserve、200MiB floor。上限：constructor8/close8/reset16/public_step160/
physics_step1632/set_state32/goal_setter24/goal_render16/goal_bind8/explicit_render528。
每操作attempt/complete/failed独立持久化，原Meter在调用前reserve，不把intent当实际计算；异常保留已用账与所有未开始行。
编译shared/AD/solve/replay成本继承原protocol，新增这里只计算实际物理counts；totalFLOPs仍unknown/null，不称等FLOPs。

## saved consume与后续RGB编码入口

独立核原journal单LF canonical hashchain、完整terminal、全部逐操作嵌套attempt-return、cap与原native计量恒等式，
逐branch/selected digest、实际requested/actual latest_action rawbits与dtype、真实native7/snapshot、reward/success、
每返回RGB triplet和全部原PNG文件/pixel SHA、warmup原source精确匹配、goal/outline不变、所有checkpoint不缺失伪补。
warmup reward严格用当时旧native goal，不将bind前reward拿当前taskgoal重算。

输出每臂initial/final与每个plan step原生native_success/native_reward/jointXY/agentXY/objectXY/velocity和wrappedangle诊断。
native_success原合同包含agent+object XY；coverage未由原wrapper返回，两个coverage字段均null，绝不用object-only代替成功。
native angle保留原min(abs(delta),2pi-abs(delta))公式，不擅自修理；wrappedangle另列。

`rgb_encoding_inputs`提供每臂真实selected-condition checkpoint0/5/10原PNG绝对path及原png basename、文件/pixelSHA、
taskid/blind/原task文件SHA、model/config SHA、原action mean/std和滚动rawhistory10：原warm10拼本臂executed_raw[:boundary]取末10。
无endpoint或早停则missing明确保留，不拿其它condition图代替；本模块不实际encode，也不提供虚构actual latent。
完整consume写独占receipt，stdout只输出路径/SHA和摘要。partial/pending/异常不会升级engineering_verified，科学门一律false。

## MAIN命令（prepare之后三文件禁止再改）

从le-wm-main运行；默认已绑定上面的实际planner与strict receipt，可显式覆盖但仍全量严格核验。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p2_contextual_composition_physical_handoff_20261028_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p2_contextual_composition_physical_handoff_20261028_v1 prepare --output results/agent_p2_contextual_composition_physical_handoff_20261028_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p2_contextual_composition_physical_handoff_20261028_v1 verify --prepared results/agent_p2_contextual_composition_physical_handoff_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREP_SHA
```

仅MAIN审读调度后执行，不由本worker启动：

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p2_contextual_composition_physical_handoff_20261028_v1 run --prepared results/agent_p2_contextual_composition_physical_handoff_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREP_SHA --output results/agent_p2_contextual_composition_physical_handoff_20261028_v1_MAIN01 --authorize-physics
.venv/bin/python -m experiments.spectral_reachability.agent_p2_contextual_composition_physical_handoff_20261028_v1 consume --output results/agent_p2_contextual_composition_physical_handoff_20261028_v1_MAIN01 --physical-report-sha ACTUAL_PHYSICS_SHA --receipt results/agent_p2_contextual_composition_physical_handoff_20261028_v1_consume_MAIN01.json
```

本交付只有RAM mock与saved metadata接线；不运行神经、physics、网络或P3作业。后续真实物理/真实RGB编码尚未发生。
独立P1和fresh队列均未准入，已知2roots不是新独立样本，无P1/P2/physical scientific release。

## 实际工程检查

已用MAIN真实strict receipt做只读material构建：8plans、完整输入闭包，未写prepare、未实例化env。
RAM测试复用原native桥的人工ArrayEnv与真实PNG编码/hashchain，不是原生物理证据；测试包含全8分母、
10+10真实回调顺序、terminal/truncation早停、事故保留、当前goal与warmup旧goal隔离，以及raw/goal/reward/
PNG/state/history/resource/branch/terminal篡改拒绝。正式prepare/verify/run/consume仍由MAIN审读后执行。

最终RAM回归：本handoff40tests，与冻结v2 saved consumer35tests联合75passed（11.48s）。
实际只读material返回8plans/687bindings/797281bytes；无生产prepare写入、无环境实例化。
没有启动生产physics worker、GPU作业或P3重复作业，没有写MAIN结果。
