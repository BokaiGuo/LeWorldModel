# Archive target-overlay probe v1 — private physical/consumer contract

接口供后续paired CUDA消费者对齐；MAIN独占prepare/物理/CUDA。本模块不执行神经，
不调用旧probe.execute、不修改任何旧globals/source/results/shared docs。
source/tests/doc已交付冻结，下述字段为固定消费者合同；最终SHA见交付消息。

## 来源/控制/资源

固定episode `[1,6229]`、lag0、20个原float32 raw动作，0/5/10/15/20五checkpoint。
每根原seed0 reset取得native goal，第二次seed0 reset placement archive state0+同goal。
不再setter，不改goal_pose/goal_state/相机/颜色/动作，只有checkpoint同body的
with_target off→on→off。每次前后完整snapshot/goal/renderer必须相等，off两图逐位相同。
未来archive state/RGB和旧off仅作为执行后的evaluation，不参与动作控制。
提前terminated/truncated即停，保partial及缺失checkpoint，不补步/换episode。

source probe source SHA `62f8fcbffacbc14cdf3641654eb2c4adb9fcadea1c41a6e762a1a0c79034d172`；
prepare SHA `899485c8ae93827ef02100f64429dd293422f617055e81c37dc1f1859b6db0d7`；
report SHA `affb90a6e67c096623b62ac1d4a1f9f2e11ceaa5d5fe8436f365d39536de2e05`。
绑定原execution_binding、prepare、report、完整旧source closure、H5 stat/schema/原切片hash及10旧lag0off PNG。
H5身份仅stat+选定切片，不是46GB全文件SHA。复用已修proprio/codec的只读reader。

完整计量caps：reset4、set_state8、physics_step408、goal_setter4、goal_render4、
explicit_render30、public_step40。每次attempt/return独立记，reset内部physics/render也计。
每次实际command必须等于archive action[step-1]的float32 bytes；无clip/tanh/再除100。
cap8MiB，final128KiB，floor200MiB。0neural/GPU/training，2分母永保留。

## Prepare / execution binding

prepare schema `p1_archive_target_overlay_probe_20261027_v1.prepare` 精确顶层字段：
`schema, source, bindings, contract, selection_fixed_before_execution, archive_future_state_for_evaluation_only`。
`source = {original_archive, bindings, source_identity_scope}`；原archive为旧prepare/report同一字典，
source.bindings包含原封存closure、旧off PNG和原report/prepare/binding文件SHA。
顶层bindings再增加本source/tests/doc SHA；contract包含固定episode/lag/horizon/checkpoint/flags/caps。
`selection_fixed_before_execution=true`，`archive_future_state_for_evaluation_only=true`。

新output的 `execution_binding.json` 精确字段为
`{prepared: absolute_path, prepare_file_sha256: external_prepare_sha}`。
report的prepare_id为复用source.digest的hash（sort_keys默认空格JSON），不是文件SHA；
消费者必须分别绑定execution_binding文件、prepare文件SHA以及report文件外部SHA。

## Report / checkpoint EXACT schema

report schema `p1_archive_target_overlay_probe_20261027_v1.report` 顶层：
`schema,status,prepare_id,approved_probe_report_sha256,approved_probe_prepare_sha256,rows,
fixed_denominator,expected_frames,counts,contract,errors,first_error,journal_chain_tail,
inputs_final_verified,wall_seconds,new_neural_calls,P1_admission,scientific_winner`。
完整status=`overlay_probe_processed`；合法早停=`overlay_probe_partial_native_stop`；
异常=`engineering_failed`，非完整不重命名成完整。早停不是工程错误；
无错误时核实际attempted==completed、physics=2reset+10public、set_state=2reset、
goal_setter=goal_render=reset、explicit_render=实际frames数，public=实际返回步数。
只有两行全full20才核全部fullcap。后续scientific consumer可拒绝partial，但不能改其工程分类。
`fixed_denominator=2,expected_frames=30,new_neural_calls=0,P1_admission=false,scientific_winner=null`。

rows按episode1/6229固定。每row预先有
`episode_ordinal,status,frames,steps_completed,missing_checkpoints,off_replay_checks,terminated,truncated`；
完成两reset后增加 `native_goal,goal_pose,renderer,placed_snapshot,requested_state0,
state0_absolute_error,state0_max_absolute_error`。
完整row.status=`overlay_full20_completed`，提前停=`native_stopped`，异常=`engineering_failed`；
尚未执行=`not_started`。frames完整15条，顺序 `(step0 off,on,off),(step5 ...),...,(step20 ...)`。

每帧receipt精确字段：

```text
step: 0|5|10|15|20
overlay_index: 0|1|2
with_target: false|true|false
png: flat output-relative filename e{episode}_s{step:02}_{overlay_index}_{off|on}.png
png_sha256: saved PNG file SHA
pixels: {shape:[224,224,3],dtype:"|u1",sha256: contiguous pixel bytes SHA}
pre: invariant
post: invariant
comparison: RGB/archive descriptive metrics
old_off_match: null on index1; {file,pixels,state,snapshot: bool} on indices0/2
off_replay_match: null on indices0/1; bool on index2
```

invariant精确字段 `snapshot,goal_state,goal_pose,goal_rgb,renderer`。
snapshot复用原native helper（state/state_float64_bits+agent/block position/velocity/
angle/angular_velocity/force/torque/sleeping/float64_bits），明确不覆盖solver hidden cache。
goal_rgb为原native缓存_goal的array_id，renderer为
`render_mode,render_size,window_size,render_action,relative,action_scale,dt,control_hz,variations`；
variations是原variation_space.value原样JSON化，不包含可变with_target。

comparison精确字段 `rgb_MAE,rgb_max_absolute_error,different_pixels,rgb_exact,green_actual,
green_archive,green_intersection,green_actual_only,green_archive_only,green_xor`。
RGB先转int16再相减；绿色mask固定G>R且G>B，不设阈值/择优或称修复。
旧off核验包括两次off的PNG文件SHA、pixel、state和snapshot，全部必须true。

off_replay_checks每条 `{step,first_frame_index,last_frame_index,bit_exact}`，
完整row有5条，索引指向本row.frames，分别0/2、3/5、6/8、9/11、12/14。
完整triplet的绝对JSON指针可用 `/rows/{i}/frames/{3*j+k}`，i=0/1、j=0..4、k=0..2。
消费者必须验证step/overlay_index/flag，不能仅依赖位置或把partial补齐。
取paired off/on通常k=0/1，但保留全部k=2 replay证据，不删除重复编码成本。

## Durable journal

`journal.jsonl`沿用hash chain `{index,previous_sha256,kind,...,event_sha256}`，tail绑定report。
kind `run_started` 带prepare_id/contract；operation_attempted/completed由Meter记实际资源。
`native_return` 带episode_ordinal/step、requested_raw/actual_raw array_id及float32 bits、
state/snapshot/reward/terminated/truncated；中间步不另存PNG。
`fresh_overlay_frame` 带episode_ordinal和完整receipt；**新PNG+receipt先保存，随后才拒绝**
snapshot变化/off replay不等/旧off不等。branch_finished记录完整row，run_finished记录status/counts/errors。
新输出所有frames都是真实新render，旧PNG绝不复制充当新帧。

## MAIN命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_archive_target_overlay_probe_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_archive_target_overlay_probe_20261027_v1 prepare --output results/p1_archive_target_overlay_probe_20261027_v1_prepare_main01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_archive_target_overlay_probe_20261027_v1 verify --prepared results/p1_archive_target_overlay_probe_20261027_v1_prepare_main01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.p1_archive_target_overlay_probe_20261027_v1 run --prepared results/p1_archive_target_overlay_probe_20261027_v1_prepare_main01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/p1_archive_target_overlay_probe_20261027_v1_main01 --authorize-physics
```

19 RAM tests通过，仅协议/算术/失败partial证据；本侧不prepare/物理/CUDA，
不与MAIN并发actualbuild，无overlay改善或科学准入断言。
