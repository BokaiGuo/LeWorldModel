# Moving target-overlay source v1 — source-only private interface

分工更新：本stem仅原两slots的construction/replay0/replay1、RGB3及真实witness terminal goalRGB；
GN/D2P真实plan分支由Jason的新独占stem实现，不在本程序读取/执行。后续CUDA由MAIN协调。
三文件已交付冻结，以最终消息三SHA为准。仅这三个新文件，不改旧封存或共享doc。

## 输入与科学边界

固定原moving sourcepack `results/p1_constructive_moving_history_source_20261027_v1_main01/sourcepack.json`
SHA `c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6`，原prepare
`results/p1_constructive_moving_history_source_20261027_v1_prepare01.json`
SHA `c1292f0ebfb1827e5da92b32ab13d2e4a12c025c4741277ab4c4f478d6ee8cb9`。
源码SHA/依赖、sourcepack/prepare/execution_binding/journal chain、原blind/certificate文件SHA、
各phase已commit witness及全部126旧source PNG绑定。certificate仅hash，不读其witness数组。
不新选root、seed、sign、placement或action；不根据outcome选参数/flag/arm。
原两moving roots/task不变，不是新独立样本。无神经/训练/P1 admission或修复断言。

源代码只复用旧source数学、saved loader/journal/Store与snapshot工具，自己的episode/execute；
不调用旧episode/execute，不patch旧module globals。runtime hooks只包新env实例以精确记账/防postsetter。
保持原gym factory `with_target=False,render_mode=rgb_array,resolution=224`，其他参数同原source。
原default renderer goal_pose不是construction goal；**从不改goal_pose/颜色/相机来对齐任务**。
on图仅临时with_target=True真实render，随后恢复False，不能把on图贴绿合成。

## 固定流程与预算

每slot依原recipe两reset：slot.reset_seed取native goal、seed0 placement；10实际float32 warmup
前5[.1,0]后5[0,.1]；从实际post-warmup root用原固定sign/cap生成原witness，
完整10步terminal创建task goal。然后原样两次独立reset+warmup，在原时点attach构造的
goal及**off goalRGB**，执行相同witness并验证native terminal/reward及逐步physics。
两replay对off/on history及真实terminal RGB均与construction逐位一致。

每episode保留原21次off（step0..20），在warmup0/5/10和witness terminal20各扩为
及witness prefix15，均off→on→off三render：每episode31帧，共6episodes/186PNG；不去重或复制旧图片。
step15配对为预声明15→20 factual teacher-forcing保留真实label/context，不是按新结果扩展。
每次off均逐位核原PNG bytes/pixels/state/snapshot；原warmup actual float32 bits与witness commit固定。
triplet前后完整snapshot、goal_state/cache、default goal_pose、renderer variations不变；
off两帧逐位一致。失败保存已返回新帧/receipt后拒绝，所有分母/blocked rows保留。
合法native early stop为partial，不工程错误、不补步、无blind payload虚构成功。

完整caps：reset12 / public120 / physics1224 / state_setter24 /
goal_render12 / goal_setter16（含4bind）/ goal_bind4 / explicit_render186。
实际attempt/return/失败计账；partial核physics=2reset+10public、set_state=2reset、
goal_render=reset、goal_setter=reset+goal_bind、public=实际返回步数、explicit=实际frames数，
只完整六episode才要求fullcap。600s wall、8MiB output cap、512KiB final reserve、200MiB floor。
未序列化solver warmstart cache，不将exported snapshot称完整内部solver身份证明。

## 给Jason/Poincare的稳定接口

schema前缀独立：`p1_moving_target_overlay_source_20261027_v1`。
prepare envelope精确字段 `{schema: NAME+'.envelope',plan_sha256,gzip_base64}`，内部plan schema
`NAME+'.prepare'`，含contract、bindings、old_inputs、old_source_prepare_id、slots、
reference_index_sha256、command_authority。不是旧source `.prepare`或旧schema改名。
execution_binding为 `{prepared: absolute_path, prepare_file_sha256: external_sha}`。

output `report.json` schema=`NAME+'.report'`。status完整=`paired_moving_overlay_recorded`；
合法partial=`partial_native_stop`；异常=`engineering_failed`。顶层固定包含
`schema,status,prepare_id,contract,rows,blind_exports,counts,errors,first_error,journal_chain_tail,
source_verified,full_protocol,new_neural_calls,new_independent_roots,P1_admission,scientific_winner,wall_seconds,
fixed_denominator,expected_episodes,expected_frames`（分别2/6/186）。
rows固定6项 `(s0 construction,replay0,replay1,s1 construction,replay0,replay1)`；
每row `{key,source_slot,kind:'source',phase,status,public_returned,frames,missing_triplets}`，成功后加
native_success、goal_pose、root_goal_sha256；partial可带reason。状态not_started/running/
source_episode_completed/native_stopped/engineering_failed，不把construction标native_success。

row.frames按实际渲染顺序，每条summary为
`{reference_key,overlay_index,png,file_sha256,pixel_sha256}`，reference_key为全episode step字符串0..20。
triplet overlay_index=0/1/2对应off/on/off；非triplet只有0。完整每row31条。
完整triplet在row.frames位置：step0 indices0..2，step5 7..9，step10 14..16，step15 21..23，step20 28..30。
missing_triplets是尚未完整通过三图校验的step字符串，初始['0','5','10','15','20']，不补齐partial。
必须核reference_key/overlay_index，不能只按位置给partial补齐。

`journal.jsonl` hash chain采用原source.digest规范。`fresh_overlay_frame`包含key/source_slot及
完整receipt：reference_key、overlay_index、with_target、png/file_sha256/pixel_sha256、
pre/post invariant、old_off_match、off_replay_match、reference_error（正常null；旧reference读取失败保错误类型/消息后拒绝）。
invariant含snapshot、goal_state、goal_pose、goal_rgb_sha256、renderer；renderer完整variation_space.value只读JSON化。
old_off_match为off时file/pixels/state/snapshot四bool，on时null；off_replay_match仅index2为bool。
`native_return`含实际requested/actual raw及dtype/bits、返回state/snapshot/reward/terminal标志。
`placement_recipe`、`goal_attached`、operation_attempted/completed/failed、branch_finished/run_finished均durable。

blind_exports固定按成功slot累积，每条 `{source_slot,artifacts:{off,on},old_root_sha256}`。
每artifact `{path,sha256,fields,rgb_history_pixel_sha256,goal_pixel_sha256,raw_action_sha256}`。
文件名 `s0_off_blind_inputs.npz`,`s0_on_blind_inputs.npz`（slot1同理）。
每NPZ严格只有 `rgb_history:uint8[3,224,224,3]`, `raw_action_history:float32[10,2]`,
`goal_rgb:uint8[224,224,3]`，无state/witness/证书/候选/任务未来。
history取**新replay0实际0/5/10**，goal取**新construction实际witness terminal20**对应off/on。
新replay1真实terminal和history均必须匹配才导出。

给下游的动作控制接口不变：原seed/placement/warmup/goal/原plans由各自封存来源绑定，
本新report只提供新RGB配对及provenance，不提供新的planner候选或自动接consumer。

## MAIN命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`，实际prepare/物理仅MAIN执行：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_moving_target_overlay_source_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_target_overlay_source_20261027_v1 prepare --output results/p1_moving_target_overlay_source_20261027_v1_prepare_main01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_target_overlay_source_20261027_v1 verify --prepared results/p1_moving_target_overlay_source_20261027_v1_prepare_main01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_target_overlay_source_20261027_v1 run --prepared results/p1_moving_target_overlay_source_20261027_v1_prepare_main01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/p1_moving_target_overlay_source_20261027_v1_main01 --authorize-physics
```

23 RAM tests通过，覆盖完整186帧/实际全账、旧off逐位对照、真实on terminal盲输入、
合法early stop、动作替换、渲染不纯、reference读取失败、journal/final fsync失败及容量。
本侧未actualbuild/prepare/物理/GPU；不是实际overlay效果、修复或P1完成证据。
