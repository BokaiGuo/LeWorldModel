# Live raster intervention v1：独立新协议，不修复旧 renderer

独占三件套 stem `agent_p5_live_raster_intervention_20261028_v1`。MAIN 审计并运行；worker 仅 RAM 测试和只读 metadata build。无神经、训练、hydrate、像素补画、碰撞回调替换、阈值更改、重试或 outcome 选择。

## 旧失败与新观察的边界

旧 renderer report SHA `1b02ed4a2ca136b4dc7f8549073ff030620df8587d654eb2a07ca86036cad0c6` 始终绑定且失败保留；不生成它的 strict receipt。新协议不是旧 full-public gate 的容差版本，而是固定 suffix RGB/state/proprio 精确复现与本次渲染不变性的不同合同。

原生 observer 的单次事前冻结434子步运行 report SHA `e717c0eeb688d9b9b621b70a3caaa7496e6225f6733865b0357d13f3a30811fa`，strict status `native_observation_verified`，actual_order_matches=true。它保存了真实 world coords/WKB/order；两种 goal component 顺序（block 顺序两种均枚举）分别产生 coverage `.35743524812972455` / `.35743524812972466`，及对应 reward `.3762476296102364` / `.3762476296102365`。这说明本次保存几何的算术存在精确顺序敏感性；本次 actual 恰为旧末位值不等于历史 world-transform/顺序已认证。历史 membership/transform certification 仍 null，旧 renderer_release=false。不能把这次 observer 改称重跑旧 renderer 求过门。

prepare 要求上述固定实际 observer report，外部 SHA 绑定完整 observer strict receipt，并重新调用纯 saved observer.consume 全量相等；闭合 observer prepare、旧 replay prepare、旧失败以及各源 SHA。原目标/动作不从 observer 顺序或 coverage 挑选。

## 固定五状态、十色格与605子步

顺序固定 xy/yx/x_minus_x/zero_zero 四词原 constructor、原 seed、原 prefix10。全部44 prefix 帧五文件（rgb/state/proprio NPY、transition/full_public JSON）逐 byte 完全一致后，再按原顺序各 suffix 前5 raw actions，到原 frame15。第五个 goal 仅用原 scene report 的原 constructor/seed/reset，到其 frame0；不调用额外 geometry_witness，不设 goal_pose。五状态都达到后才开始颜色。

每状态 shots 固定 `baseline,visible,visible_restored,white,white_restored`。baseline 复用该状态真实最后 observation 图及 profile 捕获的 render before/after；随后原生 visible render、恢复原色后原生 render、white render、恢复原色后原生 render。两个 off→on→off triplet 共享中间 restored-off，不重复算 baseline、不多编码恢复图。原生原色为 LightGreen；visible 与所有恢复图均须与本次 baseline 全像素相同，包括 contact 红点；white 不以像素掩码作放宽。

固定5状态/10干预格/25shot槽，partial 或 native earlystop 不删分母、不补帧。

| 原生 entry/return-event | 上限/完整次数 |
|---|---:|
| constructor/setup/set_state/Space constructor/reset/close | 各5 |
| seed（含ctor） |10|
| controls |60|
| Space.step，dt=.01 |605=5+60×10|
| get_obs |65|
| render/debug_draw |85=65+20|
| goal body |145=5+120+20|
| Body constructor |160=15+145|

原生 render 内部创建临时 goal body 计入，不称 physics0。0额外控制用于渲染，但完整 job 有605真实子步。65 observation+10 intervention+10 restoration=85PNG与85RGB NPY。32MiB输出cap、2MiB终结reserve、200MiB free floor、900s工作墙钟，load包含在execute计时；run前的verify不含。一次alarm加每次durable写/dispatch核墙钟；失败cleanup可能额外耗时，完整consume要求wall<=900。超cap不降精度、不删图、不缩分母。

## 不变性与失败现场

原 native render call/return 由 sys.setprofile 捕获真实快照，未替换函数。快照包括完整 public_state（含原coverage/reward相关字段，不忽略/归零）、全部环境RNG、Python/NumPy全局RNG、实际current_time_step、space/body/shape本进程identity、goal_color RGBA及对象identity。颜色不在public_state所以必须单列。

render前后完整快照canonical bytes相等；干预时public_state/global_rng/topology须与本次anchor一致，仅goal_color不同。finally恢复原色对象并先持久化restoration receipt，再核状态/对象恢复，再真实render恢复图。每张已返回图、before、after先durable保存，再检查，不能用旧last_observation冒充内部异常后的现场。若原render内部抛异常，可能没有新图；仍尽力恢复颜色并保存receipt。

不读取coverage/reward来决定颜色、状态、顺序、是否允许渲染；它们仅随完整snapshot/transition保存与比较。suffix每帧RGB/state/proprio逐byte门；完整transition/full_public原样保存并生成exact JSON pointer diff（包含缺字段、signed-zero和单ULP，不allclose）。该diff仅报告历史不等，不能写成source_state equal。prefix五文件门保持严格。未公开arbiter/cache只保live、不声称完整认证。

## Output ABI：新 encoder 必须明确支持此版本

prepare schema `<stem>.prepare`，含完整bindings、plans/goal_protocol、全部65原帧source收据、targets5（四frame15/goal0的source_public/source_rgb真实路径SHA）、observer_authority、old_failure、v3准备SHA。

report schema `<stem>.report`，成功 `live_raster_recorded`，失败 `partial_engineering_failure`；顶层prepare_sha256/prepare_digest/protocol、rows5、all_states_attained、errors/meter、artifacts全清单、journal hashchain anchor/count、wall_seconds/profiler_released；old_failure原SHA、old_renderer_release=false、historical_full_public_equivalence=null、scientific_admission=false。

每 row：state_id/status/records/baseline/shots/cells。records四词0..15、goal0；每record含image/files/exact/differences，differences为`source_differences.json`收据，含transition/full_public完整递归差异。shots固定上文五项，每个files含`rgb_image={npy,png,pixel_sha256}`、before/after收据。cells固定visible/white，含anchor、原色身份、requested_rgba、restoration收据、off_before_shot。所有收据为`{path:相对output,sha256,bytes}`。

consume接口：`consume(prepare_path, external_prepare_sha, report_path, external_report_sha)`；saved-only，不加载native或模型。全目录/PNG/NPY/SHA、原动作、constructor/seed、逐操作native delta、605dt、journal/public phase、全部exact/diff、before/after/RNG/color/restoration重新验证。return-event并非成功定义，必须与无failed的durable returned及完整artifact相交。

独立receipt schema `<stem>.receipt`，status **`strict_live_raster_complete`**，不是旧strict status。固定`encoder_inputs`10项，state-major then visible/white：

```text
image_key = <state_id>/<color>
state_id, color
rgb: {path,sha256,bytes}  # NPY
png: {path,sha256,bytes}
pixel_sha256
before, after: {path,sha256,bytes}
fresh_snapshot_digest = SHA256(compact sorted canonical JSON of after完整快照)
fresh_public_state_digest = same canonical digest of after.public_state
```

receipt另含`source_comparisons`65项（state_id/frame/exact/differences收据），供下游完整保留历史diff；所有路径相对report.parent。encoder先外部SHA绑定report/prepare和本版source，重算consume严格相等后才读这10图；不得要求suffix after等于旧source_state，也不得将新receipt骗过旧encoder。baseline/restoration图不额外编码。actual render/Q未在worker执行，不预填成功。

## CLI（le-wm-main 根目录）

```bash
# 所有占位SHA参数由实际文件sha256sum提供；不把变量当未来结果。
.venv/bin/python -B -m experiments.spectral_reachability.agent_p5_live_raster_intervention_20261028_v1 prepare --observer-report results/agent_p5_native_geometry_observer_20261028_v1_diag_MAIN01/report.json --observer-sha e717c0eeb688d9b9b621b70a3caaa7496e6225f6733865b0357d13f3a30811fa --observer-receipt results/agent_p5_native_geometry_observer_20261028_v1_consume_MAIN01.json --receipt-sha RECEIPT_SHA --output results/agent_p5_live_raster_intervention_20261028_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p5_live_raster_intervention_20261028_v1 verify --prepare PREP --prepare-sha PREP_SHA
PYTHONNOUSERSITE=1 PYTHONPATH=/home/gbk/code/LeWorldModel/le-wm-main/.jepa-physics-site .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_live_raster_intervention_20261028_v1 run --prepare PREP --prepare-sha PREP_SHA --output results/agent_p5_live_raster_intervention_20261028_v1_render_MAIN01 --main-authorized
.venv/bin/python -B -m experiments.spectral_reachability.agent_p5_live_raster_intervention_20261028_v1 consume --prepare PREP --prepare-sha PREP_SHA --report OUT/report.json --report-sha REPORT_SHA --output results/agent_p5_live_raster_intervention_20261028_v1_consume_MAIN01.json
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p5_live_raster_intervention_20261028_v1.py
```

所有prepare/result/receipt排他写入，不覆盖旧文件；prepare2MiB文件上限和200MiBfloor。RAM fixture仅计数器/lookup图，不运行solver、不产生研究证据；包括完整execute→consume、suffix精确diff、prefix拒绝、接触cache seam、全分母partial、真实故障返回图先保存、原色对象恢复、RNG漂移、缺槽/PNG/命令/diff/计账变异拒绝。

本交付实际 saved-metadata build smoke 已通过：482 bindings、五固定 targets（四词15/goal0）；未创建prepare/result，未执行physics。字节数随自身三文件的最终SHA字段变化应以最终build打印为准。RAM另覆盖颜色render内部抛异常时有恢复receipt但无伪造after图；实际测试结果与最终三SHA见冻结交接消息。
