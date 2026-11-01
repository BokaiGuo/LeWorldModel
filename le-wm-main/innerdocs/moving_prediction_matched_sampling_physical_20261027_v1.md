# 新同 prediction 预算采样基线：四计划物理执行

仅此 stem 的 source/tests/innerdoc；不改旧 source/globals/execute、封存预算或共享文档。
本模块实现而不启动真实 prepare/物理；由 MAIN 审读、提供真实外部 SHA，HOST 执行。
采用 experiment-code 的先固定实验、真实输入、种子和失败日志要求；用户限定三文件、
无神经/物理实跑优先，不额外创建训练/绘图/notes 文件。技能引用的两个模板文件不在本机。

## 给 Jason / MAIN 的固定 consume 接口

本会话没有线程消息工具，不能声称已直接通知 Jason。请转交以下接口约束：

```python
# 只允许这个固定 producer，不接受任意 module/callable 路径
from experiments.spectral_reachability import moving_prediction_matched_sampling_cuda_20261027_v1 as producer
receipt = producer.consume(output, external_report_sha256)
# receipt.schema == producer.NAME + '.consume'
# receipt.rows: 原顺序 [(0,CEM),(0,kernel_faithful_mppi_nominal),
#                     (1,CEM),(1,kernel_faithful_mppi_nominal)]，完整4条
```

该签名和字段来自 Jason 当前源码，不代表其尚未完成的文件已获最终 SHA。
MAIN 在最终 freeze 后提供 source/tests/innerdoc 三个真实文件 SHA 和实际 report SHA，
本模块不固化猜测值、不自动发现一个“最新结果”、不制造本地 MAIN 身份。
consumer 仅调用 producer 自己的 saved-only consume（含 NumPy 重建与其 prepare closure
重建检查），不运行模型或调用 producer.run。

每条 row 保留：schema、source_slot、arm、task_id、blind_sha256、task_export/task_sha256、
task_contract_sha256、commands（literal float32 raw10×2）、logits、selection_sha256、
whole_run_report_sha256；要求 with_target/physical_execution/selection_uses_physics/P1_admission/
official_MPPI 均 false。selection SHA 按 producer 原内容重建，不含 selection_sha256 与
whole_run_report_sha256。producer prepare slots、完整 source/model/task/artifact closure 和
外部 report SHA 一并验证，不借旧采样或物理结果。

consume 必须确认每组 144 次 predict API、144 predictor forward samples（真实 batch1），
四组各自完整，共576；AE/pred_proj 只是重叠模块账，不加到 API/samples 上。
本研究比较仅说这两采样算法同 predict 调用与 forward 样本，非等 FLOPs/总计算，
也不要求 walltime、导数或其余资源相等。继承的规划成本与本模块物理成本分开保存。

## MAIN 外部 selection manifest

严格字段（所有路径须 canonical absolute，所有 SHA 为实际 lowercase hex64）：

```text
schema: moving_prediction_matched_sampling_physical_20261027_v1.selection
sourcepack: 原 moving sourcepack 绝对路径
sourcepack_sha256: c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6
source_prepared / source_prepare_sha256: 原真实来源 prepare 路径及 SHA
roots: [{root_slot:0,root_sha256:真实第一根}, {root_slot:1,root_sha256:真实第二根}]
arms: [CEM, kernel_faithful_mppi_nominal]
authorities: [唯一如下 entry]
selection_fixed_before_physics: true
physical_outcome_selection: false
wall_seconds: 600（允许预声明更短正整数，不得更长）
```

authority entry 严格字段：authority 固定
`moving_prediction_matched_sampling_cuda_20261027_v1`；output 为新真实 CUDA 结果目录；
report_sha256 为 MAIN 外部真实报告 SHA；source_sha256/tests_sha256/innerdoc_sha256
为 MAIN 最终确认的 producer 三文件 SHA。这里不提供假 SHA 的可执行 manifest。
外部 manifest SHA 绑定这些信任锚，但不是密码学 MAIN 签名或科学准入证明。

## 下游 comparison / Jason sampling RGB 的稳定 ABI

```python
from experiments.spectral_reachability import moving_prediction_matched_sampling_physical_20261027_v1 as physical
receipt = physical.consume(physical_output_absolute, MAIN_actual_physical_report_sha256)
```

仅 saved-only 验证，无神经/physics。下游先验证本模块最终冻结源 SHA，再调用这个接口。
返回 schema 为本 stem + `.consume`，固定字段：

- `rows`：原物理 report 的全部4行，root-major CEM/nominalMPPI顺序不变，含 failed/not_started。
- `checkpoints`：只包含实际存在的 checkpoint，完整运行12条（4×0/5/10），不补缺。
  每条有 `branch_index/root_slot/arm/selection_digest/selected_plan/checkpoint/png_path`。
  `checkpoint` 原字典有 phase、step、png、file_sha256、pixel_sha256、
  **state（actual有限7维）**、snapshot、observation_state（同7维）。
- `selected_plan` 同 rows 中原选中计划，含 root_sha256、task/blind/task_contract SHA、
  compiler authority/report SHA、原 compiler_selection_sha256、own_plan_sha256、
  selection_digest、literal commands；不能只按 arm 或 task_id 模糊 join。
- `rows[i].root_state/goal/final_state` 是原7维；`initial_metrics/final_metrics` 是原 native
  与诊断标签，未观测为 null。goal由原source精确绑定，不用于控制输入或候选选择。
- `counts` 原实际 meter；`report.actual_operation_receipts` 带连续 operation_index、
  resource、branch/root_slot/arm/selection_digest、phase/step/raw_action（step操作时）、
  status returned/failed；`attempted/completed` 不由预期流程倒填。
- `prepared/report/bindings_sha256/physical_report_sha256/fixed_branch_denominator=4`
  保留 source→plan→execution→PNG 完整身份链；所有 PNG bytes/pixels 核验。
- `checkpoint_complete` 仅当全部4行都有0/5/10才 true，不代表 guard之外的科学准入。
  下游 RGB 如需全8幅5/10，必须对此和工程status独立检查；不得换末态、减分母或补物理。
  有 durable 完整终态账的 earlystop/工程partial可保留原4行；缺失/冲突 dispatch-return
  journal 的损坏 partial 则拒绝 consume，不伪造可验证性。
- scientific_admission/P1_admission/independent_confirmation/equal_total_compute/equal_FLOPs
  恒 false，new_neural_calls=0。

`assemble(report, events, prepared)` 是可 RAM 测试的纯数据校验 seam；正式下游应使用
`consume` 的外部SHA/closure/PNG gates。journal operation kinds 是
`operation_dispatch_intent/operation_returned/operation_failed`，receipt 内的 operation_index
为唯一索引；intent不是actual attempt。关闭env也有这些实际 receipts，cleanup可超wall
但仍最多4次，不重试。旧RGB assembler的 operation_attempted/completed 不能直接套用。

## 固定真实物理流程

每个原 moving root × 每个算法各建 fresh env，共4分支；即便 raw commands 相同也不去重。
使用原 seed reset → 原 placement recipe（placement seed0）reset → 原 float32
warmup10（前5个 [.1,0]、后5个 [0,.1]），核验真实 raw bits、RGB/state/snapshot
0/5/10、blind/task/goal/root 身份全一致，再离线 goalbind，执行已封存 raw10。
不重优化、不 tanh 二次变换、不除100、不补步、不重试、不追加候选或 replay。

底层仅复用 `p1_constructive_moving_history_physical` 的 branch/native_hooks/capture 等，
以及 extended physical 的 task_bridge/native_constructor_contract/静态 closure；
不调用任何旧 execute/material/结果报告。旧逻辑的原 placement/纯render guard 保留。
task_bridge 验证 RGB history+goal RGB/raw history SHA、输入历史、task contract、
identity192 Q、logit cap 和 logits→float32 tanh→raw10 精确一致。

env 固定 native PushT、RGB224、with_target=False、render_action=False。
goal RGB 作为 evaluator 元数据绑定，绝不显示目标；不插 goal_pose。
绑定前后 dynamics/goal_pose 不变，禁止额外 reset/state setter/render/physics。
保留 native 7dim state、goal、native_success、native_distance、原 native angle 语义、
wrapped/object 诊断；不把 wrapped 指标替换 native label，不混入作者 reported 值。

所有 warmup0..10 与 plan0..实际末步均真实 capture，完整分支22 PNG，
plan0/5/10 checkpoint 从这些 capture 选取，不做额外 physics 或补图。
native terminal/truncate 即停；earlystop 缺5/10保持缺失，不能填末态或0。
guard 失败前保留可用真实图像与 incident。constructor/未运行没有 native label 时为 null。
四分母永不缩小，分列 full_horizons、native_successes、early_stops、engineering_failures、not_started。

## 真实代码计量范围与固定 caps

| 资源 | 全4分支最大 attempted | 依据 |
|---|---:|---|
| environment_construct / environment_close | 各4 | 每分支各一次；close 失败不重试 |
| reset | 8 | 原seed、placement各一次 |
| set_state | 16 | 每次 native reset 2次 setter |
| goal_render | 8 | 每次 reset 内部生成 goal RGB一次，即便 with_target=False |
| goal_bind | 4 | 离线元数据绑定一次，不含物理 |
| public_step | 80 | warmup10 + 至多plan10 |
| physics_step | 816 | reset setter共16步 + public_step各10子步×80 |
| explicit_render | 88 | warmup0..10 + plan0..10，各11 |

这是不同轴/重叠层级，不相加成为总成本。动态 actual attempted/completed 只按
实际 callback 进入/正常返回记录；pre-dispatch journal intent 成功后才进入并计 attempt。
运行中异常不退款；正常 native 返回后写日志失败也保留 completed，逐行 own/warmup
实际计数在 finally 从 entries/returns 对账，不伪造完整 horizon。
dispatch journal 故障关闭后续 Ops dispatch；四行仍保留。

constructor 在 hooks 安装前：静态固定 native 源码检查其没有 reset/step/render/setter/setup，
返回后检查 space/render_buffer/latest_action pristine。正常构造可说明原 native 无物理；
中途异常或 constructor-return 日志失败不能推断内部成本为0，标 unknown。
physics_step 是 hooks 观察范围；reset goal_render 与 setter 本身都有嵌套成本，不能双算。
原生 partial step 即使 public_step 未返回，已进入/返回的 space.step 仍保留真实观察值。

输出累计硬上限32MiB（含 journal/PNG/partial/final），预留512KiB供最终报告，
磁盘 floor200MiB，prepare≤256KiB。wall≤600s 从 Store 创建起，于每次 dispatch
及普通 write 前检查；不能抢占阻塞的 native call。cleanup/final report 允许越过 wall
以保存现场，但仍受 output/floor 限制；不谎称绝对进程600s强制中断。

## 命令（只供 MAIN/HOST，不由本次实现者执行）

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_moving_prediction_matched_sampling_physical_20261027_v1.py
# 以下须 MAIN 提供真实参数、审核后 HOST 执行；无默认SHA或本地自授权
.venv/bin/python -m experiments.spectral_reachability.moving_prediction_matched_sampling_physical_20261027_v1 prepare --manifest MAIN_MANIFEST_ABS --manifest-sha MAIN_EXTERNAL_SHA --output NEW_PREPARE_ABS
.venv/bin/python -m experiments.spectral_reachability.moving_prediction_matched_sampling_physical_20261027_v1 verify --prepared MAIN_PREPARE_ABS --prepare-sha MAIN_EXTERNAL_PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.moving_prediction_matched_sampling_physical_20261027_v1 run --prepared MAIN_PREPARE_ABS --prepare-sha MAIN_EXTERNAL_PREPARE_SHA --output NEW_OUTPUT_ABS --authorize-physics
.venv/bin/python -m experiments.spectral_reachability.moving_prediction_matched_sampling_physical_20261027_v1 consume --output MAIN_PHYSICAL_OUTPUT_ABS --report-sha MAIN_EXTERNAL_PHYSICAL_REPORT_SHA
```

本模块无 torch/model/encoder 推理；没有训练/adapters/Qfit。gym/native 只在授权 run
的 factory 内导入，prepare/verify/导入只读 metadata。真实 sourcepack、producer report、
producer frozen SHA 与 MAIN 审核缺任何一项均不能准备真实执行。
所有 RAM tests 只是工程验证，不是科学证据。known moving2 不独立；
existing P1 和 independent freshness 不自动开放，scientific_admission 永远 false。

2026-10-27 冻结验收：本模块最终79项RAM测试通过（5.33s），覆盖实际7维checkpoint、
完整4分母、earlystop/工程失败、逐分支故障、原raw/history/goal/task绑定、meter实际entry/return、
journal损坏、SHA门禁、32MiB/floor/wall以及纯saved下游assemble；不是科学证据。
加入下游consume前的有界联合回归192 passed（48.03s），包含原baseline physical、
extended physical与Jason新sampling CUDA测试。本次没有真实prepare、神经或物理执行。

MAIN通知新CUDA实际report（本模块没有替MAINownconsume）：
`results/moving_prediction_matched_sampling_cuda_20261027_v1_cuda_MAIN01/report.json`，
外部SHA `60872ad114700731d82d293a616a4dede54a53b7be6514ad0cdf231d838d3ed8`。
该值由MAIN在最终manifest传入，非模块默认值或本地自签；producer三文件finalSHA同理由MAIN提供。
此三文件自本次SHA交付后冻结，下游消费者应按冻结接口接入，不改旧budget合同。
