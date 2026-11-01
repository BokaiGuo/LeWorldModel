# 固定四 observed-prefix CEM64 物理 adapter

独占新 stem `agent_observed_CEM_physical_20261028_v1` 的 source/tests/innerdoc。
不改 fixed24、原 observed helper、producer、封存 results 或共享文档。没有派生
agent。使用 experiment-code 技能的固定输入、逐次资源收据与失败持久化模式；
用户限定只有三文件、RAM tests，所以不新增 plots/notes 或实验目录。
技能引用的两个旧 `~/.claude/skills/experiment-code/references/` 文件不存在；
按本地实际源码和本任务约束实现，不以缺失模板扩展写集。

本实现只到准备交付 MAIN 审读/测试/prepare/主机物理。实现者没有调用 prepare、
run、GPU、physics、训练或下载。RAM 状态机不构成 native/scientific evidence。

## 精确重建路径：存在，但逐支运行时仍必须验真

完整读过 `agent_baselines_resume_20261027_next.py` 和其 innerdoc，以及
`moving_history_observed_suffix_physical_20261027_v1.py` 和其 innerdoc。
只复用后者的 saved 原证据 loader、paired_capture/native_hooks 和原底层
place/capture/metric/Store 工具，不调用任何旧 branch/execute/episode/witness，
不 patch 旧 globals。

已执行只读 `observed.load_original`：原 fixed4 完整验收，每支 warmup0..10
有11帧、plan0..5 有6帧，原 report/journal/PNG、source recipe/history/root/goal
完整 closure 存在。因此没有“只能 reset(savedpoststate)”的输入缺口。
这不是声称尚未运行的新 CEM 物理重建已成功。

顺序固定为 slot0/1 × `line_search_GN`/`full10_conditional_D2P`，CEM arm 均为
`observed_prefix_CEM64`。稳定键 `(root_slot,prefix_arm,arm)`，不能只按
slot/CEM arm 合并两种观察 prefix。

每支 fresh env 原 seed reset，然后根据原 root_seed/placement recipe 重新
`sample_state` 并 placement reset(seed0)。后者 state 是重新计算的原初始
placement，不是保存的 post-warmup/post-prefix state。实际 warmup10 原 raw
float32（前5 `[.1,0]`，后5 `[0,.1]`），逐帧核对原 PNG byte/pixel SHA、
actual7 state、observation_state、两个 body snapshots 及 float64 bits；
另核对 source warm0/5/10 的原 history/state/dynamics 和 root+goal 哈希。

offline 绑定原 goal state/RGB，只调用原 `_set_goal_state` 和 `_goal` 赋值；
禁止额外 step/render/set_state，禁止修改原 goal_pose。渲染 `with_target=False`、
`render_action=False`、RGB224，goalRGB 不叠加到控制观察。原 prefix5 原字节
执行，plan0..5 全帧精确核对原保存观察。所有17个 paired 帧检查与 goal 检查通过，
才在 step6 前记录 `suffix_admitted_after_exact_prefix`，在同一 env 执行自己
sealed CEM winner suffix5。中间没有 reset/setter、近似 restore 或另一支回退。

公开的 dynamics/body bits 和重放路径可认证；未导出的 solver warm-start/cache
没有独立序列化，不称 hidden cache byte certification。旧 receipts 没有独立记录
goal_pose，本实现只检查自己的 bind 不改变它，不伪造旧 goal_pose 证据。

## 真实 producer/consumer authority（不是未来 SHA）

MAIN 已给出、实现者只读 hash 核对：

- producer source `agent_baselines_resume_20261027_next.py`：
  `bc8e588bc11a4ab50572ace8a99ef397f05f0ee8b6edfab5ae5a42f022528e10`
- CUDA `results/agent_baselines_resume_20261027_next_cuda_MAIN01/report.json`：
  `83f9b6fa5ad991022af6bc38063c0fa6c72fca350578692b3c30f4a7e4854a6b`
- 原 `baseline.consume` 精确 return 保存为
  `results/agent_baselines_resume_20261027_next_consume_MAIN01.json`：
  `b81abd4c2f6600bae69513228a9b6896e2a67f09c3e0ab46fc6901a4034f47d3`
- MAIN 保存 wrapper `consume_observed_CEM_MAIN_20261028_v1.py`：
  `442a153ca01c525d1315146d4ffeac2debfce7432486c5adb7fb1db171e8c9da`
- 原 prefix 来源 `results/p1_constructive_moving_history_physical_20261027_v1_main01/report.json`：
  `a5322dc4f483e82b5ed9d7b3996cf8a63845261d65774f90a4ae341773016aaf`

这些新 CEM 值只在本说明/MAIN 命令中列出；源码不 hardcode 它们为默认 authority。
参数 `consumer_source_sha256` 指实现真正 consume 函数的 producer source，
所以必须等于 producer_source_sha256；另用 `consumer_wrapper_source`/
`consumer_wrapper_source_sha256` 绑定 MAIN 保存工具，不能混淆。
wrapper 只验 SHA，不执行其代码。只接受固定 producer 名，不允许任意模块。

`material(arguments)` 先验证这些外部文件 SHA，再导入固定 producer，调用其
saved-only `consume`（无模型/物理），要求 return 和 MAIN 保存 JSON **完全相等**。
不得仅凭 raw producer `commands_released` 放行，不允许 MAIN wrapper 添加
metadata 来填缺字段。再次 saved-only verify producer prepare/input/module arrays
closure，完整复制 bindings，不复制大模型/图像文件。任何缺失或漂移阻塞 prepare。

纯 RAM seam `validate_plans(consumed,producer_prepare,data,report_sha)` 检查：
固定四分母/顺序；原 parent_task/parent_selection/blind/physical_prefix digest；
observed_context digest；prefix NPY identity；原 checkpoint5 receipt 完全相等；
prefix raw5；full commands=prefix+suffix；model_plan 自身 digest 和实际全报告 SHA。
不从 logit/tanh 重建命令，不重新排序或优化，不按 native outcome 选根。
四个 modelQ 保存在 `plans[].compiler_plan.model_Q`，只是模型评分，绝不填 native 指标。

只读实际 ABI smoke 已通过4计划，modelQ 原值为
`[0.376455057508769,0.04392555886363612,0.8703004317208628,0.12826448967828571]`。
此 smoke 只读保存 JSON/原证据并调用 RAM validator，未调用本模块 material/prepare，
未运行新 consumer authority 重建、模型或物理；完整 prepare 仍由 MAIN 执行。

## Native 输出与资源合同

`execute` 的执行输入只带原 warmup/prefix/goal receipts，主动移除旧 suffix、
old terminal 和 old_final_metrics。saved 原 loader 为认证 closure 会读取旧完整
report，但不把旧 endpoint 放进新 row，不做旧终点回填或配对收益差值。

actual7 顺序为 `[agent_x,agent_y,object_x,object_y,object_angle,agent_vx,agent_vy]`。
native reward/state distance 使用完整7维；native success 使用前4维位置距离<20
及原 native angle term<pi/9，不含速度，不替换为 object-only 或 wrapped-angle
成功。原 native angle 特殊值如负角项也原样保留；wrapped/object metrics 另作诊断，
`object_success=null`。未知/尚未执行的 state/metrics/native_success 都为 null，
不以0或modelQ补齐。

每次实际观察都保存新 PNG/state/snapshot。warmup0..10 和 plan0..10 完整22帧/支，
plan checkpoint0/5/10 只有实际到达才出现；新 checkpoint10 只来自新 CEM 返回。
native suffix terminated/truncated 立即停，无 padding、retry、补步、换根或额外旧对照。
prefix/warmup 提前停是工程失败，不许可 dispatch suffix。异常保留最后成功观察；
它不是对 native step 内部抛异常时精确故障现场的宣称。

完整固定流程各资源分别上限/预期如下，不相加为“总成本”：

| 资源 | 四支完整数 |
| --- | ---: |
| constructor / close | 各4 |
| reset | 8 |
| public_step | 80（warmup40+prefix20+suffix20） |
| physics_step | 816（16 setter steps+80×10） |
| set_state | 16，仅原 reset 内 |
| goal_setter | 12（8 reset+4 offline bind） |
| goal_render | 8，仅原 reset 内 |
| goal_bind | 4 |
| explicit_render / PNG | 88 |

constructor/close 计 factory/close 实际 callback entry/return。native source 构造器
设置 `space=None`，没有 reset/step/render；reset 才 `_setup`，每 reset 两次
set_state/一次 goal setter/一次 goal render；close 仅处理可选 pygame window。
hooks 从成功 constructor 后覆盖 reset/warmup/prefix/suffix，close 前全部恢复。
构造中抛异常的不可见内部工作为 unknown，不能冒称已被 native hooks 计量。

Meter 独立记 `intents` 与 `counts.{attempted,completed}`，intent 持久化后才进入
callback，completed 在 callback 返回后立即加，不因 return journal 失败退款。
每笔有 op_id，嵌套 resource/branch/step/raw identity、return/failure 收据；
失败 attempt 不退款。durable intent 与实际 callback entry 之间无法对进程被杀
作原子证明；日志缺口严格拒绝 saved 验收，不伪造完成。close 可绕过 wall/storage
predispatch 以清理已构造 env，但仍 cap；cleanup journal 失败也先执行 close，
然后标工程失败。成功构造后即使 return journal 失败，holder 仍确保 close。

固定4 row 永不缩分母；区分 full_suffixes、early_stops、engineering_failures、
not_started，full_suffixes 和最后一步才 stop 可重叠，不能相加当总分母。
所有资源仍逐轴报告，不声称与旧 ridge（65 vs5 predict）等预算，更不称
equal_total_compute/FLOPs。上游 source/planner/physical/RGB 成本继承已付，
本模块不重复记为免费或退款。

wall≤600s 从 run 进入 saved verify 前开始，predispatch 检查；阻塞 native call
不被抢占，close/finalization 可能越限，最终越限则 partial，MAIN 应加外部 watchdog。
总 output≤8MiB，final reserve512KiB，disk floor200MiB；prepare≤512KiB，
独占新输出、fsync、禁止覆盖/删除旧结果。cap 触达后保留 partial，不删除收据放行。

## Python/CLI 接口（以下只有 MAIN 可以实际执行）

`prepare(output,arguments)` 参数严格为 CLI 对应10项 authority 加 wall（见命令；不能缺省
authority）；`verify(prepared,expected)` 完整 saved closure 重建；
`run(prepared,expected,output,factory,authorization=AUTH)` 要求
`AUTH='MAIN_OBSERVED_CEM64_SUFFIX_PHYSICS_ONLY'`；
`consume(output,actual_report_sha256)` 为 saved validator，无新物理/神经。

cwd `/home/gbk/code/LeWorldModel/le-wm-main`，下面所有 authority 均实际已存在，
但新物理 prepare/report 的 SHA 尚不存在，必须使用 MAIN 实际产出的值：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_observed_CEM_physical_20261028_v1.py

MOD=experiments.spectral_reachability.agent_observed_CEM_physical_20261028_v1
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m "$MOD" prepare \
 --compiler-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_baselines_resume_20261027_next_cuda_MAIN01 \
 --compiler-report-sha256 83f9b6fa5ad991022af6bc38063c0fa6c72fca350578692b3c30f4a7e4854a6b \
 --producer-source-sha256 bc8e588bc11a4ab50572ace8a99ef397f05f0ee8b6edfab5ae5a42f022528e10 \
 --consumer-source-sha256 bc8e588bc11a4ab50572ace8a99ef397f05f0ee8b6edfab5ae5a42f022528e10 \
 --consumer-wrapper-source /home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/consume_observed_CEM_MAIN_20261028_v1.py \
 --consumer-wrapper-source-sha256 442a153ca01c525d1315146d4ffeac2debfce7432486c5adb7fb1db171e8c9da \
 --consumer-report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_baselines_resume_20261027_next_consume_MAIN01.json \
 --consumer-report-sha256 b81abd4c2f6600bae69513228a9b6896e2a67f09c3e0ab46fc6901a4034f47d3 \
 --original-physical-report /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_moving_history_physical_20261027_v1_main01/report.json \
 --original-physical-report-sha256 a5322dc4f483e82b5ed9d7b3996cf8a63845261d65774f90a4ae341773016aaf \
 --wall-seconds 600 \
 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_observed_CEM_physical_20261028_v1_prepare_MAIN01.json

.venv/bin/python -B -m "$MOD" verify --prepared /ABS/NEW_PREPARE.json --prepare-sha ACTUAL_PHYSICAL_PREPARE_SHA
.venv/bin/python -B -m "$MOD" run --prepared /ABS/NEW_PREPARE.json --prepare-sha ACTUAL_PHYSICAL_PREPARE_SHA --output /ABS/NEW_PHYSICAL_DIR --authorize-physics
.venv/bin/python -B -m "$MOD" consume --output /ABS/NEW_PHYSICAL_DIR --report-sha ACTUAL_PHYSICAL_REPORT_SHA
```

`report.schema=agent_observed_CEM_physical_20261028_v1.report`，完整工程状态为
`observed_CEM_suffix_physical_recorded_not_admitted`，否则 `partial_or_failed`（CLI2）。
consume 重新验 prepare、hashchain、operation intents/attempts/returns/caps、
固定4身份、原17 paired帧、所有新 PNG byte/pixel seals、native7/goal/reward/stop
chronology、checkpoint和新终点。完好 partial 可输出 `saved_partial_verified_not_admitted`；
damaged/incomplete journals 或不能重建的场景拒绝，不“修复”为通过。
完整 consume 状态 `saved_physical_verified_not_admitted`，也不等于 native success。

RAM 覆盖真实 producer JSON 布局、完整路径、prefix drift、body bits、goal drift、
native stop、失败不退款、constructor/return journal/cleanup、source/report/consume
hash 缺失及错绑、重封链 mutation、PNG byte/pixel/shape、未知 null 与固定4分母。
最终测试数量/三文件 freeze SHA 以交付消息为准。

P1/P4/scientific_admission/independent_confirmation 永远 false。固定 known roots、
新物理工程报告或本模块 consume 不自动开放 independent freshness 或研究准入。
