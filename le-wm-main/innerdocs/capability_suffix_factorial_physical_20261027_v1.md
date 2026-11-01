# 固定24个 v2 suffix 的物理 adapter

本 stem 的 source/tests/innerdocs 独占交付，仅工程实现；不是专职审计或研究goal闭合。
不改 MAIN producer/consumer、旧 helper、共享文档、封存结果；不派生 agent。
实现者没有运行实际 prepare、神经、GPU或物理，无下载/训练/Qfit/CPU神经回退。
采用 experiment-code 的先固定流程、真实输入、失败日志要求；用户限定三文件与RAM验证优先，
不额外创建图表或实验结果。技能引用的两个本地模板文件此前检查缺失，未依赖它们。

已完整读取原 v1 compiler/innerdoc/外层协议及原 probe 源码/innerdoc；收到MAIN更新后
完整读取 v2 compiler、v2 consumer、v2 innerdoc及协议修订。v1已保存的312次调用不删除，
其CUDA/NumPy tanh ULP问题不以放宽门槛修复，本 adapter明确禁止v1物理release。

## 唯一允许的输入与 release 链

固定 producer：`capability_suffix_factorial_cuda_MAIN_20261027_v2`。
固定 saved consumer：`consume_suffix_factorial_MAIN_20261027_v2`。
不导入/接受任意模块路径。MAIN外部提供全部实际SHA，源码不填未来SHA、不搜索最新目录。

prepare的Python接口：

```python
prepare(NEW_OUTPUT_PATH, {
    'compiler_output': ABS_ACTUAL_V2_CUDA_DIRECTORY,
    'compiler_report_sha256': MAIN_ACTUAL_V2_REPORT_SHA,
    'producer_source_sha256': MAIN_FINAL_V2_PRODUCER_SOURCE_SHA,
    'consumer_source_sha256': MAIN_FINAL_V2_CONSUMER_SOURCE_SHA,
    'consumer_report': ABS_MAIN_VERIFIED24_JSON,
    'consumer_report_sha256': MAIN_ACTUAL_VERIFIED24_FILE_SHA,
    'physical_output': ABS_ORIGINAL_EIGHT_ANCHOR_PHYSICAL_DIRECTORY,
    'physical_report_sha256': MAIN_ACTUAL_ORIGINAL8_PHYSICAL_REPORT_SHA,
})
```

路径为canonical absolute且无symlink组件。所有SHA须实际lowercase hex64。
先核v2 producer/consumer source SHA、compiler report SHA与MAIN acceptance JSON SHA，
才导入指定consumer并调用 `consume(compiler_output, compiler_report_sha256)`。
返回值必须与MAIN已经保存的验收JSON完整一致，包括全24 plans。
验收schema=`consume_suffix_factorial_MAIN_20261027_v2.report`、
status=`all24_saved_verified`、branches=24、commands_released=True。
consumer不写输出；重新调用仅saved数组/metadata/SVD校验，无新模型或physics。

raw compiler schema必须为v2.report、all24_plans_sealed、errors=[]，其
commands_released=False是正确的待验收原状态，**既不是失败，也绝不是物理放行依据**。
只有MAIN实际saved consumer通过并提供真实验收文件SHA才闭合；不能单靠改release布尔值。
外部SHA是MAIN调用方的信任锚，不是本模块签发的新证书或密码学MAIN身份认证。

## 实际 v2 ABI与 raw 唯一权威

读取producer.prepare的cases24及report.rows24、forwards312；三条件固定为
open_full/teacher_full/teacher_small，anchor-major，保留原root_slot/condition/arm/task_id/
selection_digest/original_plan/prefix和scale（1/1/.1）。所有条件都执行，不按outcome删根或挑优。

consumer每条plan包含 index/anchor/variant/scale/root_slot/condition/arm/task_id/
selection_digest/original_plan/prefix/suffix/commands/raw_receipt/predicted_Q/winner_index。
逐项join原八计划，包括blind/source/planner SHA，prefix5原float32 bits必须一致。
计划raw_receipt必须等于自身winner.raw_receipt，并与该case实际candidate forward
`13*i+2+winner_index`的raw收据一致；suffix必须与该buffer及独立replay raw逐位一致。
独立replay收据另与result.replay_raw_receipt绑定，不把“收据相同”误当作独立重放。

物理侧绝不执行 `tanh(params)`、参数缩放/clip、重归一化或近似动作重建。
`.1`只在编译条件生成raw时生效；本模块只执行原已保存suffix列表。
MAIN v2consumer负责312 raw→原normalization→真实AE输入的精确bridge。
其NumPy tanh函数诊断容差不被本模块拿来产生或比较执行命令，执行链无容差。

suffix_seal绑定actual compiler report SHA、完整case和winner；execution_sha256再绑定
consumer原plan、case SHA、winner raw receipt和suffix seal。`verify`重建全部saved输入，
不允许prepare后偷换suffix列表。v1任何schema、缺actualhash/receipt/完整24验收均拒绝。

## 确定性重放：不是 setter 近似 restore

`probe.material(original8_directory, report_sha)`仅验证旧八分支saved轨迹/PNG/元数据及源码，
不会执行旧physical8或probe40。每个新case各建fresh native env，共24；相同suffix也不去重。

复用原probe底层place/capture/dispatch/native_hooks/compare_trace：原seed reset → 原placement
reset(seed0) → 原raw warmup10 → 相同源history3/root/goal/goalRGB绑定 → 原prefix5。
原warmup0..10与plan0..5的51次off/on/off render，逐项核PNG字节/pixel SHA、native7、
两body snapshot/float64 bits、forces/torques/sleep、goal_state/goal_pose/goalRGB与renderer。
包括未选中的on-frame，任一不同均在第一个suffix dispatch之前拒绝。

随后连续执行本case自己的sealed suffix5；suffix前无reset/_set_state/逐字段body恢复。
native setter仅原reset使用；它本来就不覆盖所有动力学/solver cache且会额外step，不能冒充restore。
底层snapshot无法独立序列化solver warm-start cache，所以本方案是原路径确定性重构与全部
可导出trace精确核对，不声称隐藏cache字节认证。若要求该更强证明，当前前置不足。

所有控制dispatch with_target=False；只观察时按原off/on/off切renderer，再恢复False。
goal_pose不插入/不改写。native默认outline并不是构造任务goal，不能混淆两者。
goal绑定只沿原evaluator metadata，不产生额外physics或目标驱动的动作选择。

每步保留真实7维state、native reward/terminal/truncated、native_success/native_distance及原
angle语义。agent+object联合任务标签优先，wrapped角/object位姿/native7距离/速度/latent不混合加权。
response另保存相对prefix5的delta_native7与wrapped angle delta，不冒充任务通过。
未知start/final/native labels为null；缺响应是空列表，不填0、不替换为旧终点。

## 输出24分母与实际 meter

report schema为本stem+.report，完整工程status为
`factorial24_suffix_physical_recorded_not_admitted`，失败为partial_or_failed。
rows始终24行，含not_started/engineering_failed/native_stopped/suffix_completed。
source/consumer/compiler/prepare/selected winner seal完整保留；每行有warmup、prefix、suffix
各自attempted/returned，full_suffix与earlystop、engineering failure另列。
suffix实际native stop即停，不补5步，不换根、不重试；失败分支不会从24分母移除。

所有原warmup0..10、prefix0..5及实际suffix1..5均保存off/on/off。
完整分支66PNG，24分支1584。checkpoint0/5来自原root/prefix；checkpoint10来自新suffix第5步，
额外注明phase=suffix、suffix_step=5，不借旧full10。早停缺10保持missing_checkpoints=[10]。
guard失败前尽可能保留原图/invalid incident，不伪造未返回动作的post-state。

| 资源轴 | 固定完整流程最大attempted |
|---|---:|
| constructor / close | 各24 |
| reset | 48 |
| public_step | 480 |
| physics_step（space.step） | 4896 |
| set_state | 96 |
| goal_setter | 72 |
| goal_render | 48 |
| goal_bind | 24 |
| explicit_render | 1584 |

physics=2×reset+10×public_step；set_state=2×reset；goal_setter=reset+goal_bind；
goal_render=reset。这些是不同单位/嵌套层级，不相加成总成本，也不是FLOPs等预算证明。
partial只保留真实actual计数，绝不填固定cap。constructor/close自成资源轴。

meter使用operation_dispatch_intent先持久化意图，再在实际callback入口计attempt、
正常返回即计completed，随后写operation_returned；异常operation_failed不退款。
实际callback返回后journal故障仍保留内存completed，finally据meter对账，不能把未dispatch
的意图当actual attempt。每个receipt有唯一index、resource、branch/anchor/variant/身份和phase/raw_bits。
native hooks原active guards阻止render/goalbind中的隐式physics与suffix setter，finally恢复hooks与with_target。
close即使wall/journal失败也尝试，最多每个成功返回env一次；constructor-return日志失败仍清理holder。
主错误保留first_error，cleanup/restore错误另列，构造中途异常内部工作unknown，不填0。

总outputcap192MiB（含PNG/journal/partial/final），最终保留8MiB；prepare≤16MiB，floor200MiB。
wall900s从run验证前开始，dispatch前检查；不能抢占阻塞native call，cleanup/final保留允许越界，
但仍不突破output/floor。final重新核全部输入seal并做actual census；超wall不称完整成功。

## MAIN CLI（本worker不执行）

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`；NEW目录/prepare路径必须不存在。

```bash
.venv/bin/python -B -m experiments.spectral_reachability.capability_suffix_factorial_physical_20261027_v1 prepare --compiler-output ACTUAL_V2_CUDA_DIR --compiler-report-sha ACTUAL_V2_REPORT_SHA --producer-source-sha MAIN_FINAL_V2_PRODUCER_SHA --consumer-source-sha MAIN_FINAL_V2_CONSUMER_SHA --consumer-report MAIN_VERIFIED24_JSON --consumer-report-sha ACTUAL_VERIFIED24_FILE_SHA --physical-output ORIGINAL8_PHYSICAL_DIR --physical-report-sha ORIGINAL8_ACTUAL_REPORT_SHA --output NEW_PREPARE_JSON
.venv/bin/python -B -m experiments.spectral_reachability.capability_suffix_factorial_physical_20261027_v1 verify --prepared PREPARE_JSON --prepare-sha MAIN_ACTUAL_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.capability_suffix_factorial_physical_20261027_v1 run --prepared PREPARE_JSON --prepare-sha MAIN_ACTUAL_PREPARE_SHA --output NEW_PHYSICAL_DIR --authorize-physics
.venv/bin/python -B -m experiments.spectral_reachability.capability_suffix_factorial_physical_20261027_v1 consume --output ACTUAL_PHYSICAL_DIR --report-sha MAIN_ACTUAL_PHYSICAL_REPORT_SHA
```

`consume(output, external_report_sha)`为saved-only validator：核binding、全prepare closure、
actual journal hashchain、nested intent/return、24身份、51起点、全部PNG bytes/pixels及native7。
保留durably verifiable partial原24行；损坏/缺terminal或缺actual-return证据的partial拒绝consume。
`validate_verified`与`validate_saved`是RAM seam，不替代正式external SHA/consumer/PNG gates。

## 验证与前置状态

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_capability_suffix_factorial_physical_20261027_v1.py
```

测试全为显式RAM mock/arrays/PNG与真实hashchain，不调用原probe40/physical8 execute，
不初始化模拟器或模型，不产生真实prepare/新结果。测试不是科学或真实恢复成功证据。
MAIN actual v2 CUDA report、实际consumer通过与外部SHA均由MAIN完成；本实现不伪造这些前置。
即使全部24物理成功，仍是受旧校准启发的known roots探索，不自动开放existing P1或independent freshness，
scientific_admission恒false，不声称task-winning/等FLOPs/全局优势。

2026-10-27最终RAM验收：adapter41项 + MAIN v2 core11项，共52 passed（31.25s）。
此前首轮38项通过后，按实际v2源码补强plan.raw_receipt/312forwards/独立replay精确链；
最终52项对应当前源/tests版本。执行了CLI --help，没有执行material/prepare/run/consume真实实验。

MAIN实际通过的v2输入已只读核对文件SHA，非本worker自签/自运行：

- producer源码SHA：`c90d7423eee76cc0b25448d741007efab6d6de1b26f1159b3cbf01bad806f10e`。
- consumer源码SHA：`acfc20ca985f6cce726ff54258d6281a8bbf1b90ad19b191bed3178b44328624`。
- `results/capability_suffix_factorial_cuda_MAIN_20261027_v2_MAIN01/report.json`：
  `fc620c9fdcbb5046c1d076c79cd0703da999275ec921de799f3747c0204d69ab`。
- `results/consume_suffix_factorial_MAIN_20261027_v2_MAIN01.json`：
  `3b17d6b6a7323545d0fb81034dac5e0c1d479b5a76de307bb2ca1a1ae1d8ad58`。
- producer实际prepare SHA：`330412c675844d7edff23ade45012cf4ef4920f64bd47a31a0c49e6872f673f5`，
  由compiler report中的prepared/prepare_sha256闭合，不冒充physical prepare SHA。
- 原八锚点 `results/p1_moving_overlay_recompile_physical_20261027_v1_MAIN01/report.json`：
  `56962e9370fcf249a2ed88f004a400414b491f503a4c3a9db1bd951cf984a521`。

MAIN可直接使用以下实际输入命令（本worker未执行，NEW_PREPARE_JSON须选择不存在的路径）：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.capability_suffix_factorial_physical_20261027_v1 prepare \
  --compiler-output /home/gbk/code/LeWorldModel/le-wm-main/results/capability_suffix_factorial_cuda_MAIN_20261027_v2_MAIN01 \
  --compiler-report-sha fc620c9fdcbb5046c1d076c79cd0703da999275ec921de799f3747c0204d69ab \
  --producer-source-sha c90d7423eee76cc0b25448d741007efab6d6de1b26f1159b3cbf01bad806f10e \
  --consumer-source-sha acfc20ca985f6cce726ff54258d6281a8bbf1b90ad19b191bed3178b44328624 \
  --consumer-report /home/gbk/code/LeWorldModel/le-wm-main/results/consume_suffix_factorial_MAIN_20261027_v2_MAIN01.json \
  --consumer-report-sha 3b17d6b6a7323545d0fb81034dac5e0c1d479b5a76de307bb2ca1a1ae1d8ad58 \
  --physical-output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_overlay_recompile_physical_20261027_v1_MAIN01 \
  --physical-report-sha 56962e9370fcf249a2ed88f004a400414b491f503a4c3a9db1bd951cf984a521 \
  --output NEW_PREPARE_JSON
```

三文件冻结SHA由交付消息记录。MAIN仍须完整读码、实际prepare/verify及主机物理执行；
本worker没有验证真实模拟器重放成功，不捏造科学通过。
