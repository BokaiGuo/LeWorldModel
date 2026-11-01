# Constructive blind-plan physical consumer v1

本文件与同 stem source/test 独占新增；不修改 constructive source、compiler 或旧证据。
本 consumer 只产生校准物理比较，不证明 P1、plateau、控制器效能或独立样本数增加。
作者仅运行数组/mock/已存档文件只读测试；实际 physics 由 MAIN 授权执行。

## 设计与权限

固定已审计 sourcepack SHA
`9ad3a414fd2148258524d4a7e9d4f070f59532ec2e5943cfa61f9145b3a41bd9`，两根顺序不变。
NEW selection manifest 在物理执行前冻结，固定 1--4 个 arm 的顺序、两根完整笛卡尔积与每臂自己的十条 raw commands。
不得从 native outcome/证书 witness 选择控制器；不得在失败后补根、重试命令或改顺序。
`compiler_report_SHA` 是外部实际报告文件 SHA；JSON Pointer 必须逐字绑定其 source_slot、arm、commands、blind payload SHA。
控制侧不需要 native rootsha；evaluator 使用 manifest.root_slot = compiler.source_slot 映射原始两根。
root/goal/state/certificate 仅离线重建与评价可见，不传给候选生成器。

证书 NPZ 只读取 `goal,goal_rgb,observed_states,root_history`；禁止读取
`witness_commands,witness_logits,future_states`，不调用 builder.episode/witness。
sourcepack 完整性检查复用已封存 builder 的读取工具，不是独立科学复审。
报告字段名/完成状态由未来 MAIN 显式给定，consumer 不宣称从字符串或 SHA 证明 compiler 数学、盲性或选择算法正确；
MAIN 仍须审核未来 compiler schema/选择过程。预声明不能从文件本身证明真实时间顺序，MAIN 管理先封存后执行。

## 每臂物理协议

1. 原 reset_seed reset；原 sample_state(root_seed,proposal=0,attempt=0) 生成 requested state。
2. 原 placement seed=0 reset(requested, original_goal)，不是用 saved poststate 作为 reset 输入。
3. 十个 raw zero warmup：这是真实新增 public steps，不是 padding、不称零物理。
4. 0/5/10 RGB、state、block dynamics 与 blind/source 原始三帧逐元素精确相等，否则不执行 plan。
5. 原生类源码/参数认证，with_target=False，render_action=False，224 RGB；原生 `_set_goal_state` 与缓存 `_goal` 附加构造 goal/真实 goalRGB，
   检查无 setter/reset/render/physics 与动力学变化；目标附加后根图像必须仍相等。
6. 只执行该臂预选 own raw10×2 float32 commands，逐步 fresh RGB。不得 clip/round/pad。
   原生 terminal/truncation 即停止，不强凑十步，不借 witness 接着走。

每张有效 RGB **先落盘 fsync PNG，然后 durable fresh_observation(state/snapshot)，然后 durable guard_receipt**；
guard 检查 agent 像素、画布位置、returned state、render 无动力学副作用；失败后该臂不得再 dispatch。
失败行 `last_observation` 直接指向失现场。非法 RGB shape/dtype 尽可能保存 raw NPZ、state/snapshot、invalid_render 与拒绝 guard_receipt 后停止。
文件写入失败本身不能保证失现场已持久化，必须保留 first_error，不能称完整。
每步 returned_transition 保存实际 action、state/reward/native flags/goal 与 independent native/wrapped/object metrics。
物理步由实际 `space.step` hook 计账（包括 reset 新建 space）；attempt 与 completed 分开，失败不退款。

原 native angle `min(abs(delta),2pi-abs(delta))` 缺 mod 可出现负角误判。
本 consumer 不修环境、不放行原生 terminal，warmup 仍服从 native；日志与评价并列 native/wrapped discrepancy。
object_position_error / wrapped_object_angle_error / object_pose 与变化量是数值统计，`object_success=null`（未擅定科学阈值）。
终态 success 必须是原 native actual eval，reward=-distance 与 terminated=success 精确核对。

## 完整分母、预算与输出

每臂上限：2 reset、20 public step（10warmup+10own）、22 explicit render、2 reset 内 goal render、
4 `_set_state`、204 actual physics step、1 goal bind。
N=2×arms；全部行在第一操作前创建，未执行保留 not_started，partial CLI exit2。
完整两 arm 示例为 4 臂：8reset/80public/88explicit/8goalrender/16setter/816physics/4goalbind。
每完整臂保存 warmup0..10 与 plan0..10 共22PNG（覆盖0/5/10以及所有 guard 所需失现场）。
终止提前则实际计账，不伪称全 horizon。all cap 为前置 gate；独占 NEW output，8MiB含所有 partial 文件，保留256KiB final reserve；
启动需8MiB+200MiB，逐写/操作继续检查200MiB floor。墙钟最大600秒。
每臂 first/secondary cleanup 错误保留；final bindings、census、journal、hash、report 分别尽力执行；无隐式重启恢复。

## NEW selection manifest

顶层必须恰有：
`schema,sourcepack,sourcepack_sha256,source_prepared,source_prepare_sha256,roots,arms,plans,selection_fixed_before_physics,physical_outcome_selection,wall_seconds`。
schema = `p1_constructive_plan_physical_20261027_v1.selection`；最后两个声明分别 true/false。
roots 是 `[{'root_slot':0,'root_sha256':实际值},{'root_slot':1,'root_sha256':实际值}]`。
plans 按 root-major × arms 原顺序，每行恰有：

```text
root_slot, arm, commands (exact float32-representable raw10x2),
compiler_report (absolute path), compiler_report_SHA,
compiler_schema, compiler_status,
commands_pointer, source_slot_pointer, blind_sha_pointer, arm_pointer,
selection_digest
```

例如 pointers `/rows/0/selected/raw_commands`, `/rows/0/source_slot`, `/rows/0/blind_sha256`, `/rows/0/selected/arm`；
只是格式示例，不假定未来实际 compiler 字段。
selection_digest = `source.digest(row excluding selection_digest)`；不是 native outcome digest。
源 prepare 原文件 SHA `98ffb806018a270f5ef27ffa76be929393a74355c213bc274bf158c72d80b086`。
prepare 只存小型绑定/commands/hash manifest，不复制 sourcepack/模型/大闭包；run 再验全部 SHA。

## MAIN 最小命令

在 `/home/gbk/code/LeWorldModel/le-wm-main` 下，MAIN 先写并审核实际 selection manifest，然后外部计算其 SHA：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_plan_physical_20261027_v1.py
sha256sum /ABS/selection.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_plan_physical_20261027_v1 prepare --manifest /ABS/selection.json --manifest-sha MANIFEST_SHA --output /ABS/NEW_prepare01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_plan_physical_20261027_v1 verify --prepared /ABS/NEW_prepare01.json --prepare-sha PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_plan_physical_20261027_v1 run --prepared /ABS/NEW_prepare01.json --prepare-sha PREPARE_SHA --output /ABS/NEW_exclusive_output --authorize-physics
```

最后一条运行真实物理，仅 MAIN。当前无未来 compiler 实际 manifest、未 prepare/run，不能宣称接口已通过真实 compiler 或物理验证。
mock 模拟器只用于程序/故障合同；已存 source 只读测试不运行模拟器，不构成独立可达性或科学准入证明。

本轮最小测试命令实际完成：40 passed。全部 fixture 为内存数组，无 pytest tmp/prepare/result 产物；
其中一项只读加载上述真实 sourcepack、原 prepare 与白名单 bridge 字段；这不等于执行或独立复审真实物理。
失败帧顺序、全分母、actual hook 计账、原生 terminal、无 witness 调用、source_slot/blind SHA/own command 绑定、
close/branch journal/final journal/hash/report 错误，以及磁盘 gate 均有专门 mock 检查。
