# Moving-history 固定 GN/D2P plan 的同 body 轮廓配对

本 stem 是新的物理 producer，只实现 MAIN 已批准的固定回放；不是代码审计，
不是新 planner，不是 P1 准入。旧 zero-history overlay 不属于本实验。
不修改旧 source、runner、模型、结果或共享协议文档。

## 固定输入与权限

- 原 moving physical report：
  `/home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_moving_history_physical_20261027_v1_main01/report.json`，
  SHA `a5322dc4f483e82b5ed9d7b3996cf8a63845261d65774f90a4ae341773016aaf`。
- 原 physical prepare SHA：
  `226749e2bf8acdcd7a4a4a08be7336445daad0b6ad12185b24cbc0fb4560b1ff`。
- 原 moving sourcepack SHA：
  `c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6`。
- 原 moving physical source SHA：
  `44d90599eab881c773d5efa09177dfe6b395fa3f4eef41070340681fea5ec96e`。

控制输入只取原 prepare 的固定 raw float32 full10 commands：root-major 两根 ×
`line_search_GN` / `full10_conditional_D2P`。不从物理结果选根、选臂、改命令、
调阈值或重试。旧 source certificate 不是候选控制器；不调用 witness/episode，
不解码证书动作。只复用 saved source/selection 验证与 schema-independent helpers，
本 stem 自写 branch/execute/native hooks，不 monkeypatch 旧模块全局。

遵从 `/home/gbk/code/LeWorldModel/docs/D2P_MOVING_TARGET_OVERLAY_PROTOCOL_20261027.md`。
原 seed → sample_state recipe → placement reset → moving raw warmup10 重建，
完整 RGB0/5/10、状态与两 body snapshot 必须和原 blind 数据一致。
goal attachment 只绑定原构造 goal_state 与原真实 off goalRGB，不产生物理或 render，
不修改 goal_pose。goal_pose 的默认原生轮廓不是构造 task 的真实目标。

## 计账与停机

完整执行上限：4固定臂、8 reset、80 public steps（40 warmup + 40 plan）、
816 实际 `space.step`、16 state setter、8 reset 内部 goal render、4 goal bind。
88 原 off 观测全部保存并按文件SHA/像素SHA/state/snapshot/observation_state 对旧封存。
plan0/5/10 各复用已有首张 off，额外 render on 和 off：
`88 + 12 * 2 = 112` explicit renders/PNG，绝不是124。
每臂28PNG，其中22原off与6额外配对；无神经/GPU/训练/拟合/动作重新选择。

每次实际操作前 durable attempted、返回后 completed；内部 physics 用真实 hook，
不把 10×publicstep 的公式伪装成实际hook。公式只在最终 census 检查。
reset 内部 goal render 与显式 render 分开。每次写文件与dispatch检查8MiB总cap、
200MiB floor、300秒wall cap；预留256KiB给最终报告。独占新目录不覆盖。

native success、terminated/truncated 均遵守原环境；原生角度判据的未取模问题不改动，
同时保存 low.metric 的 wrapped/object-only诊断，不用这些诊断选择动作或覆盖native。
full10、native early-stop、engineering failure 分开；early-stop只减少实际操作计数，
科学分母始终4。缺checkpoint明确保留，消费者不得补图、补动作或少算分母。
未完成 warmup 不执行 plan。没有替补臂。

每个可返回的实际图像先保存，再写 observation/guard receipt，最后才验收，
包括on造成的状态/goal/renderer变动或无蓝色agent的失败现场。
on期间仅临时切换 with_target，finally恢复False；所有控制步与reset均False。
内部 step/render 抛异常时，不能保证已有故障后frame；last_observation只是最后一次
实际完成显式渲染，绝不能冒称故障现场。异常计数保留 attempted/completed差值。
首错、secondary cleanup错分别保留；journal terminal/hash/report各尽力独立封存。

## 新 schema / consumer 接口

prepare：`p1_moving_target_overlay_plans_physical_20261027_v1.prepare`。
report：`p1_moving_target_overlay_plans_physical_20261027_v1.report`；正常记录status
`moving_target_overlay_plans_recorded`（仍可能有合法缺checkpoint），工程失败
`partial_or_failed`，CLI非零退出。

report.rows 永远预置4行。关联键 `(root_slot, arm, selection_digest)`，不是source_slot
或猜测行号。每行 `checkpoints` 为已尝试的plan0/5/10配对；
`missing_checkpoints`保留未完成triplet。每个checkpoint：

```text
{step, off_before: receipt, on: receipt|null, off_after: receipt|null, complete: bool}
receipt = {phase,step,mode,with_target,png,file_sha256,pixel_sha256,
           state,snapshot,observation_state,before,after,old_off_match,...}
```

`off_before`引用同一次既有off观测的收据，不新render。`before/after`包含完整导出的
两body snapshot（含float64 bits）、goal_state/bits、原goal_pose/bits、off goalRGB SHA、
renderer配置/variations。snapshot不声称覆盖不可导出的solver warm-start缓存。
`old_off_match`仅off有，on为null。所有PNG路径相对新output，消费者必须核report、
execution_binding、prepare、完整journal链及PNG SHA，并保持完整4臂分母。

journal事件含 run_started、branch_started（固定命令）、operation_attempted/completed/
failed、placement_request、returned_transition（实际raw及bits）、fresh_observation、
guard_receipt、goal_attached、overlay_triplet_completed、overlay_restored、branch_closed、
run_finished。报告含实际operation ledger、原report/sourcepack SHA、固定分母和无准入声明。

当前不导入 Arendt 新source execute，也不要求它已就绪。本producer只提供同body的
native on plan图；`source_goal_on_certified=False`，`source_goal_on_authority`明确要求
下游独立外部SHA桥接 Arendt witness-terminal 的真实goal-on证据与同condition history。
本报告不能自行生成/认证goal-on，也不允许physical setter补造goal图。

## MAIN 最小命令

以下仅供MAIN完整读三文件、通过测试并批准后执行；实现agent未运行prepare或物理。
从 `/home/gbk/code/LeWorldModel/le-wm-main` 运行：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_moving_target_overlay_plans_physical_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_target_overlay_plans_physical_20261027_v1 prepare --output results/p1_moving_target_overlay_plans_physical_20261027_v1_prepare01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_target_overlay_plans_physical_20261027_v1 verify --prepared results/p1_moving_target_overlay_plans_physical_20261027_v1_prepare01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_target_overlay_plans_physical_20261027_v1 run --prepared results/p1_moving_target_overlay_plans_physical_20261027_v1_prepare01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/p1_moving_target_overlay_plans_physical_20261027_v1_main01 --authorize-physics
```

将ACTUAL_PREPARE_SHA替换为真实prepare返回值，绝不预造。run前必须满足200MiB+8MiB
空间门，目录不得存在。测试仅RAM数组mock，不创建tmp/results，不导入真实模拟器或模型。
mock通过不等于真实回放成功，更不代表已知根变成独立新样本或P1通过。

实现交付验证：独占测试文件47项RAM mock通过（6.19秒），不创建tmp/results。
覆盖112完整计账（非124）、88off的逐类错配、on失败现场先保存/恢复显示开关、
禁止render内部physics、原动作替换、合法early-stop保持四臂、step内部异常不伪造
故障后frame、最终journal/hash/report分别失败及首错保留。不是实际物理执行记录。
即使新paired source已经由MAIN执行完成，也不反向引入本producer的sourceclosure；
其on目标认证由下游独立外部SHA桥接，旧source/goal off保持不变。
