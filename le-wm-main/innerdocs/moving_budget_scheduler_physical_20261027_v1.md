# Moving budget scheduler 固定八分支 physical v1

状态：MAIN已转交Poincare最终冻结接口及三SHA，authority已固定，完整recipe/计量/guard/durable partial/MAIN CLI实现完成。仅RAM/mock验证，不等于actual物理。MAIN审读后可prepare，实际run还需完整scheduler actual report外部SHA及明确physics授权；worker没有prepare/物理/神经授权，不另写独立auditor。

最终scheduler三SHA，ownconsume在import前核对并加入bindings：source `f4f4d5e013e3fc3aac896da58faaafadcf5e2d4e53f5f2f7b61203707e09ff9d`；tests `58f2acf38164dff6699948f1c2c6d5f9b4b5391e03984b4b60a6142ac17db0c0`；doc `303405900b416475956f1ecb74b08e4c8b6e20394df5693807234371c995cac1`。任一漂移均拒绝，不重新封旧producer。

## 固定协议及输入接口

依据外层 `docs/D2P_MOVING_BUDGET_SCHEDULER_PREREG_20261027.md` 完整协议，SHA `8e8cd4091dbaed5991dc119638ffdc00c8aed0852d0ed4c54cc6b1b3a82009bf`。固定已用moving两根，不是独立确认，不以新模型Q/pose/RGB表现筛根或删策略。本consumer不计算配额、不改scheduler候选、不拟合Q、不重规划、不读取旧physical outcomes。使用experiment-code的预算预声明/日志/故障测试模式；P1/P2/P3 admission全部false。

唯一producer名字：`moving_history_budget_scheduler_cuda_20261027_v1`。接口已由MAIN/Poincare冻结确认。own `consume(output,report_sha)`必须返回：

```text
schema: moving_history_budget_scheduler_cuda_20261027_v1.consume
status: four_policy_commands_verified_not_admitted
errors: []
plans: ALL8 model_plan（顺序可由稳定身份join，不按Q排序）
P1_admission/P3_admission/independent_confirmation/compute_advantage/walltime_advantage: false
```

每model_plan必须有producer自己的`.model_plan` schema、source_slot（严格int0/1）、root_sha256、task_id、blind_sha256、task_export、task_sha256、task_contract_sha256、bank_sha256、arm、commands、selection_sha256、whole_run_report_sha256、physical_execution=false、selection_uses_physics=false及原FLAGS；允许producer原有candidate_index/model_cost/anchors等字段，原样计入其selection seal。命令必须是原封存winner的10×2 float32-exact raw，在[-1,1]，不clip/round/atanh/二次100。ownconsumer已独立核候选/选择/whole+segmented replay；physical不重新选择。完整artifact/prepare/model/source closure绑定，不能拿旧GN/D2P锚点或handoff代替新scheduler report。

source.digest canonical是普通`json.dumps(sort_keys=True,allow_nan=False)`再SHA256，须与最终producer的`core.digest`一致；不是encoder compact JSON加newline。每plan的selection_sha在去掉selection_sha256/whole_run_report_sha256后校验。与producer prepare slots再核root/task/blind/export/task_contract/bank身份；无位置猜测。

physical manifest exact字段与前一consumer同构，但schema和authority独立：

```json
{
  "schema": "moving_budget_scheduler_physical_20261027_v1.selection",
  "sourcepack": "/home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_moving_history_source_20261027_v1_main01/sourcepack.json",
  "sourcepack_sha256": "c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6",
  "source_prepared": "/home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_moving_history_source_20261027_v1_prepare01.json",
  "source_prepare_sha256": "c1292f0ebfb1827e5da92b32ab13d2e4a12c025c4741277ab4c4f478d6ee8cb9",
  "roots": [
    {"root_slot": 0, "root_sha256": "ORIGINAL_MOVING_ROOT0_SHA"},
    {"root_slot": 1, "root_sha256": "ORIGINAL_MOVING_ROOT1_SHA"}
  ],
  "arms": ["U_uniform", "D_geometry", "S_swapped", "N_no_signal_uniform"],
  "authorities": [{
    "authority": "moving_history_budget_scheduler_cuda_20261027_v1",
    "output": "/ABS/NEW_SCHEDULER_ACTUAL_OUTPUT",
    "report_sha256": "MAIN_ACTUAL_COMPLETE_REPORT_SHA"
  }],
  "selection_fixed_before_physics": true,
  "physical_outcome_selection": false,
  "wall_seconds": 600
}
```

root/output/report占位由MAIN按实际source/未来scheduler填写，禁止伪造SHA；一旦manifest封存不能据收益换策略。严格四策略完整顺序，不允许单策略或部分根。物理固定root-major `(0,U/D/S/N),(1,U/D/S/N)`，scheduler method-major输出通过身份join，不是选有利顺序。即使U=N或其他策略命令相同，也分别实际执行8分支，不去重/退款。

## 资源与执行

复用已冻结 `p1_constructive_moving_history_physical_20261027_v1.branch/native_hooks`，以及extended consumer的纯recipe检查、close和Store primitive；不调用旧整个physical runner，不修改其globals或已封存source。helper source SHA `a0f7348ebe8c6e061b5d63a1ba4a8ecb064a26e7f7a1ef4be57fe8b07270d799`。

每分支真实重建原seed/placement，原两reset，真实raw warmup前5`[.1,0]`后5`[0,.1]`。核warmup0/5/10全RGB/state/两body动态/f64bits/rawhistory精确一致，核root+goal，再无physics/setter/render副作用地attach原goal。原target-free RGB224/render_actionFalse/relative/action_scale100/native eval保持；无执行时神经反馈。所有capture先保存PNG再RGB/canvas/state/pure-render guard；无放宽颜色或缺帧替代。模型输入历史/目标和动作witness来自scheduler ownreceipt + 原moving encoder public loader，禁止误接zero-only消费者。未序列化solver warmstart，不声称全隐藏缓存克隆。

固定完整上限：reset16、set_state32、goal_render16、goal_bind8、public_step160（warmup80+selected80）、physics_step1632（reset32+public1600）、explicit_render/PNG176、construct8/close8。所有逐dispatch先attempt持久化，再实际call/return日志；计量包含reset内部space.step，goal_bind期间禁止step/render。native terminated/truncated/guard可提前停止，不pad不补采样；完整八分母含not_started/failed/native_stopped，不把partial终点混full10。异常不退款或自动重试。

constructor先以原nativefile固定SHA/AST保证无reset/step/render，再核返回后space/render_buffer/latest_action未启动。hooks从首次reset起覆盖。constructor半途失败机器字段`row.constructor_internal_work="unknown"`，report `total_internal_physics_scope="unknown_constructor_work"`；hook计数0不等于constructor总工作0。成功pristine为`sealed_native_pristine_no_physics`；未触发槽为`not_attempted`。returned env即使completion日志失败仍close；close即使wall/journal失效也尝试，首错不被close/finalization覆盖。

NEW run总cap16MiB，含PNG/partial/journal/final，final reserve512KiB，floor200MiB（需216MiB headroom）；prepare≤256KiB。wall≤600s是dispatch前守卫，不是可中断一个阻塞nativecall的硬实时保证。新neural/image encoding/training=0。上游scheduler520predict/sharedJ/SVD/原编码/锚点成本由其ownreport单独完整绑定，不能说物理成本包含其中或等总算力。RAM容量fixture不是实际产物容量保证，cap不足应失败而不是裁数组或调额。

## 输出与MAIN CLI

prepare包含`schema,manifest,manifest_sha256,plans,contract,bindings,sourcepack_sha256,source_prepare_id`。physical plan保留root_slot/root_sha256/arm/rawcommands/task_id/blind/export/task_contract、compiler authority/report SHA、compiler selection SHA、own_plan SHA、selection_digest。

report schema `moving_budget_scheduler_physical_20261027_v1.report`；成功status为`moving_budget_scheduler_physical_recorded`，失败`partial_or_failed`。包括prepared/prepare_sha256、errors、rows、fixed_branch_denominator=8、counts、journal_sha256/journal_chain_tail、source身份及科学flags false。每row为固定root/arm/selection_digest/selected_plan、initial_exact/full_horizon/own_attempted/own_returned、first_error/secondary_errors、endpoint_scope及checkpoints；constructor unknown单列。成功仅表示工程记录完成，允许合法native早停，并非八次任务成功。

checkpoint沿用exact旧schema：`phase,step,png,file_sha256,pixel_sha256,state,snapshot,observation_state`。完整plan0/5/10，所有0..10新PNG也保存；失败缺失5/10不得补值。后续NEW RGB producer可按所有八分支的5/10读取，但本任务不做encoding。

以下为MAIN可用CLI；authority已固定，manifest须填写未来完整scheduler actual report SHA和原root SHA，MAIN复核后prepare/verify；实际物理仍须MAIN明确授权：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_moving_budget_scheduler_physical_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.moving_budget_scheduler_physical_20261027_v1 prepare --manifest /ABS/selection.json --manifest-sha MANIFEST_SHA --output /ABS/new_prepare01.json
.venv/bin/python -B -m experiments.spectral_reachability.moving_budget_scheduler_physical_20261027_v1 verify --prepared /ABS/new_prepare01.json --prepare-sha PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.moving_budget_scheduler_physical_20261027_v1 run --prepared /ABS/new_prepare01.json --prepare-sha PREPARE_SHA --output /ABS/new_physical_main01 --authorize-physics
```

只读saved/modelconsumer验证并不构成独立物理证据；RAM fakeenv只能证明实现合同。两已用根、静态跨根调度均不能替代P1/P3前置或完整C3。

## 针对性回归与冻结交接

新增32个定点fault cases：对八个固定分支位置各自注入constructor半途异常、close异常、首个selected step异常、合法truncation。均核全部八identity/branch_closed分母、其余七分支完整执行、raw/资源attempt-return不退款、不以失败重排策略、不补缺checkpoint。close失败与物理full10分列，不能将close异常抹去真实已执行终态；constructor失败的hook零计数与unknown内部工作量分列。只使用RAM数组和fakeenv，不创建test tmp或实际prepare/results。

最终交接由MAIN转交，未声称当前环境直接联系Poincare。新增三类source/tests/doc漂移拒绝及真实冻结producer `model_plan/core.digest`兼容测试：`whole_run_report_sha256`明确排除在selection seal之外，consumer不把追加report字段当原plan内容。仅导入producer pure数学函数构造RAM plan，不调用其material/consume/run或实际任务。goal API仍被旧blocked未完成目标占用，未伪造其完成或科学P1完成。
