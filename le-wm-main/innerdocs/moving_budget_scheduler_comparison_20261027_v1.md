# Moving budget scheduler 四策略 saved comparison v1

这是独立的**实现任务**，不是另派代码审计或神经/物理任务。只新增本stem source/tests/本文，不改scheduler、Meitner consumer、旧结果或共享账。新实现goal只涵盖本实现与mock验证，不代表P1/P3科学准入。

## 输入接口与协调

完整读取Meitner `innerdocs/moving_budget_scheduler_physical_20261027_v1.md` 及实现后对接；目前没有可用agent消息工具，由MAIN转交接口。MAIN现已提供真实scheduler/physical终态SHA，comparison只hash已实际封存的physical源码，不导入/执行其模块。下面实际验证不包含任何新模型或物理执行。

唯一planner authority是已冻结 `moving_history_budget_scheduler_cuda_20261027_v1.py`，SHA `f4f4d5e013e3fc3aac896da58faaafadcf5e2d4e53f5f2f7b61203707e09ff9d`。使用其自有 `consume(output,report_sha)`，必须返回完整 `.consume` schema、status `four_policy_commands_verified_not_admitted`、errors空和固定8plan。该入口是saved-only，复核完整candidate/replay/history/ledger，不运行模型。

Meitner接口固定为 `moving_budget_scheduler_physical_20261027_v1.report`，status `moving_budget_scheduler_physical_recorded`、errors空、fixed_branch_denominator=8；report内绑定prepared/prepare_sha256、counts、journal_sha256/journal_chain_tail、sourcepack/source_prepare_id。prepare为独立`.prepare`，其中manifest封存原8plan/authority/source closure；外部 `--physical-source-sha` 必须等于该真实prepare中producer source binding。合法native earlystop可记录，但不能冒称full10。partial/failed report拒绝发布完整comparison，不把工程失败当科学负结果。

每row读取 `selected_plan`、`selection_digest`、`root_slot/arm/branch_index`、initial_exact、own_attempted/own_returned、goal/root_state/final_state、initial_metrics/final_metrics、checkpoints、终止与错误字段。稳定join键是 `(slot,root_sha256,task_id,blind_sha256,arm)`，不是行号或模型成本排序；method-major planner与root-major physical可重排输入，仍join相同8格。原model selection seal、physical selection_digest、own_plan_sha、compiler report SHA、literal float32 raw10均绑定。不得借用GN/D2P历史endpoint。

实际journal纯saved解析：全链index/hash/tail、execution_binding、run_started prepare、branch_started冻结命令、每次operation attempt/return、返回state/goal/reward/terminal、fresh RGBguard/checkpoint和branch_closed。实际producer的branch_closed无外层branch字段，按receipt.branch_index和完整receipt身份join，不猜顺序；此实际接口已有RAM回归。仅hash已保存PNG，不调用render/encoder；不复制大资产。固定全8分母，warmup/selected public step和reset/warmup/selected内部physics分别记账，构造内部工作必须由producer确认为pristine，unknown不冒充0。

## 输出与指标边界

新schema `moving_budget_scheduler_comparison_20261027_v1.report`，status `saved_comparison_assembled_not_admitted`。每策略/根输出选中model_Q、candidate_index、allocated_blocks、已评价H2 candidates、exclusive forward/AD/SVD ledger及实际physical资源/终态。双根均值和逐根 D−U（主）、D−S/D−N（次）仅为描述性差异；没有置信区间或独立任务确认。

native指标按冻结kernel精确复算：native_distance是raw7维joint-state L2，reward为负距离；native_success是前四xy坐标联合norm<20与原native角表达式<pi/9，**没有速度成功条件**。后两state坐标会影响native reward/distance，不据此另加success门槛。native角表达式多圈时可负，原样报告其negative/disagreement；另列独立wrapped angle、object position误差、wrapped joint-state L2与wrapped_allxy_success，不改环境、目标或阈值。2pi表示差不冒充几何错位。所有原始goal/initial/final state7保留。

model Q不等于真实pose、reward、native success或真实改善。每行RGB栏固定 `status=missing_not_encoded,actual_same_Q=null`；没有同条件actual192/goal编码就不计算actual Q或模型/真实Q排序，不用pose代理填空。本v1没有RGB输入适配器；将来接新编码receipt应另开版本/显式协议，不能看到结果后悄悄补值。

完整8full10时可报告fresh共同H2终态描述；若合法earlystop则保留全部格，说明native guarded endpoints长度不同，共同full10配对摘要不可用。0/8不能推出理论不可能，更不能消除模型/代理/协议等解释。

## 成本账与不可宣称项

复制引用真实scheduler actual520 predict/AE/predictor/pred_proj，shared8只计一次；2J、384数学方向/显式grad与backward hook、2SVD全保留。原8/8视觉acquisition与GN/D2P历史288predict及原完整receipt单列，不消账。U/D/S独立部署逻辑136、N128不是新增实际调用，也不是等全资源向量；每根block分配不同，不能说逐根等候选预算。原合法域与各方法实际评价的池前缀亦应区分，不能拿H3或其他动作域混表声称匹配。

物理actual full上限：8分支、16resets、80warmup+80selected public steps、32 reset内部+800warmup+800selected=1632physics steps、176显式render、32set_state、16goalrender、8goalbind、8construct/close；合法早停使用真实更低计数、不pad。comparison本身new_neural_calls/new_physics_calls恒0；不把saved读取算成新模型预测。wall/内存引用原实际receipt，不从次数估算FLOPs或速度收益。

所有结果 P1/P2/P3_admission、independent_confirmation、full_C3_demonstrated、compute_advantage、walltime_advantage 均false。这是已用moving2根的静态跨根调度描述，不能替代原C3/P3任务内重配置/时域调度/真实质量成本曲线及P1/适用P2前置。

## 执行与限额

测试只用RAM/NumPy/JSON，无torch模型、真实AD、sim、tmp目录或研究result；限制BLAS线程。writer仅新建独占目录的report.json，16MiB cap、200MiB floor，起始至少216MiB空间。实际报告约155KiB，主体为完整输入bindings与8格checkpoint引用，不复制journal/PNG/大数组；硬cap始终生效。所有输入先只读验证；失败不发布完整comparison、不覆盖已有目录。

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_moving_budget_scheduler_comparison_20261027_v1.py

OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.moving_budget_scheduler_comparison_20261027_v1 --scheduler-output /ABS/ACTUAL_SCHEDULER_OUTPUT --scheduler-report-sha ACTUAL_SCHEDULER_REPORT_SHA --physical-output /ABS/ACTUAL_PHYSICAL_OUTPUT --physical-report-sha ACTUAL_PHYSICAL_REPORT_SHA --physical-source-sha FINAL_MEITNER_SOURCE_SHA --output /ABS/NEW_COMPARISON_OUTPUT
```

这些外部SHA只能由MAIN真实终态给出；agent没有编造hash、生成prepare或启动模型/物理。未来RGB16编码仍由MAIN执行，当前v1保持missing，不能补接未定义schema。

## 真实saved素材验证（不是新CUDA/物理）

已对接MAIN实际scheduler report `results/moving_history_budget_scheduler_cuda_20261027_v1_cuda_main01/report.json`，SHA `710892008bbd47aa188a9b9391972976d3a94d2a0067e46911ff37f05be8369b`，prepare `330bb32be9cc3ccf330835b94e336fff9aad057a6b85ce468678d65397369aaf`；ownconsume实际只读成功。physical report `results/moving_budget_scheduler_physical_20261027_v1_main01/report.json` SHA `c9169b685e4e7f7d79ea2101aa81ad55b95593f96d6158ed8ff0815de136bd70`，prepare `bddd73792da6ecabcc669e75d9c466940f763571990498f3363375336c7d6344`；实际source SHA `60ad315b96c2c9eda21cc79fe4b07710eafd6d16027d8b0ff46641bb794e3f1a` 同时被该prepare绑定。

纯saved重建8full10/0native success、176PNG、1632physics，完整warmup/selected/reset分账一致。两根w为162.32405169996156/156.2867146609139，D/S均取整为10/10，U/N本就10/10；每根四策略literal commands、model Q和实际终态均完全相同，所有D−U/S/N差为0，但确有8次独立物理dispatch。policy_equivalence机器字段直接记录这些等价关系，不能把整数配额碰撞写成实现未派发或P3理论负结果。

四策略双根相同均值：model Q `0.38319819392269583`、native raw-state L2 `306.8156168517841`、object position error `38.28866715041848`、wrapped object angle error `0.1770800073650429`。它们不是同单位或同一指标；native success与wrapped success均0/2/策略。新RGB同Q仍missing，不把此模型Q均值冒充真实收益。

复现实际saved-only输出（新目录，绝不覆盖）：

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.moving_budget_scheduler_comparison_20261027_v1 --scheduler-output results/moving_history_budget_scheduler_cuda_20261027_v1_cuda_main01 --scheduler-report-sha 710892008bbd47aa188a9b9391972976d3a94d2a0067e46911ff37f05be8369b --physical-output results/moving_budget_scheduler_physical_20261027_v1_main01 --physical-report-sha c9169b685e4e7f7d79ea2101aa81ad55b95593f96d6158ed8ff0815de136bd70 --physical-source-sha 60ad315b96c2c9eda21cc79fe4b07710eafd6d16027d8b0ff46641bb794e3f1a --output results/moving_budget_scheduler_comparison_20261027_v1_saved01
```
