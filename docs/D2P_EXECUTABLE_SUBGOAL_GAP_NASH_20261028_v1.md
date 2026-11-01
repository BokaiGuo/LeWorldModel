# Experiment1：Executable Subgoal Gap输入/评分合同

本轮交付为工程输入与saved算术评分，不是三臂实验完成。只新增本文、
`le-wm-main/experiments/spectral_reachability/agent_subgoal_gap_contract_20261028_v1.py`
和同stem tests；不修改任何旧源码、封存结果或共享总账。无后代agent、联网/下载、模型加载、
神经/physics、训练/mapper/Q/normalizer拟合、阈值放宽或outcome选根。
使用完整读取的experiment-code技能：先核既有资产，再补具体缺口；不按结果择优或造正例。

## 1. 先检索后的具体缺口

已核 `D2P_RESEARCH_REQUIREMENT_EVIDENCE_MATRIX_20261026.md` 的E1、
`D2P_BASELINE_FAITHFULNESS_AUDIT_20261026.md`、
`p1_experiment1_learned_subgoal_bridge_20261027_v1.md`、
`p1_hilewm_text_contract_20261027_v1.md` 及最新
`D2P_BASELINE_COVERAGE_PAULI_20261028_v2.md`。已有资产分别是：

| 资产 | 真实能力 | 不能替代 |
| --- | --- | --- |
| audit/analyze_reacher_d2p_waypoint_tracking | 动作生成waypoint、同源执行/反馈、完整向量偏差和causal receipts | arbitrary latent优化臂、learned prior、native success |
| p1_experiment1_learned_subgoal_bridge innerdoc | SAGE generator接点、坐标/时域/normalizer配对设计 | 已实现的三臂runner或已到手checkpoint |
| p1_hilewm_text_contract saved文本 | Hi-LeWM high rollout→首个预测subgoal→low cost的正式接口 | 本地真实high/low配对推理、SAGE直接generator |
| p1_hilewm_bounded_object_fetch v2 | bounded获取工具与失败保留 | 配对资产已可用 |
| 已封存P2 native两边见证 | prefix实际预测mid后编译suffix，原动作/context证据 | arbitrary/learned子目标或独立P1 |

未找到统一的三臂可执行saved输入/分离评分合同，所以新增纯标准库scorer；不是再导出旧别名。
不新造planner、learned prior、三臂运行结果或物理消费者。

Hi-LeWM最新本地probe：
`le-wm-main/results/p1_hilewm_bounded_object_fetch_20261027_v2_probe_main01/report.json`
SHA `669bf41043dfae01a9bca183fe230d61baad29234c5cc73038c80409844e0b90`，
status=`failed_preserved`，first_error=`TransferError: Truncated range`，unpickled=false。
这不是0%成功；无完整可信高/低层/normalizer/预处理配对资产，本地Hi-LeWM臂保持unknown。
没有发起网络请求；已有上游失败不重试、不换URL、不续传。

## 2. 可执行输入：三臂固定，不以缺失删臂

入口 `score_file(input_path, external_input_sha, (bindings_path, external_bindings_sha))`；
RAM API `score(spec, artifacts)`只检查结构/算术，不能当已完成字节认证。
CLI绑定输入与证据表的外部SHA，逐文件流式复核全部证据，拒绝未来SHA、缺文件、symlink、
重复JSON key、NaN/Inf、bool冒充数字及输入中额外字段；全部输出只向stdout，不写结果。

```text
spec = {
 schema: agent_subgoal_gap_contract_20261028_v1.input,
 latent_dim:正整数, checkpoints:[固定public step序号，递增唯一],
 trial_denominator:N, trials:[trial,...恰好N], evidence_sha256:[实际证据字节SHA,...]
}
trial = {trial_id, shared, arms:[arbitrary,learned,compiled]}
shared = {root,history,task_goal,model,encoder,normalizer,q,controller,
          observation,budget,native_success_evaluator: 各为SHA256}
row = {arm,shared_sha256:canonical(shared)SHA,status:unknown|failed|completed,
       proposal,target,predicted,observed,native,resources,error}
proposal = {source_sha256,generator_receipt_sha256:null|SHA,action_witness_sha256:null|SHA}
target = checkpoint×latent_dim完整数组
predicted = null | checkpoint×(完整向量|null)
observed = checkpoint×(完整向量|null)
native = null | {success:严格bool,evaluator_sha256,evidence_sha256,task_goal_sha256}
resources = {明确资源轴:非负字面整数|null,...}
bindings = {绝对证据文件路径:外部核实的实际字节SHA,...}
```

三臂名字固定为 `arbitrary_latent_waypoint`、`existing_learned_prior`、
`dynamics_compiled_witness`。所有臂shared摘要必须相同，不能改root/history/goal/model、
controller、观察权限、Q、normalizer、预算或success规则。shared各SHA均需实存证据。
root/history的专用语义为canonical完整witness root及root.context；其余资产应明确绑定
不可变配置/权重/原生成功规则。预处理属于observation，不能仅凭latent维数认定配对。
task_goal是原任务目标，不是latent subgoal，更不是PushT renderer的goal_pose outline。
不能用object-only位置替代原PushT包括agent+object的位置成功条件。

unknown行要求proposal/target/predicted/observed/native均null、资源值全null、非空原因；
因此缺learned资产不会变成随机prior、final-goal或零成本。failed保留错误与已观察prefix，
未观察checkpoint为null；completed只表示低层trace完整，不等于native success已知。
所有槽位始终保留，输出每臂N分母、unknown/failed/native_known/native_successes分开；
不将unknown记false，不报告删去缺失后的“总成功率”。

### learned配对回执

仅在真实资产到手并经MAIN审查后生成小型JSON回执；当前没有这样的真实配对回执，
不得用下述schema回填true伪造。绑定表同时必须含实际prior checkpoint与receipt文件。

```text
{schema:STEM.learned_pairing,
 method:HiLeWM_hierarchical_adaptation|SAGE_generator_adaptation,
 prior_checkpoint_sha256,model_sha256,encoder_sha256,normalizer_sha256,
 observation_sha256,history_sha256,task_goal_sha256,
 targets_sha256,lowlevel_targets_sha256,
 generator_attempted:正整数,generator_completed:相等整数,fallback:false}
```

各坐标/输入摘要必须等于shared，两个targets摘要都必须等于实际低层target完整数组摘要；
只调用high predictor却不交给low不合格。Hi-LeWM与SAGE身份分开；Hi-LeWM使用
`rollout_high`后首个预测state，SAGE generator有不同offset/fallback合同，不能互换。
Hi-LeWM正式D25/H50等matrix与当前H10不是同协议；SAGE H50的[25,25]也不能套成Hi-LeWM
硬门槛。必须选一个预声明共同新适配器合同并披露偏离，所有新臂共同重编码/重执行。

这些byte/字段检查不证明回执作者诚实，也不代替MAIN对许可、训练来源、immutable revision、
checkpoint tensor坐标与固定RGB探针对齐的审查；scorer不反序列化checkpoint、不加载模型。
它是输入/评分合同，不是外部身份认证、完整runtime计量器或可直接放行的runner。

### dynamics witness回执

`action_witness_sha256`必须指向已有v2完整 `{edges,receipt,transition_validation}` JSON。
复用原compose在saved edges上重建receipt，核root/完整history/model/encoder/Q/normalizer；
边的累计raw动作步必须恰好对应checkpoints，各target必须恰为对应输出context末latent。
因此任意latent不能仅改label成为“compiled”。这只复核保存的动作/context绑定，
不是另一次神经重放或已证明物理可达；low实际跟踪仍须另行真实观测。

## 3. 评分：实现偏差、预测偏差、原任务结果分栏

每个预声明checkpoint分别计算：

`lowlevel_l2 = ||observed_z - target_z||₂`

`lowlevel_mse = sum((observed_z-target_z)²) / latent_dim`

对predicted_z也报告独立同式偏差；不拿predicted替actual，不拿最终任务goal替subgoal。
末checkpoint就是endpoint偏差，完整列表是轨迹checkpoint偏差；没有连续trajectory就
不声称逐物理tick误差。缺观察保持null，零和负结果保留，不加合成跨指标总分。
本版不发明可实现比例阈值；若需“可实现”比例，须MAIN在独立校准冻结具体阈值/定义，
当前只报告连续偏差。native success必须来自相同task_goal与原native evaluator的单独
实存证据，严格bool；latent误差零也可能native=false。无原success回执就是unknown。
资源只报告原声明的actual轴，缺测null；同预算配置/候选数不是等FLOPs，scorer不计量硬件，
不验证运行时cap执行。计量与durable dispatch由MAIN将来的runner承担。

## 4. 已接通真实saved诊断：保留协议差异

真实旧Reacher报告SHA `035493fc45022e0971a526ec04c22b6f471bbea7b1294ba8f4a858cd6b81a750`；
selection SHA `3a8f627cce045c8178b246bad157a31fe3f8b92f3d92aa7cd0b0f985774e4398`。
新 `legacy_waypoints` 先要求外部report SHA，再调用原analyzer.load完成既有原协议、
root/selection/reference、384个stage/causal receipt、动作/向量/Q/计数校验及误差重算。
输出所有8 roots×4 families×2 modes=64 branches、384 checkpoints原偏差；不是64个独立根。

其waypoints来自动作rollout；没有arbitrary latent优化臂、没有learned prior。协议是Reacher、
6×5步、damping .01、4迭代；replay与tracking动作/反馈预算不同，不是本轮PushT三臂公平对照。
`final_qpos_distance`与原goal RGB cost保留原字段，不反推出native success，后者全部null。
命令只打印诊断JSON，不新增/覆盖封存产物：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_subgoal_gap_contract_20261028_v1 legacy-waypoints --input results/reacher_d2p_waypoint_tracking8_20261026_cuda.json --sha256 035493fc45022e0971a526ec04c22b6f471bbea7b1294ba8f4a858cd6b81a750
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_subgoal_gap_contract_20261028_v1_tests.py
```

未来真实三臂输入到手后的CLI形状为 `score --input ABS_INPUT --sha256 EXTERNAL_INPUT_SHA
--bindings ABS_BINDINGS --bindings-sha256 EXTERNAL_BINDINGS_SHA`。当前没有真实三臂输入，
故不给伪造可运行的输入文件或未来SHA。RAM fixture不算真实prior/物理执行结果。

## 5. 明确未完成

arbitrary latent提议/优化规则与实际生成、同controller低层跟踪runner；Hi-LeWM/SAGE完整
配对资产/许可/坐标等价及真实generator调用；共同新任务root/goal/history/预算预声明；
原native terminal回执和exact恢复、全资源实际计量；独立P1/freshness与MAIN物理放行。
P1未过不解锁physical或宣称E1/组合成立。本工具始终scientific_pass=false、
physical_execution_authorized=false，成功退出仅表示saved合同/算术检查通过。

## 6. 实际验证与冻结交接

本stem单测24 passed；与原saved analyzer回归联合 **52 passed in 8.39s**：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_subgoal_gap_contract_20261028_v1_tests.py tests/test_analyze_reacher_d2p_waypoint_tracking.py
```

已实际执行上文legacy-waypoints CLI，退出0；stdout摘要为root_count=8、branch_denominator=64、
checkpoint_denominator=384、native_unknown=64、arbitrary/learned/native_success=null、
scientific_pass=false。只读原证据，无新结果文件。原analyzer import产生临时matplotlib
cache提示，不是神经/物理执行或验证失败。三臂score CLI只在RAM metadata fixture上验证，
包含实际临时文件SHA匹配/篡改拒绝，不能称真实learned-prior结果。
测试覆盖低层与native分栏、零偏差但native=false、unknown不补0、缺分母/观察、完整动作
见证与target错配、history/controller/goal不一致、learned坐标/fallback/未消费/未完成、
partial失败保留、bool/非有限JSON、外部字节绑定及旧真实causal链。
未运行任何新神经、CUDA、physics、下载或训练；未重跑P2 saved consumer。
