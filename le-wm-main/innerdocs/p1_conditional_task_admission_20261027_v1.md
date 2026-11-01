# P1 conditional-task admission：纯数组合同与当前四根审计

## 交付边界

仅新增同名 core、tests、本文及 `results/p1_conditional_task_admission_20261027_v1_saved_audit/report.json`。
不修改任何 compiler v1/v2、旧结果或共享 docs；未训练、未启动 CUDA/神经/物理执行，未读取物理 future 文件。
core 仅依赖标准库与 NumPy。main 已执行的 compiler CUDA 不是本实现执行的实验。
先完整读取 `docs/p1_conditional_root_preconditions_20261027_v1.md`、Reacher 机制/条件队列文档及 plateau/escape 实现与测试，再实现本合同。
使用 TDD skill 增加先失败的数组回归，再补实现；这些测试验证合同，不证明预测准确性或效能。

本合同输出始终 `P1_admission=false`，不认证真实机制、真实效能或数学驻点。
`model_task_eligible` 仅表示冻结模型/任务/有限资源下的条件任务筛查。
不包含“β1选择非贪心”筛根条件，也不把该模型筛查的通过率冒充独立科学验证。

## 实际 audit-current 结果

已运行保存数组审计，状态 `unknown_preconditions`。4根、8个 root/Q 臂、80候选的保存结构/动作尺度/normalized H2 history/同Q非线性分数一致；没有重放神经网络或独立认证 AD。
保存库相对各臂 stage-greedy 的严格 tradeoff 共3条：新Q根2 index5、新Q根3 indices0/4。不是新的模型或物理成功。

| root_index | root SHA 前缀 | domain | 直接平台 | 两段目标可达 | fresh |
| --- | --- | --- | --- | --- | --- |
| 0 | 9173218f71898716 | procedural_seed | unverified | unverified | false |
| 1 | 23b8de7837c39fb4 | procedural_seed | unverified | unverified | false |
| 2 | 4afefa3be02a327aa | procedural_seed | unverified | unverified | false |
| 3 | cf1b9bb913bf7815 | procedural_seed | unverified | unverified | false |

来源报告是 MODEL-ONLY 的 `pusht_d2p_p1_root_anchored_v5_calibration_20261026_cuda/report.json`，
文件 SHA `99cbb6b6dc4653eb629427353f6f6d05022c219333f3f7362cf4ca43a3076bc2`。
其四个 root SHA 与 v2 prepared 完全对应，`root_seed=202753010`、count4、oversample4、cohort calibration。
collector 按序从16个 boundary-contact proposals 取 unique 且初态非终止根，并没有运行 gradient plateau/target attainment 筛查。

`source_episode` 在这些 procedural-root 生成/报告中不是一个真实存在的 dataset identity，故报告为
`not_applicable_procedural_roots`，绝不以“缺 dataset episode”阻断 PushT。
实际缺失的 provenance 是每个被保留根的原 proposal index、生成/reset recipe 收据；不能把返回列表 index 或优化器的 `seed+index` 补造为生成 seed。
`make_boundary_contact_roots` 使用 terminal-pose 根与 `default_rng(seed+70000)` 接触采样；
下层 `terminal_pose_cases` 的随机 goal reset 使用 `seed+index`，另有显式 state/goal 的 seed0 重置。
所以不同父 seed 也可能重叠随机 goal-reset 来源；本新合同额外检查保守的半开 reset seed span，不把显式状态重放用的 seed0 当独立来源。
历史 seed202753010 的已知默认16根 terminal-pose 生成上界是32次随机 reset 尝试；这是源码上界，不是逐根实际 reset index。
未来 runner 应排除这一整个保守来源范围，以及其他已使用来源清单；不能只排除四个 root hash 后声称完全 fresh。

每根实际缺口已写入报告的 `roots[].missing_screening_fields`，不是等待真实结果后再决定门：

- `direct_runs`：预声明的所有初始化/重启，iteration0..B2 的连续完整20维 H2 优化轨迹，B1<B2；每步 proposal、是否接受、精确保留的 incumbent、该 incumbent 上的全H2 Jacobian/gradient。
- `witness`：合法完整两包 logits/raw 命令、两个完整 normalized-history contexts、whole/segmented replay、同Q/goal非线性终点代价。现有合法候选可作为候选见证，但没有冻结 τ 的目标达成判据，不能自动称 attainable。
- `journal`：上述 direct、witness producer、witness verification 的逐次 attempted/completed 资源收据。原 compiler 的66/11/12每臂账目不能充当尚未执行的 direct 筛查预算。
- `authority`：独立 calibration 规则、模型数组、来源与阈值冻结收据；不是强制用户手选数字。
- `identity/policy`：域特定有限 fresh 来源、排除清单、校准/确认分割与不足即停止。历史四根不能更名为 fresh confirmation。

## 可直接接入的 CUDA screening 数组接口

这里没有 CUDA runner。main 后续 runner 负责真实模型执行、源闭包/权重认证、durable journal 与外部 seal 时间顺序；数组 core 负责机器拒收不一致收据。
所有 SHA256 必须小写64位；内容 identity 使用 `json.dumps(sort_keys=True, allow_nan=False)` 默认空格，文件 SHA 与内容 digest 不是一回事。

### 1. procedural finite fresh scan

`procedural_scan(plan, plan_sha256, receipts)` 不调用 simulator，仅核验保存元数据：

```text
plan = {
  schema: NAME+'.procedural_scan',
  generator_sha256, environment_sha256, reset_recipe_sha256,
  proposal_limit_per_seed: positive_int,
  physical_future_labels_read: false, frozen_before_generation: true,
  slots: [{candidate_id, root_seed, reset_seed_span:[low,high), split}, ...]
}
receipts = [{identity, status:'completed'|'failed'|'metadata_rejected', error}, ...]
identity = {candidate_id, root_sha256, task_id, split, source}
source = {kind:'procedural_seed', generator_sha256, environment_sha256,
          reset_recipe_sha256, root_seed, reset_seed_span:[low,high), proposal_index}
```

上面 `[low,high)` 是半开区间说明，实际 JSON 是二元素数组 `[low,high]`。
冻结有限 seed slots、固定顺序/分割、每 seed 最大 proposals、初态 domain predicate，再做任何根生成。
runner 仅允许 reset/render/初态接触或非终止检查，禁止 step 动作读取物理 future。
新 generator 必须有界；不能盲用可能隐藏 while-resampling 的旧 root factory。
每个 slot 恰好一份 receipt；失败/拒绝用 null root/task hash 和非空 error，成功用真实 root/task hash。
每 seed 至多一个保留根，proposal_index 必须小于冻结上限；每个 slot 的随机 reset seed span 必须互不重叠。
内部 rejected proposals 的详细日志由冻结 generator 归档；core 检查的是有限 slots 的收据和边界，不伪称独立重做 simulator attestation。

将完整 `{plan,plan_sha256,receipts}` 放入 `policy.metadata_scan`，
将 plan 内容 digest 放入 `policy.source_bindings.metadata_scan_plan`；`policy.scan` 必须等于 census 中成功的 ordered identities。
这不允许模型失败后追加 seed；元数据失败也不能偷偷扩池。

dataset 模式使用 `source={kind:'dataset_episode',dataset_sha256,episode,start_row}`，`metadata_scan=null`，按整个 episode 去重/排除，start_row 不能绕过。
`policy.excluded_sources` 接收相应域 source 对象；procedural 按 environment+parent seed 或随机 reset span 重叠排除，不因 generator/recipe 改名而变 fresh。
`excluded_roots` 额外排除已用 root SHA，`exclusion_sources` 绑定历史排除 manifest 的真实文件 SHA。
SHA 自身不能证明排除清单完整；runner 必须封存其来源，尤其不能遗漏已有物理校准/结果使用过的来源。

### 2. 同任务、完整 H2 与连续 direct

`task` 精确字段见 `validate_task`：environment/task contract SHA、root_context、root_raw_action_history、goal、projection、模型/config/metric/goal-source SHA、representation、latent_dim/horizon/packet_steps/control_dim、action_mean/std、packet_logit_cap、feedback。
固定 native projected CLS192、open-loop H2、每包5×2 time-major，两个10维 packet logits，逐包 L2 cap。
`task_id=digest(task)` 绑定同Q/goal/完整根历史/动作尺度/模型与反馈日程。
root context 为三状态、两between-state action tokens；root raw action history 是10×2，严格 float32 normalization 对齐，不接受仅 allclose。
若使用 declared bootstrap，它的因果来源应由 runner 源合同明确，不得冒充真实已观测过去。

prediction 的精确字段是 `{task_id,logits,commands,contexts,whole_replay,segmented_replay,cost}`。
logits 为2×10，commands 为10×2，精确绑定 float32 tanh；context layout 固定 `states_with_between_state_actions_v1`。
contexts 长度2，每段依次 shift latent/action history，动作严格为 `(raw-mean)/std`，不可再除100。
whole_replay 保存最终context；segmented_replay 保存两个context，均要求精确相等。
`cost=mean((Q@(z_terminal-goal))**2)` 使用全部192维，不使用 own-reach surrogate 代替实际非线性终点。

completed screening row 精确字段：`identity,status,task,direct_runs,witness,journal,source_bindings,physical_future_labels_read`。
失败行精确字段：`identity,status,journal,error,physical_future_labels_read`；status 为 failed/not_started，失败预算不退款。
`direct_runs[restart][iteration]` 精确字段 `{iteration,incumbent,proposal,accepted,derivative}`。
iteration0 必须使用冻结 initialization，无 proposal/derivative、不接受；此后严格更低代价才接受，否则逐字段保留上个 incumbent。
B1不是重启点；B2后的最好值取所有预声明 restarts 的最小值。
derivative 字段 `{task_id,at_logits,at_terminal,raw_terminal_jacobian,objective_gradient}`，192×20 Jacobian 的列是 **logits**，raw 指未投影的 terminal CLS，不指 raw control。
必须包含 tanh/normalization 链且条件点等于前一 incumbent；core复算 `2*(QJ).T@Q(z-goal)/192`。
这里的 allclose 仅是导数收据算术一致性容差，不是 plateau/驻点科学阈值；core没有独立 AD 验证，也不重实现被冻结的优化器更新算法。

### 3. 预算、calibration、confirmation

policy.direct 冻结 method（full_H2_GN/full_H2_gradient）、implementation/configuration SHA、B1/B2、initial_logits、strict_improvement。
direct 每 restart 的完整 prediction+whole+segmented，每次6 predict/6 action_encoder；每个优化步骤另付 H2 ADforward 的2/2、1 fullH2 Jacobian，GN另1 SVD。
总 direct forwards 为 `R*[6*(B2+1)+2*B2]`，Jac 为 `R*B2`，GN SVD 为 `R*B2`。
本接口不支持免费额外 line search/未计账 batched proposals；需要这类算法必须先新扩展合同和预算，不可塞进当前 receipts。
producer（搜索或继承候选生成）的 predict/actencoder/Jac/SVD 预算单独预冻结且绑定 witness_producer 源码，不能把旧候选生成成本当免费。
witness verification 固定6/6/0/0；不改变封存 compiler 的算法、候选、66/11/12预算。
journal 每行 `{index,phase,evaluation_id,resource,samples,completed}`；phase=direct/producer/witness，resource=predict/action_encoder/jacobian/svd。
direct evaluation_id 为 `run_R_iter_I`（每个0..B2）或 `opt_R_iter_I`（每个1..B2），producer和witness的ID分别固定 `producer`、`witness`。
completed row 必须匹配全部阶段/逻辑分组预算；failed row 保留 attempted/completed，禁止超cap与补根。

`calibrate_authority(policy, recipe, calibration_rows, expected_recipe_sha256)` 提供真正可执行的独立数组阈值推导。
recipe schema=`NAME+'.calibration_recipe'`，environment/task_contract_sha256、method=`empirical_quantiles_v1`、minimum_rows、quantiles/offsets（均显式包含 epsilon_plateau/epsilon_escape/tau）、source_bindings、frozen_before_calibration=true。
recipe 在读 calibration 前外部封存，digest 预先绑定 `policy.source_bindings.calibration_recipe`。
只接收完整、按冻结顺序的 calibration completed rows，拒收 confirmation/漏行/失败后替换。
分别从 direct_improvement、escape_improvement、witness_cost 取显式 quantile（NumPy linear），加显式 offset，再截非负；没有默认 quantile、offset、epsilon 或 τ。
这是一种 outcome-free **model-cost 阈值规则接口**，并不替研究者论证物理 success 阈值、样本量充分性或统计校准质量；规则选择也不能依据本批已有物理结果。
生成 authority 保存 recipe 与全 calibration arrays；`freeze_contract(policy,authority,external_authority_digest)` 会重新推导数值并检查 calibration roots/source families，防止手改值后仅重算SHA。
也可接收独立评审的外部authority（derivation=null）；同样要求来源、任务绑定和 calibration identities。
authority 必须在 confirmation 模型筛查之前外部封存；布尔 chronology 声明和 SHA 不能自己证明真实时间顺序，外部预注册/runner seal 是必要信任边界。
缺 authority => unknown，不等于需要用户立即选数字，也不是任务失败。

`check_task(contract,row)` 的四项合取为：

```text
0 <= min_restart L(B1) - min_restart L(B2) <= epsilon_plateau
L_direct(B2) - L_witness > epsilon_escape
L_direct(B2) > tau
L_witness <= tau
```

`select_cohort(contract,rows)` 要求完整冻结池 census，包括 calibration/confirmation 和所有失败；仅按固定顺序选 confirmation eligible roots。
任一缺失/失败、metadata配额不足或合格 confirmation 数不足，均输出空 selected_roots，不追增扫描池。
状态 model_conditioned_cohort_only 仍不开放物理P1准入；后续真实机制/效能、非贪心选择、强基线等需独立协议。

## 最小命令与验证

实际运行61项通过（本新合同45项 + 既有Reacher纯数组准入16项）；包括calibrate→freeze→screen CLI、无合格根exit2、独占输出不覆盖。
另核对sealed compiler v2 source/tests/innerdoc SHA，均与交接值一致；未声称本次执行其CUDA。

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p1_conditional_task_admission_20261027_v1.py tests/test_reacher_d2p_model_escape_admission.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p1_conditional_task_admission_20261027_v1 --audit-current
```

runner 后续可用同模块的 `--calibrate calibration_input.json`、`--freeze freeze_input.json`、`--screen screening_input.json`。
对应 payload 分别为 `{policy,recipe,rows,recipe_sha256}`、`{policy,authority,authority_sha256}`、`{contract,rows}`；数值数组完整 schema 的可执行fixture在同名tests。
任一模式可加 `--output NEWPATH`，使用独占创建，拒绝覆盖已有文件；父目录需已存在。
screen没有选出cohort时exit2，audit发现unknown前置仍exit0（成功完成缺口审计，不是任务通过）。

已生成 audit 报告 SHA：`316f58f2808ad67b3c800926f4be4878365479897bb2c62d46b403569cb1927a`。
报告内绑定 core SHA：`d28d48c183791bfe1cae03e8a78ba27e28039cfd355993a19b32992cd7f6e211`，及v2 prepare/protocol/model report/origin report的文件SHA。
这是保存数组审计绑定，不是全传递 source/model seal 重验；本实现不通过导入 sealed compiler 或隐式 monkey patch 来运作。
