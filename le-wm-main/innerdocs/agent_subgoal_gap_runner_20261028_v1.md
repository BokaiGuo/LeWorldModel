# E1共同controller runner骨架 v1

只新增本stem source/tests/本文；不改contract v1、MAIN文件或任何封存证据。使用完整读取的
experiment-code技能，先固定接口、资产与失败路径，后跑RAM callback测试；不运行神经、
CUDA、physics、训练、网络或下载，不派生后代。不按结果挑根或调科学阈值。

## 范围与当前状态

本模块有实际三臂dispatch控制流，不是manifest或旧函数alias。三臂顺序固定为arbitrary、
existing learned prior、dynamics compiled witness；所有trial×3槽位先建账，失败停止后
未派发槽仍保留。每臂从同一封存input恢复，再提议、调用同一个controller、单独native评估。
共同checkpoint horizon、模型/encoder/Q/normalizer、观察、raw动作域和预算不允许臂内覆盖。
既有contract v1评分/compiled witness/learned回执验证直接复用，不改其数学或冻结SHA。

这是**可集成骨架**，不是现成真实模型/物理adapter。没有真实Hi-LeWM/SAGE配对资产、
controller绑定、exact restore实现和MAIN权威材料，真实run均不可执行。不能把metadata
fixture或`engineering_ready_NOT_authorized`当三臂结果、P1/P2/E1或物理放行。

## prepare输入与资产清单

CLI仅prepare/verify，stdout打印JSON，无模型导入/结果写入。prepare允许明确null资产，
生成blocked+缺失清单，不补文件、未来SHA或默认controller；错误SHA/已存在坏配置直接拒绝。

```text
spec = {schema:NAME.input,latent_dim,checkpoints,assets,phase_caps,trials}
ref = {path:绝对本地文件路径,sha256:实际外部核实的字节SHA} | null
assets = {
 model,encoder,normalizer,q,controller,observation,budget,native_success_evaluator,
 learned_checkpoint,learned_pairing,arbitrary_rule,compiled_rule,adapter_source,restore_contract: ref
}
trial = {trial_id,shared:contract-v1完整SHARED,
         input:ref,restore_binding_sha256,proposal_inputs:{三个原arm名:ref}}
input文件 = {root:原完整witness root,task_goal_sha256,restore_binding:完整起点合同}
```

prepare重算root及root.context的canonical摘要，与shared.root/history对应；task_goal与
restore_binding摘要必须一致。shared其他8个共同资产摘要必须与asset文件字节对应。
restore_binding应明确原native setter/snapshot、状态、goal7、warmup/history、renderer与
wrapper身份，禁止approximation。具体字段由MAIN的真实restore_contract固定，不在骨架
发明PushT setter或认为返回一个摘要就完成物理恢复。MAIN须核原state恢复方法及实测精确性。

`learned_pairing`是预运行资产配对文件，不是未来generator调用回执：

```text
{schema:NAME.pairing,method:HiLeWM_hierarchical_adaptation|SAGE_generator_adaptation,
 prior_checkpoint_sha256,model_sha256,encoder_sha256,normalizer_sha256,observation_sha256}
```

真实checkpoint本地字节和坐标资产须匹配。Hi-LeWM与SAGE保留不同方法身份，不把随机、
final-goal fallback或普通proposal bank当learned。checkpoint许可、训练来源、immutable
revision和逐tensor/预处理语义认证由MAIN审查；哈希格式本身不是配对真实性证明。

`controller`文件：

```text
{schema:NAME.controller,latent_dim,checkpoints,action_low,action_high,
 bindings:{model,encoder,normalizer,q,observation,budget,native_success_evaluator,
           adapter_source,restore_contract:各资产SHA}}
```

`budget`必须精确等于 `{schema:NAME.budget,phase_caps:spec.phase_caps}`，不容两个版本。
每个trial还有三份真实proposal输入ref，不能因缺learned查询就只运行另两臂。
prepare绑定本source/tests/doc、原contract和compose源码；verify重建同一prepare并检查
所有原输入/source SHA，不能prepare后继续修改闭包。

## MAIN集成API：一个controller，显式真实回调

`Adapter`为固定dataclass：backend、bindings（精确等于prepared.asset_hashes）、restore、
proposers（恰好三个arm）、controller、native、preflight。无字符串动态import、模型加载、
CPU fallback、pickle反序列化或下载入口。

```text
restore(input, work)
 -> {shared_sha256, restore_binding_sha256}  # 实际精确恢复后才能返回
proposers[arm](input, proposal_input, work)
 -> {target, source_sha256, learned_receipt, witness}
controller(input, target, checkpoints, work)
 -> {shared_sha256,targets_sha256,predicted,observed,raw_actions}
native(input, tracking, work)
 -> null | {success:bool,evaluator_sha256,task_goal_sha256}
preflight()
 -> {cuda_only:true,frozen_eval:true,parameters:完整tensor SHA,buffers:完整tensor SHA}
```

controller只有一个callable对象，所有臂收到完全相同input合同/checkpoints，不能换controller。
restore须通过work.call记录一次restore_calls；controller raw_actions必须完整固定horizon，
有限且在原controller raw域，绝不在骨架里clip、tanh或再normalize。神经/真实物理仅MAIN
adapter执行；adapter必须在每次真实primitive入口用work.call包住它，不能在observer外偷跑。
source哈希只绑定MAIN审过的adapter，不自动证明Python回调内部行为。骨架没有实现原
native setter、实际normalizer/history转接、CUDA模块hooks或同步；这些是明确待接项。
HOST_CUDA路径还逐个检查Python callback的实际sourcefile SHA与adapter_source一致；因此
MAIN应在一个新封存adapter文件中定义这些薄入口（可调用原冻结内核），不能只在bindings里
声称controller来自某文件。RAM模式不要求此生产限制，且永不生成真实执行证据。

arbitrary source必须原arbitrary_rule；learned source必须真实learned_checkpoint，并返回
contract-v1的实际generator回执（attempted/completed、fallback=false、输出targets确实交给
低层、同history/model/encoder/normalizer/observation/task_goal）。还须计到generator_calls>0。
compiled source须compiled_rule，完整witness须经原compiled_pairing逐节点/context/动作验证，
targets恰为动作生成节点输出，不接受任意latent改名。proposal完整payload在控制前持久化。

native返回null时明确unknown，不能从latent偏差或object-only位置反推success。返回bool时
须原evaluator/task_goal身份且计到native_evaluator_calls；PushT原位置成功包含agent+object。
预测/observed向量和raw动作另存，低层偏差与native success分栏，不汇总为复合分数。
这是原生回执接口，不独立核实native实现是否忠实；MAIN负责接原validator/evaluator。

## 成本分解与durable边界

固定phase为restore/proposal/controller/native，三臂使用相同phase上界，不互相转借、不padding。
KNOWN轴：callback_calls、restore_calls、generator_calls、image_encoder_calls、projector_calls、
predict_calls、predict_samples、action_encoder_calls、jacobian_calls、math_vjp_directions、
svd_calls、solve_calls、physics_steps、renders、native_evaluator_calls。
UNKNOWN轴：flops、cuda_kernel_calls、backward_engine_api_calls、device_time_ns、peak_gpu_bytes、
inherited_context_cost。每个cap完整含全部轴；known非负字面整数、unknown严格null。
callback_calls=1/phase只计外层host callback，不是predict/GPU/physics调用，不能相加冒充FLOPs。

`work.call({真实操作known轴:数量,...}, callable)`：先检查累计reservation不超该phase cap，
写reserve intent、entry boundary，才调用；成功host返回写return后才进入returned下界。
失败保留pending/reservation，不推算真实内部完成量、不退款。计量/日志失败poison，禁止
callback吞掉错误后继续调用；original调用错误与失败日志secondary分开。
controller必须计满实际固定physics_steps，否则失败；restore必须exact一次；native未知
不伪造成功或调用计数。generator/encoder/AD等实际hooks仍由MAIN adapter提供。

生产Journal独占新stem results文件，每条hashchain，flush/fsync(file)，新建后fsync(parent)，
单事件上限4MiB。数组/phase payload在返回后逐条保存；journal不可写即停止，无重试/覆盖。
没有自动resume、补根、另选winner。崩溃/SIGKILL/磁盘耗尽不保证最终report；已有日志保留。
当前没有整任务硬wall/output/disk quota/watchdog，也不宣称为完整资源匹配runner。
共享上游context获取/FLOPs等未知，所有报告`full_cost_instrumentation_complete=false`。
预/后preflight全量hash相等只是MAIN adapter回执一致，不是本worker独立加载模型证明。
费用表始终保留trial×3 arms×4 phases，planned_upper与reserved/returned分开；未派发phase的
reserved/returned/pending均null，不能把未启动/缺测槽填成实际零费用或省略失败分母。

## run gate与命令

`run(prepared, external_prepared_sha, adapter, journal_path, authority_path, external_authority_sha)`：
先verify、拒绝任何blocker，再检查HOST_CUDA adapter完整绑定，然后核外部MAIN authority，
最后才创建Journal/调用任何callback。MAIN authority精确schema：

```text
{schema:NAME.MAIN_authority,reviewer:MAIN,prepared_sha256,
 P1_passed:true,freshness_passed:true,physical_run_authorized:true}
```

这是外部MAIN实际审核的引用，非密码、非本runner可签发证书；字段填true不证明科学通过。
当前P1/freshness与资产缺失，不能创建这种授权供真实run。本worker不生成或使用它。
所有结果仍scientific_pass=false、physical_execution_authorized=false，不授权后续物理任务。

`run_fixture(prepared, sha, adapter, emit)`仅允许backend=RAM_METADATA；仍须所有metadata
资产/controller配对到位，结果fixture_only=true；不接受HOST_CUDA adapter冒充fixture。
RAM日志不是实际执行证据。fixture物理步数只是dummy callback计数，不是物理操作。

CLI只向stdout输出prepare/verify，不提供自动加载模型/运行物理命令：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_subgoal_gap_runner_20261028_v1_tests.py
```

MAIN实际输入到手后：`python -m experiments.spectral_reachability.agent_subgoal_gap_runner_20261028_v1
prepare --input ABS_SPEC --sha256 EXTERNAL_SPEC_SHA`；verify同理读取prepared JSON与其外部SHA。
CLI的--sha256认证完整文件字节（含stdout末尾换行/排版）；API verify/run和MAIN authority的
prepared_sha256使用canonical JSON内容摘要。CLI先通过外部字节SHA，再重建canonical prepare；
不能把带换行文件的byte SHA直接当API canonical SHA。测试包含真实stdout的prepare→verify接缝。
没有真实spec/paired assets，因此不提供不存在的文件路径或未来SHA。

## 未完成项

真实Hi-LeWM/SAGE配对资产与逐tensor/RGB坐标证明；真实arbitrary生成器/compiled producer；
统一controller的实际冻结CUDA模块/hook/source closure；原native exact setter/完整state+goal7+
history/warmup/renderer恢复与独立核验；原native metrics producer；全资源计量与硬限制；
独立P1/freshness及MAIN物理批准。当前不运行神经/physics、不解锁E1/P1/P2，不声称组合成立。

## 本轮实际验证

上述专属轻量回归 **31 passed in 0.28s**；CLI `--help`退出0，prepare/verify完整stdout字节
roundtrip、独占fsync metadata journal均在临时RAM/JSON fixture上实际测试。覆盖三臂同一
controller/恢复输入、missing prior/controller/query导致零callback、P1拒绝、生产callback
source不匹配、固定失败分母、native unknown/false、generator调用计量、target身份、
cap先于执行/不能吞错重试、reserve/entry/return日志失败、原primary保留和未知费用null。
测试所有physics/generator计数均为dummy metadata callback，未执行物理或神经计算。
旧contract v1源码SHA重新核对仍为
`f8d48e7dec75b01ebac1bed915456dbc163acf0a83fac00bfb3e5463aee11c25`。
没有真实准备文件、HOST CUDA/physics作业、真实三臂结果或可用MAIN authority；不得以测试
ready fixture代替资产。仅本独占三文件交付并冻结，待MAIN审读/接真实adapter。
