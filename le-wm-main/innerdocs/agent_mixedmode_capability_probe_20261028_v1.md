# Mixedmode fixed pre/post capability probe contract v1

本文件与同 stem 源码/tests 为新增独占交付。旧 mixedmode、allmode、physical、MAIN 产物完全不改。本模块**没有模型或模拟执行器**，只实现 saved build/prepare/verify、将来 evidence ABI 的 consume、精确复用资格函数及零相对响应算术。不得直接把 prepare 交给旧 physical runner：`execution_ready=false`，实际执行须 MAIN 审核一个新 adapter，并在其独立封存协议中补齐 render/internal-physics/wall/output 预算与权限。当前合同不授权运行。

## 固定干预、分母与含义

保持 mixedmode 原四项顺序 slot0/off、slot0/on、slot1/off、slot1/on，绝不看胜负选子集。每项角色 before/after，各固定五个 probe：zero、(+.1,0)、(-.1,0)、(0,+.1)、(0,-.1)，每个原 raw float32 命令重复五步。这与既有 `moving_overlay_capability_probe_physical_20261027_v1` 的五探针相同，而不是根据新结果选择方向。只覆盖常量 XY 控制，不能声称覆盖完整10维 packet。

每条独立新环境，以原 random seed + sample_state placement recipe 两次 reset 重建（不 reset(savedpoststate)，不解标签造目标）；原 warmup 前5[.1,0]后5[0,.1]。before 在 warmup10 后探测，after 再派发所选 literal prefix5 后探测。禁止动作二次 tanh、/100、裁剪或失败替补。四任务×两角色×五探针=40分支。完整成功时 reset80；warmup400+selected-prefix100+probe200=700 public steps，不把700当包括reset内部physics的总成本。constructor/close各40；所有资源的attempt/return独立记账，失败保留下界。继承 source编码、5352predict/892J/171264方向/896helperSVD等成本不是本实验新调用，须单列，不能称同总算力。

测量原生7维真实 endpoint 的同角色 `probe - zero` 响应，角度独立wrapped，再报告 after_response-before_response。agentXY、objectXY和其余state分量完整保留；不能把object-only无改善当agent+object整任务失败。零探针缺失则该角色所有响应null，失败分支null，不补零。不拟合新Q/阈值、不选能力赢家。前后时间/速度/接触/solver缓存不同：本轮只提供真实有限响应的描述性变化，尚无匹配时间控制，**不证明因果能力改变**；终距改善更不能替代它。没有真实数据时unknown。P1/独立确认始终false，固定库的零/负结果不否定理论。

## 已封存输入与 prepare

冻结 mixedmode producer SHA `ec912669c678960aac0d004a2e764b4bf036194673da5d1c63d0b535bc199396`；MAIN实际prepare SHA `8f6d027116b76334e80f9a761688ffd33825bc25d825484012d37230f083ed6b`。MAIN新告实际CUDA5352完成、report SHA `d6166b98958af70cecb8529135396c77467b7c803bfe12a1a5de29c422c22ad2`；本worker没有运行其consume或分析结果，实际目录须 MAIN 填实参，绝不猜目录或伪造prepare。

`build` 首先调用 frozen producer.verify/full consume，保留四个完整slot（packet/history/context/goal/Q/task文件SHA）、原选中raw10及split、选择digest、任务原生7维目标/根和renderer/goal图片SHA。复用 `agent_allmode_physical_20261028_v1.source_material/root_metadata` 读取原recipe/warmup17参考，不导入模拟器执行；其allmode42 material/run接口**不接受mixed222**，不能改名欺骗。完整bindings包含producer所有payload文件SHA而不复制数组。自身三文件也封存。metadata最大8MiB、diskfloor200MiB；write_new独占文件不覆盖。实际material尺寸未测，本轮不创建prepare。没有真实complete报告时build抛明确输入/consume错误，不输出可执行证书。

MAIN在审查后可执行（从le-wm-main）：

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_mixedmode_capability_probe_20261028_v1 prepare \
  --output /ABS/ACTUAL_MIXEDMODE_CUDA_DIRECTORY \
  --report-sha d6166b98958af70cecb8529135396c77467b7c803bfe12a1a5de29c422c22ad2 \
  --prepared /ABS/ACTUAL_MIXEDMODE_PREPARE.json \
  --prepare-sha 8f6d027116b76334e80f9a761688ffd33825bc25d825484012d37230f083ed6b \
  --source-sha ec912669c678960aac0d004a2e764b4bf036194673da5d1c63d0b535bc199396 \
  --destination results/agent_mixedmode_capability_probe_20261028_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_mixedmode_capability_probe_20261028_v1 verify \
  --prepared /ABS/NEW_PROBE_PREPARE.json --prepare-sha ACTUAL_PROBE_PREPARE_SHA
```

路径占位是未决依赖而非可直接复制的实际命令；`--help`可立即运行。本模块不含run子命令，禁止把该prepare称为已完成探针或交旧post-only执行器执行。

## 新 evidence ABI，供 MAIN 新 adapter 接线

`consume --prepared ... --prepare-sha ... --evidence ... --evidence-sha ... --executor-source ... --executor-sha ... --destination results/agent_mixedmode_capability_probe_20261028_v1_analysis_MAIN01.json`。

报告schema为本stem`.evidence`，包含prepare_sha256/executor_source_sha256/status(complete|partial)/rows40/new_neural_calls0/errors/counts/journal{file,sha256}/journal_tail。counts各资源{attempted,completed}，含constructor/close/reset/public_step/internal_physics_step/explicit_render，不能隐藏hook开销；complete reset80/public700/constructorclose40严格检查。internalphysics/render只核日志下界，真正上限必须在新executor预封存，当前null不是无限授权。

每row按jobs原顺序，job_digest/status(completed|failed|native_stopped|unstarted)/attempted/completed/end_state/error。失败end_state=null，全部失败仍保40。成功额外包含start_verified/context_digest/causal_prefix_commands/causal_replay_verified/rgb_guard_accepted/root_state/root_sha256/goal/goal_pose/goal_pixel_sha256/start_state/frames6。frames为初始+五步fresh RGB，{step,file,sha256,pixel_sha256,state,pre,post}，pre=post invariant包含snapshot（完整body/velocity等，不能只用state7）、goal_pose、renderer。consume检查PNG uint8 224² RGB、完整file/pixel SHA、snapshot state与实际返回state、目标/renderer。实际executor还须复用原RgbObservabilityGuard、每个时刻off/on/off及完整warmup RGB exact校验；这不是用几个`true`布尔替代MAIN对该executor实现的审阅。

平面文件路径不得traversal/symlink。journal完整换行、<=8MiB，每事件index/previous_sha256/event_sha256，digest为sort_keys紧凑JSON，末事件run_finished。事件job_digest关联固定任务；operation_attempted/operation_returned带operation_index/resource，全轴实际计数核对，失败attempt无return保留。causal_replay_returned带step/raw_bits/state/terminated/truncated，必须逐位派发原warmup10+对应selectedprefix5，并在warmup10真实到达原root；不能换成任意savedcontext。probe_attempted/probe_returned带step/raw_bits，return另含state。初始probe状态与最后causal replay return相同。无retry，<=5步。executor必须在原native终止/异常时停分支并保留记录，不发后续动作。该ABI是新显式合同，不伪装旧probe、旧close事件或producer schema；目前没有符合它的actual报告。

## 新候选与旧42相同时的精确复用

`reuse_check`仅给资格/缺项，不实际继承证据。必须由旧**完整own-consumer**验证原产物，保留原report/prepare/journal/PNG及consumer/source SHA。复用需同root/condition/task文件SHA（整个context/Q/model/goal）、blind/source、model/renderer、resetrecipe和warmup、同role、逐位同完整causal raw历史和probe raw5。不能用selected_index<42、模型cost相等、末state相近或prefix长度代替。函数的consumer_verified输入只允许完整旧consumer归一化后传入，不能把自填布尔当认证；返回evidence永远null，实际转存尚待MAIN adapter。原始规范化记录与旧完整证据必须一起保存。

before不在旧post-only库中，不能借旧after替代。相同prefix5和上下文只能支持同after探针资格；若后5 suffix不同，**不能**继承旧full10 endpoint/native success。欲复用完整任务物理还须raw10完全相同且执行因果链/目标/原环境源码一致，使用真实full10 own-consumer（旧prefix-only material不足）。snapshot相等不单独证明未导出的solver cache相等；历史重放、源码与版本也要绑定。继承是known-root非fresh，不新增独立确认；不匹配或旧失败保持unknown。

## 验证与尚未完成

仅RAM合成JSON/PNG/journal，不创建tmp fixture/prepare/results；限制BLAS线程。测试模拟NEW evidence ABI而非冒充实际数据，不导入模型或运行physics。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_agent_mixedmode_capability_probe_20261028_v1.py
```

未完成：实际producer saved build/consume与尺寸检查；MAIN审核新40分支executor及物理渲染/内步/墙钟存储上限、精确旧证据复用转换；实际prepare/物理/consume与因果时间控制。无新neural需求。实现goal完成不代表这些依赖完成，更不代表P1科学准入。
