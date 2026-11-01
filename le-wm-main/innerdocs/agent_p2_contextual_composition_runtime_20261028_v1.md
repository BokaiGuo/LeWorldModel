# moving known2roots / four-slot contextual composition runtime

本交付只允许 MAIN 审读后 HOST CUDA 工程执行；worker 不运行模型、物理或网络。
个人有界交付：新增同 stem source/tests/本文，真实 dispatch 接线、RAM 回归、输入 schema 验证后冻结。
已有个人 goal 命名空间保留旧 blocked 目标，不覆盖或完成父研究目标。

## 为什么不是重复既有实验

equal144 已运行的是 moving 两个 off 根的 beam20x6 与 root-joint top16 spectral，
不含自身 predicted context 的 suffix Jacobian 重编译。
旧 frontier-v2 的 joint80 / conditional40 是另一四根、另一 Q 的实验。
原 allmode-v2 在本四槽使用42个单包前缀，以 stage+reach 评分，独立 packet-ball，
也不是本版共同 joint-H2 proposal bank、终态原 Q、80/40 WM-call frontier 对照。
因此不扩成三段，也不复跑旧实验或拿其结果代替本版。

复用 `assets.build` 严格消费既有 overlay 的16图像编码和四个 task exports；
复用 frontier-v2 的 `compile_cuda`、`suffix_cuda`、`run_arm`、`guarded_entries`、
`validate_result` / `validate_actual` / `validate_operations`。
`run_arm -> legacy.saved_witness -> d2p_composition_witness_v2.compose` 为真实原有动作见证路径。
原 source 不修改；没有 fixture=True 生产分支、没有观察 teacher15 的 selector 输入。

## 预声明算法与完整分母

固定 root-major (0,off),(0,on),(1,off),(1,on)，只有2 known roots，不是4独立根。
每槽从原 root10 的 observed z0/z5/z10 和两个真实 warmup action tokens 开始；
goal/Q/model/normalizer 均使用该条件的原任务。off/on 只标识原 renderer 条件，
不制造 goal_pose 或更改 native state。本版无 native 操作。

每槽新算原 joint-H2 zero nominal/J、ridge+spectral bank：
原40 = zero + 三个DLS尺度 + top9 × 两radius × 正负；追加40 seeded mixtures，
PCG64(20261027+四槽index)，不跳过零模态、重画、去重或按 outcome 选词。
joint 臂评估全部80个 root 生成 full10；conditional 臂保留原前40相同前缀，
每个前缀执行后在其自身**模型预测的完整中间 context**重新计算 suffix nominal/J/ridge。
中间 latent/action histories 均来自原 recurrence，native state 不传入 compiler。
两臂分别 stable min(cost,index)，cost 只取原 identity192 nonlinear terminal MSE。
固定480候选、8计划；任意失败不补候选/根，不释放部分计划。

保留 frontier 原联合20维 logit L2 cap=.75√10，suffix residual radius 由 prefix 占用精确计算，
radius=0仍调用 nominal/J，但不调 suffix SVD/solve。damping=.002。
动作使用原 frontier NumPy tanh(float64 logits) 后转换 literal float32。
这是明确的 frontier 动作域移植，**不是** equal144 独立 packet-ball projection 动作域，
也不冒称当前 P1 selector 原先采用此规则。不用动作重评分冒充新规划。

## 成本与持久化

每槽 shared base4 predict（nominal2+AD-forward2）。
joint80×2+winner segmented2+whole2=164 exclusive；
conditional40×(prefix1+suffix nominal1+AD-forward1+suffix1)+winner4=164 exclusive。
每臂 standalone168，paired unique332/槽，总1328 predict/action_encoder/predictor/pred_proj。
原 shared base只付一次，不把 standalone 相加。
原 nonlinear Q callback 另有80/40硬上限和逐次cost_attempt/cost_return，完整120项必须与
实际候选context digest/评分逐一对应；consumer的独立重算不冒充生产调用。
每槽41 Jacobian /7872 mathematical output VJP方向；shared2 SVD/1ridge，
suffix最多40 SVD/40ridge，按实际正 residual radius 记账。
这些不是 backward engine kernel计数或等FLOPs；未测 FLOPs、上游采集/历史搜索成本均 null。
原16 encoder/projector是已付 inherited encoding，本版新增0，不记作免费。

Gate 在原 native 入口前 fsync reserve_intent，再保存原 entry_attempt/terminal；
intent 的 actual_completed=false，不能当执行证据。原完整664双入口/槽与83项操作请求+返回
重建对账；996真实模块输入/输出/shape/上下文 per-slot 持久化。
nominal/J在下一AD/solve前写独立数组并绑定 task/context/prefix、数组原始 SHA。
失败保留 reached tickets/候选、四槽状态和480分母；后续槽 not_started。无重试或 padding。
运行128MiB输出上限，8MiB final reserve，200MiB磁盘floor，900s全任务边界时限，
不能抢占卡住的原生调用；磁盘故障/SIGKILL不能保证 final report。

consume 不执行模型：复用原 native全候选/原Q/全witness/replay验证，重算bank和suffix ridge，
核严格文件 census、immutable loader bytes、prepare/source/model、数组 SHA 与 binding，
每槽所有332 packet 的真实 raw→normalization→action_encoder→predictor→pred_proj→rolling
context 接线，包括AD forward、所有候选、双winner replay。未知和失败不改成成功。
不消费物理结果；封存只代表工程规划证据完整。无 independent P1/fresh cohort，
P1/P2/physical_release始终false，禁止将本结果视为组合成立或物理放行。

## MAIN 命令（从 le-wm-main）

```bash
.venv/bin/python -m pytest -q tests/test_agent_p2_contextual_composition_runtime_20261028_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p2_contextual_composition_runtime_20261028_v1 prepare --output results/agent_p2_contextual_composition_runtime_20261028_v1_prepare_MAIN01.json
```

使用 prepare stdout 的真实 SHA（不得预造）：

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p2_contextual_composition_runtime_20261028_v1 verify --prepared results/agent_p2_contextual_composition_runtime_20261028_v1_prepare_MAIN01.json --prepare-sha PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p2_contextual_composition_runtime_20261028_v1 run_cuda --prepared results/agent_p2_contextual_composition_runtime_20261028_v1_prepare_MAIN01.json --prepare-sha PREPARE_SHA --output results/agent_p2_contextual_composition_runtime_20261028_v1_cuda_MAIN01
sha256sum results/agent_p2_contextual_composition_runtime_20261028_v1_cuda_MAIN01/report.json
.venv/bin/python -m experiments.spectral_reachability.agent_p2_contextual_composition_runtime_20261028_v1 consume --output results/agent_p2_contextual_composition_runtime_20261028_v1_cuda_MAIN01 --report-sha REPORT_SHA
```

consume Python返回8计划，含 full10 literal commands/raw SHA、精确 root/context/goal/task
与 witness、候选库/selection digest、prepare/report SHA 和工程/科学分离标志。
不能从 observed selector suffix5自动接到 imagined 中间context；本版不存在该捷径。

## 验证边界

RAM fixture仅验证调用、分母、原 own-context suffix、动作/张量接缝及变异拒绝，
fixture_only=true不能经原 native validator 导出真实结果。saved schema回归只读原四个
task export/report，不运行 prepare采集、encoder或physics，不触 reserved4fresh。
生产 prepare/run_cuda/consume 仍由MAIN执行；本地测试不是科学证据。
