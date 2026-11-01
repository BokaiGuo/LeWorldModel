# P3 同任务 common predict budget：独占实现准备

只新增同 stem source/tests/innerdoc；旧 native/signal/runtime/结果和共享总账不改。不启动神经、GPU、physics 或后代 agent。MAIN 审读源码后才能执行下列 CUDA 命令。无训练/拟合/adapter/normalizer/Q 修改，无 outcome 选根或阈值放宽。实现/测试通过不等于 P1、P3 或科学通过。

## 已检查旧队列与不能复用的部分

`agent_p3_runtime_contract_20261028_v1.choose/request` 是同任务 root 的 type/horizon/depth 决策；`agent_p3_local_signal_cuda_20261028_v1` 提供真实 nominal+AD/J、原短 H 参考 chi/gain 和 strict saved consumer；`agent_p3_native_program_20261028_v1` 已接真实 bounded ridge→四候选→whole/segmented native replay 和完整数组闭包。沿用这些实际函数，不重写神经 kernel，不改 globals。

MAIN 已运行的 native report 固定 SHA `7f5f16ffe9d440a5cd2731e1e87c2325b1b39de9c18508ebdcb28409cbe858d3`；本实现只哈希保全该先驱结果，不读其 winner/modelQ 或物理 outcome 来定规则。MAIN 所报 strict consume+终态 dot 回执 SHA 为 `189208fa11b7ff51ea0f1f72e04647b89220257bc808889ff59c3272278b73a6`，这里不把该消息当作新 comparison 实验。

旧四候选控制实际 H1/H3 回放费用不同，不能称等预算。旧 signal panel 在两个方法间共享一次采集；不能把它复制成两份、也不能只写两次扣账冒充两次执行。本实现要求 **两份独立、完整、原 sealed signal run 的实际 report/journal**，一份完全计入 fixed，一份完全计入 adaptive；两份原 consumer 都通过且输出路径/report SHA/journal SHA 不同、原 tasks/root/goal/Q/model/normalizer/规则精确相同。相异 SHA 是防意外复用检查，不是硬件远程证明。

## 事前固定比较，不读真实 outcome

每 task×method 均实际支付 H1/H3 两个 J：各自独立 nominal 和 AD forward，合8 predict API/8 samples，384 VJP、384 grad、384 raw-backward callbacks，2J、2 array SVD；原 panel 每 root 还产生 fixed/adaptive 两个 schedule，全部照实计入，包括本方法未采用的那个廉价决策。不存在额外 filler predict。

搜索预测预算固定24，推导为旧最小四候选×两遍独立回放×最大冻结 H3。整数配额 `N=24/(2H)`；H1为12候选，H3为4候选。所有候选都是实际搜索请求、均完整两遍回放并有资格赢；不烧空调用、不重复无用 winner replay 来填账、不人为延长 H1。若动作退化碰撞仍保留全部预定候选，不能事后去重、换 bank 或调预算。相同动作可由退化模型自然出现，不保证一定分离或收益。

因此每 task×method **32 predict API/32 samples**（8信号+24搜索），96 context tokens。两task两method合128 API/128 samples：信号32已分两份独立 receipt 全额实付，待运行搜索96。若原决策仍 fixedH1/adaptiveH3，则32候选、64 replay：fixed各12候选、adaptive各4。adaptive若合法选H1则也12；不为了维持 H3 改阈值。任何 reconfiguration/composition、子节点、缺失信号或非整除预算都阻断 prepare，不用假的 progress objective 代替。

**H1/H3执行时域不同**：每候选分别5/15动作步。本比较只能称同predict搜索预算对照，不能称同物理horizon或同闭环任务成功。prepare/report/consume顶层显式给 `equal_execution_horizon`（从两task实际决策逐对复算；本轮H1/H3为false），各分支给 `execution_horizon_packets`/`raw_action_steps`；strict消费拒绝把false改true。fixed也付H3是 **受控双信号shared-pipeline消融**，不是成本最省的固定H1强基线；两方法用独立实付执行，shared指相同管线设计而非共用一份receipt。顶层 `fixed_H1_cost_minimal_baseline=false`、`same_closed_loop_success_claim=false`；后续若要最省成本H1强基线，必须另立真实获取成本与预算方案，不能借本实验冒称。

原 core.choose 与它的四候选决策保持字节不变并记录 original_decision；这是 **新的显式 common-budget quota wrapper**，不冒称原四候选请求未变就能执行12候选。新 `common_budget_root_progress` 请求绑定原 queue SHA/response SHA/root/goal/model、candidate_index 和实际 quota；panel 逐项实际消费。原请求 prefix 必须空，H3是root的联合15动作预测，不是3个已重新规划child；executed_extension_depth=1，requested_depth保留但不执行全树。

每分支复用旧 `native.generate` 中原 `bounded_ridge` solver（damping=.002、joint cap=sqrt(5)*.75），保留原 Q/自身 H nominal、原 action normalizer；新采样固定 `i/(N-1)`，direction除以max(1,maxabs)后乘fraction、一次cast raw float32。H3四点与旧 `[0,1/3,2/3,1]` 一致，H1扩成12点，这是预定的搜索密度/预测horizon权衡，不是训练或事后阈值优化。新请求和生成器已对接真实 `NativeProgram.replay`，不是不可执行 gate 清单。

## 资源账、失败与严格消费

每 branch 分开给 signal/search/total 所有 core.AXES；API、samples、tokens、AE、predictor、pred_proj、J、VJP、grad、callbacks、SVD、solve、replay、metric、schedule不混算。每branch额外生成1solve/1SVD，signal2SVD；AD成本保留独立信号账，搜索AD为0（实际未调用）。signal/search inclusive wall 分别记录，不相加当总wall；signal整panel wall保留，但分摊到root的host/model加载墙钟未知记null。backward-engine调用/CUDA ns/FLOPs未知null；等predict不等于等FLOPs、wall、总资源。

新 search profile直接由真实队列计算：96predict、96samples、288tokens，AE/predictor同96calls/samples与288tokens，pred_proj96calls/288rows，4solve/4SVD/4program_compile，replay与metric随实际quota。新增搜索工程cap沿旧wall900s/output256MiB/reserve16MiB/diskfloor200MiB/CUDA allocated3GiB/reserved4GiB，TF32关、autocast关、SDP MATH。两份signal执行各有原已封存工程cap，不冒称三段合计只900s。前后验证及源哈希开销纳入本段wall。

prepare/verify各会通过两个原strict消费者，另做8次saved-array SVD。完整run前后verify另16次；consume verify8次+四generator重算4次=另12次。这些单列，不能把总SVD说成只有搜索4次。array_svd helper wall包含bounded_ridge的二分等数组工作，不是独立SVD kernel计时；嵌套inclusive wall不可相加。保留继承视觉/history采集成本的unknown，不将cached acquisition填0，不声称端到端等预算。

predispatch debit、失败不退款、不retry、不换动作；完整预定分母先持久。失败保留所有候选slot和null，不从部分结果选winner。strict consume完整重算quota/banks/actions/原Q/winner/分支32API和32samples，复用native.validate_arrays检查逐包raw/normalization/AE/predict/predictor/pred_proj原float32数组与精确parent-child状态/历史闭合。每候选whole/segmented每个边界精确一致，journalchain/attempt-return/实际input shape/model前后所有buffers/source closure均绑定。saved验证不执行神经，不是对神经结果的独立重算。

## 当前停点和MAIN命令

本工作位检查时已见一份真实 signal report `results/agent_p3_local_signal_cuda_20261028_v1_cuda_MAIN02`，SHA `5c776ea604f615081b9f50b92648e22e2612ef46d8443bf1949f69341ed4864d`。MAIN随后通知第二份独立signal主机CUDA已发起；本工作位不轮询或等待，也不编造其未来SHA。prepare仍必须取得其真实完成report并严格消费：**不能用同一份report两次生成可接受prepare**。没有worker comparison CUDA/report/收益结论。

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。第一条只输出静态事前规则，无模型：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_common_budget_20261028_v1 plan
```

MAIN可将上面已有完整signal receipt指派fixed（全额计费），另独立采一次给adaptive；也可两边都重采，但不得使用同一执行的复制件。原prepare02不改，SHA `23b42bbe311f83e06c6a8d263fbf15f1b6e2d1f8ec8ff3d4df8aa905c6243f4d`。仅MAIN审读调度后执行：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_local_signal_cuda_20261028_v1 verify --prepared results/agent_p3_local_signal_cuda_20261028_v1_prepare02.json --sha256 23b42bbe311f83e06c6a8d263fbf15f1b6e2d1f8ec8ff3d4df8aa905c6243f4d
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_local_signal_cuda_20261028_v1 run --prepared results/agent_p3_local_signal_cuda_20261028_v1_prepare02.json --sha256 23b42bbe311f83e06c6a8d263fbf15f1b6e2d1f8ec8ff3d4df8aa905c6243f4d --output results/agent_p3_local_signal_cuda_20261028_v1_cuda_COMMON_ADAPTIVE01 --authorization MAIN_APPROVED_HOST_CUDA
```

从上条**实际**返回取 ADAPTIVE_SIGNAL_SHA；未来SHA不能预填。接下来：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_common_budget_20261028_v1 prepare --fixed-signal-output results/agent_p3_local_signal_cuda_20261028_v1_cuda_MAIN02 --fixed-signal-sha256 5c776ea604f615081b9f50b92648e22e2612ef46d8443bf1949f69341ed4864d --adaptive-signal-output results/agent_p3_local_signal_cuda_20261028_v1_cuda_COMMON_ADAPTIVE01 --adaptive-signal-sha256 "$ADAPTIVE_SIGNAL_SHA" --output results/agent_p3_common_budget_20261028_v1_prepare_MAIN01.json
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_common_budget_20261028_v1 verify --prepared results/agent_p3_common_budget_20261028_v1_prepare_MAIN01.json --sha256 "$PREPARE_SHA"
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_common_budget_20261028_v1 queue --prepared results/agent_p3_common_budget_20261028_v1_prepare_MAIN01.json --sha256 "$PREPARE_SHA"
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_common_budget_20261028_v1 run --prepared results/agent_p3_common_budget_20261028_v1_prepare_MAIN01.json --sha256 "$PREPARE_SHA" --output results/agent_p3_common_budget_20261028_v1_cuda_MAIN01 --authorization MAIN_APPROVED_HOST_CUDA
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_common_budget_20261028_v1 consume --output results/agent_p3_common_budget_20261028_v1_cuda_MAIN01 --sha256 "$REPORT_SHA"
```

PREPARE_SHA/REPORT_SHA也必须取对应命令实际输出；所有输出路径必须新建，已存在时拒绝覆盖。RAM联合回归：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_p3_common_budget_20261028_v1_test.py tests/agent_p3_native_program_20261028_v1_test.py tests/agent_p3_local_signal_cuda_20261028_v1_test.py tests/agent_p3_runtime_contract_20261028_v1_test.py
```

新测试支持已import torch的全进程；冷导入性质另用独立Python subprocess验证，不删除sys.modules，不执行模型。旧sealed runtime的冷全局断言保留，四文件联合请新Python进程启动。

## 本工作位实际验证记录

- 最终代码四文件联合174 passed /24.11s；仅新测试在已import torch但不构造/执行模型的进程中47 passed /11.16s。
- 原真实signal report strict消费通过；用其真实saved J做prospective NumPy生成检查，固定H1各12个、自适应H3各4个合法不同动作，实际4次solve/4次SVD/4次array program_compile，neural/replay/physics为0。这只验证已有数组可接生成器，**不是第二份独立采集，不是completed comparison**；没有写synthetic研究结果/prepare/report资产。
- 同一真实signal路径/SHA重复用作两方法时，实际调用load_panels在消费前即拒绝；不伪造第二份账。静态plan CLI实际exit0。
- RAM覆盖完整search与saved-consume roundtrip、真实material接线（仅上游输入验证器换显式RAM metadata）、两份采集独立性拒绝、全部资源分轴/失败不退款/完整候选分母、动作/normalizer/native module/endpoint/metric/winner/预算与作用域变异拒绝；不执行模型。
- 三独占新文件whitespace diff-check无警告；旧native/signal源码SHA保持冻结值。未运行本comparison CUDA/physics，也未创建后代。

## 科学边界

只比较两known tasks的同root/modelQ搜索，不保证分开或获益。不是完整同任务C3树、不是物理收益、不是新根泛化。缺第二份独立J、MAIN审读/实际search CUDA与strict消费、freshness/P1与适用P2独立证据、paid child signal+真实reconfiguration/composition/tree适配和物理后果；所有admission/physical_release/compute_advantage/walltime_advantage flags保持false。PushT成功含agent+object，不将object-only投影非正当作完整失败。
