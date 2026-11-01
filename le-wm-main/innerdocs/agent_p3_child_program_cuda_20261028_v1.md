# P3 paid-child progress compiler / independent replay

独占 source/tests/doc，stem `agent_p3_child_program_cuda_20261028_v1`。不修改 frozen v2、原 compiler、合同或 MAIN 产物。worker 只运行 RAM/saved tests，不执行 prepare/CUDA/physics/network。使用 experiment-code 的明确 logging/测试约束；不训练、不作图、不改 scientific gate。

## 完整分母和算法

只接受实际 `agent_p3_child_signal_cuda_20261028_v2.consume` 的完整持久 receipt，并在 prepare/verify 内重新完整调用 producer.consume；外部 report SHA、receipt SHA、输入 manifest SHA、prepare SHA、模型和所有源码/输入文件 SHA 全绑定。未知未来 receipt SHA 不填。拒绝 prepare/status摘要、旧 root J、仅 chi 标量或失败/partial source。上游 SVD/NumPy消费成本另列，不冒充新的CUDA付费或零成本。

固定8 episodes×b5/b10/b15＝24 child，每 child fixed/adaptive＝48 method。原 child ID、root/task、真实 observed RGB latent3/action2、raw10、executed-prefix ancestry不改。保原 choose 候选4/8、H1/H3、chi/gain/SVD阈值、depth规则。已执行物理 boundary/5 不是 model tree depth。本实现只执行从 observed child 出发的一次局部 extension；不假装完整 depth3 tree。

`2*H*c > 24` 在构造 backend/生成前拒绝整个节点，保完整 decision/queue SHA和槽位，不缩配额、不挑子集、无padding。H1×4 search8、H1×8 search16、H3×4 search24；H3×8需48而拒绝。每可执行方法额外支付1 predict做 selected raw5 replay。固定 search cap24不表示实际都花24，未用预算不补调用。

progress调用冻结 `bounded_ridge(Q@same_child_J, Q@(goal-nominal), damping=.002, original L2 CAP)`；方向除 `max(1,max(abs(direction)))`，fraction `i/(original_c-1)`，仅一次 cast float32。4候选与旧规则相同；8候选是显式新wrapper，不冒充旧common预算12/4分配。每 bank真实支付1 program_compile/solve/array_SVD，未复算新的J。

每个原候选独立whole-H、segmented-H重放；segmented在每包边界由完整原float32上下文重建tensor，不缓存候选/模型计算。逐边界 exact 比较。全部候选返回后以原 Q terminal mean-square→稳定原index选唯一winner，再从同 observed parent独立replay其真实raw5，必须等于winner第一边界。封存原始AE/predict/predictor/pred_proj IO、完整packet/boundary、输入history/normalizer、bank与每候选路径和Q、实际scope/attempt/return ledger及selection digest。只输出model计划，不释放physics动作。

旧native generator明确没有封存reconfiguration生成目标/规则。若未超预算但请求reconfiguration，本实现明确 `blocked_undefined_reconfiguration`，不把progress改名。超预算优先标预算拒绝。MAIN本批已确认 pending 为fixed23 H1×4、adaptive18 H3×4；adaptive5 H3×8拒绝，另2 not_reached，因此本批预计41可执行，5拒绝，2未知，不能扩展为一般reconfiguration已解决。

## 真实资源、失败与限制

当前报告实际SHA（MAIN公布，不代表本worker重新消费）：`cccb96d7d57dcf041021f8c43ab1f027a1012732ec56068ef1fb85348829bf5a`，路径 `/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_signal_cuda_20261028_v2_cuda_MAIN02/report.json`。来源prepare MAIN02 SHA `242dc62b1817bc3925cd836740cc9eaf60c8f94644e327b494fb834eb98ec298`；prepare仍经完整producer重建，不单凭本文SHA认可。

上游全部46 acquisition实际368predict/92J/17664 scalar VJP/92SVD/46schedule保留。预计compiler另新增657predict＝fixed23×9＝207，adaptive18×25＝450；41新SVD/solve/compile、369 replay、164 metric；0新J/VJP/visual encoder/projector/physics。已付费但拒绝的5 adaptive signal照记，不退款。固定方法加上其付费signal为23×17＝391predict；adaptive为23×8+18×25＝634predict；不声明实际相同API/sample、FLOPs或wall，历史P4 RGB/physics亦不并入这个局部和数。

run默认在verify/import模型前要求 `MAIN_APPROVED_HOST_CHILD_PROGRAM_CUDA`。真实运行CUDA-only、eval/frozen、float32、TF32off、MATH SDP、无CPU fallback，无下载。512MiB总output、256MiB final reserve、disk floor200MiB、wall900s；wall从verify前至最终源检查/journal close，不含最终report write。复用冻结v2 lossless单member gzip Store/增量reader，不改旧globals。每方法实际账独立，不把inclusive nested wall加总。

失败保留partial candidate/ledger与未调度槽，停止后续执行，不partial-select winner；unknown不是0。strict consumer拒绝任意缺失分母、旧/跨child J、配额/动作/归一化/历史变更、浮点/IO/shape/链接/嵌套/顺序篡改、Q或winner篡改、缺失独立prefix，以及任何工程失败。saved consumer NumPy SVD与Q只核保存数组，不独立认证神经计算或AD导数真实性。全report仍一次读取；journal按方法增量验证，不持全journal。无可执行方法时run不加载模型，consumer要求模型/host/peak均null，不能由此冒充CUDA结果。

所有P1/P2/P3/P4、freshness、full C3、physical release、compute/wall优势门保持false。未完成：定义和审核reconfiguration真正目标、child后代新付费信号及全组合树、native物理桥/闭环效用、fresh cohort与科学准入。测试不计科学证据。

## MAIN CLI（所有路径absolute；未来SHA须填实际值）

实际完整strict回执已由MAIN落盘：`/home/gbk/code/LeWorldModel/le-wm-main/results/consume_child_signal_MAIN_20261028_v1_MAIN02.json`，94023655bytes，SHA `efccb0ad57ff76e126e7027e9518b1c24b092c806a71a0ed55dcd96fa7ac90b5`。源码仅接受这份实际MAIN02 report/receipt，禁止替换cohort。RAM saved测试只读投影确认41 ready/5 budget refused/2 not_reached及657 predict预算；这不是worker实际strict消费或CUDA执行。

`inputs`命令为MAIN排他创建固定四字段 manifest：`signal_output`、`report_sha256`、`signal_receipt`、`receipt_sha256`。不能用prepare或compact摘要代替。该canonical manifest的已计算SHA为 `f7455c34f7b1035fc8043b7500e4d70ba23b441ab21454cd3cdbded378a90ad2`（仅确定输入内容的hash，不是尚未生成prepare/report的hash）。worker不写MAIN manifest。receipt读上限256MiB；本compiler不改producer的receipt写入限制。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m pytest -q /home/gbk/code/LeWorldModel/le-wm-main/tests/agent_p3_child_program_cuda_20261028_v1_test.py
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_child_program_cuda_20261028_v1 inputs --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_program_cuda_20261028_v1_inputs_MAIN01.json
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_child_program_cuda_20261028_v1 prepare --input /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_program_cuda_20261028_v1_inputs_MAIN01.json --sha256 f7455c34f7b1035fc8043b7500e4d70ba23b441ab21454cd3cdbded378a90ad2 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_program_cuda_20261028_v1_prepare_MAIN01.json
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_child_program_cuda_20261028_v1 plan --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_program_cuda_20261028_v1_prepare_MAIN01.json --sha256 ACTUAL_COMPILER_PREPARE_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_program_cuda_20261028_v1_plan_MAIN01.json
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_child_program_cuda_20261028_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_program_cuda_20261028_v1_prepare_MAIN01.json --sha256 ACTUAL_COMPILER_PREPARE_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_program_cuda_20261028_v1_verify_MAIN01.json
```

仅MAIN审读后HOST CUDA：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_child_program_cuda_20261028_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_program_cuda_20261028_v1_prepare_MAIN01.json --sha256 ACTUAL_COMPILER_PREPARE_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_program_cuda_20261028_v1_cuda_MAIN01 --authorization MAIN_APPROVED_HOST_CHILD_PROGRAM_CUDA
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_child_program_cuda_20261028_v1 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_program_cuda_20261028_v1_cuda_MAIN01 --sha256 ACTUAL_COMPILER_REPORT_SHA --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_program_cuda_20261028_v1_consume_MAIN01.json
```

inputs/prepare/plan/verify/run/receipt输出均排他，禁止覆盖或重入。无receipt时consume会打印完整数据，实际执行建议始终指定receipt。run输出compact SHA/counts/resources。未授权CLI run及所有RAM tests均不触模型。已绑定真实producer receipt；未来compiler prepare/report SHA仅由实际生成后填写。完成实现不等于MAIN验收或P3/C3通过。

## worker实际验证与冻结边界

已实际运行 compiler + frozen v2 signal + frozen child contract 联合回归：175 passed in 61.34s，其中compiler69项。命令为上述pytest入口后追加绝对路径 `/home/gbk/code/LeWorldModel/le-wm-main/tests/agent_p3_child_signal_cuda_20261028_v2_test.py` 与 `/home/gbk/code/LeWorldModel/le-wm-main/tests/agent_p3_child_scheduler_contract_20261028_v1_test.py`。包含真实saved metadata 41/5/2投影、精确成本、RAM完整strictconsume、回执hash/上游consume失败优先、输入闭包、原始数组/资源/候选/选前缀/链重封篡改拒绝、8候选/重复动作付费及稳定tie、冷导入无torch/physics。实际prepare/run/真实consume、CUDA/physics/network均未由worker执行。三件套交付冻结后不再原地修改；后续修复新版本或MAIN明确接管。
