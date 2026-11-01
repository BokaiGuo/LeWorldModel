# Moving-history GN/D2P 最小部署 wrapper

仅本stem source/tests/doc。MAIN独占真实HOST CUDA；不训练、不新增Q、无CPU神经/物理、不改任何sealed来源，不新增独立审计器。prepare/consume的数组合同校验属于动作发布接口，不是另一审计工作流。

## 最小复用接口（已对接Arendt当前源码）

上游模块实际全名为 `p1_constructive_moving_history_encode_cuda_20261027_v1`。直接调用其 `load_task(path, expected_sha)`，返回且只允许 `gn_task,d2p_packet,q,task_id`；自有loader负责新export schema/task_contract/source/blind hash firewall。wrapper再核对H2、原normalizer、identity192 Q、模型字节身份及exact native float32数组。

不调用冻结blind planner的interface/load_export/build/plan_all/consume/旧encoder.load_task，不替换其module globals。只复用 `blind.execute_arm`（GN optimize/NativeCuda，D2P compile_arm/TorchBackend、原ledger/module guards）、`blind.selected`、`blind.Store`、source_closure与纯序列化。原execute_arm内部既有scoped ADTrace instrumentation会临时监测并恢复torch AD API，D2P内部scoped SVD计量同样沿用；新wrapper不通过monkeypatch注入任务、不新增这类global替换。它们要求单线程独占CUDA执行，不得并发共享模型。

额外模块pre-hook只读取真实action_encoder `(1,3,10)`与predictor `(1,3,192)`的float32输入，保存每个forward的完整token/latent，不修改输入/forward、不增加模型调用。失败时保留已journal的context attempts及原ledger counters，退出移除hook；cleanup失败不覆盖首错。

两显式输入路径固定：

```
results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_00.json.gz
results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_01.json.gz
```

实现期间不虚构SHA；MAIN actual完成后已固定：task00 `a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b`，task01 `101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74`，encoder source `246b9194fd31c874d7f0da67c4dec17365f93294e6ed460fe46e2509673e1b63`。prepare要求显式提供且与固定值相等，不能借参数追认新版本。export task_contract绑定MAIN encoder prepare `4386cf3babd40df0eb21abdf8bed52913596f012fb017ddb4986af96ddafa599`。两actual task ID分别`bf89d0d3c5738763c40a2d6481c6782df218de398d48bd7df33543d3444d21fd`与`6643e3cbd4322303fdd8eb158532d75e1c0f2ce720d13a8258d12a9527801b12`，本侧只读自有loader已加载成功，无模型调用。

两个slot固定0、1，各两臂GN→D2P，四分支完整分母不去重、不重试。raw history真实前5次[.1,0]、后5次[0,.1]，值为float32存盘精确值；不是logits/tanh输入，不再/100、不补零。使用原float32 mean/std逐元素subtract/divide，按两个time-major5×2 packet形成root action_history。

## 原预算口径（纠正156/root歧义，不改算法）

冻结 `p1_constructive_blind_planners_cuda_20261027_v1` source SHA `e56bee610f4eb412cc03fbe49b3df91054d0defe23d24b6fd9c77e6a7f1e9f27`。

| 实际每根新资源 | GN | D2P |
|---|---:|---:|
| predict | 78 | 66 |
| action_encoder | 78 | 66 |
| Jacobian | 4 joint H2 192×20 | 11 conditional H1 192×10 |
| 数学VJP方向 | 768 | 2112 |
| 实际grad API | 768 | 11（vectorized内部） |
| 实际jacobian API | 0 | 11 |
| SVD | 4 | 12 |
| linear_solve/proposal/selection | 各4 | 按原D2P合同（不伪称为0） |

GN 78=initial triple H2的6 + 4×(ADforward2 + 5trial H2的10 + selected triple H2的6)。156 predict是**两根GN总和**，或单根predict+action_encoder的合计神经入口数，不能称每根156predict。此wrapper保留原算法，每根768VJP；若MAIN真要求156predict/root，需要另订新算法协议，不能空跑padding/暗改次数。

D2P 66=2 root nominal/AD + 10×6 prefix/conditional nominal/AD/suffix/whole replay + 4 winner whole/segmented。两根总288predict+288action_encoder，其中GN156、D2P132。ADforward已包含，非等算力比较；原8图encoder/projector及source physics另行已支付，不计为免费，不将MAIN source reachable标签作planner选取条件。

## 选取冻结与非零全路径动作发布

所有候选/原数组由原kernel计算，durable journal先保存attempt/输入/候选/derivative，然后存own result/plan。GN保留全部35次evaluated H2、4 AD H2、20 line trials与4次stable选择；D2P保留10prefix/各自suffix/nominal/J/solve、13个replay receipt。winner动作原样tanh→raw，不clip/reselect。

own consume要求自己的schema/report外部SHA、全部4arm结果/无错误/model unchanged/source verified，准确artifact census/gzip EOF、own prepare closure与每个selection digest。逐项重算原float32 token，核对每次实际native input的完整latent与3token（GN78、D2P66/root），而非将重复receipt相等当作输入token算术。还核对所有GN候选/line projection/selection/triple replay和D2P prefix-suffix拼接、同Q cost与winner全/分段动作。原ledger/AD API attempt-return对齐预算，不从成功标签推断调用数。

四计划只供**未来NEW physical consumer**使用，不兼容旧schema/旧zero-history root，不继承旧endpoint。own consume返回完整commands/source_slot/task_id/task_export SHA/blindSHA/own selection seal/reportSHA。尚无物理结果，P1/P2 admission=false；moving history改动未被宣称能改善规划。

## 大小、测试与MAIN命令

新prepare<256KiB，输出总cap8MiB（失败字节也计入），disk floor200MiB；Store的原final reserve256KiB，无下载/删除/拷贝checkpoint。trace包含全部实际输入，真实压缩体积未知，超限必须失败保留，不调大cap或删数组。测试使用NumPy解析mock与RAM压缩，不调用torch/neural/physics，不落盘大fixture；不构成模型效果/actual history采集证据。

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
export PYTHONDONTWRITEBYTECODE=1
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_moving_history_planners_cuda_20261027_v1.py
TASK00_SHA=a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b
TASK01_SHA=101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74
ENCODER_SOURCE_SHA=246b9194fd31c874d7f0da67c4dec17365f93294e6ed460fe46e2509673e1b63
M=experiments.spectral_reachability.moving_history_planners_cuda_20261027_v1
P=results/moving_history_planners_cuda_20261027_v1_prepare01.json
.venv/bin/python -B -m "$M" prepare \
  --task-exports results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_00.json.gz results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_01.json.gz \
  --task-shas "$TASK00_SHA" "$TASK01_SHA" --loader-sha "$ENCODER_SOURCE_SHA" --output "$P"
# MAIN独立核验prepare SHA并设置PREPARE_SHA后：
.venv/bin/python -B -m "$M" verify --prepared "$P" --prepare-sha "$PREPARE_SHA"
.venv/bin/python -B -m "$M" run --prepared "$P" --prepare-sha "$PREPARE_SHA" --output results/moving_history_planners_cuda_20261027_v1_cuda_main01
# MAIN核验真实terminal report SHA后：
.venv/bin/python -B -m "$M" consume --output results/moving_history_planners_cuda_20261027_v1_cuda_main01 --report-sha "$REPORT_SHA"
```

交付验证：41 RAM/NumPy tests通过；四分支完整mock压缩产物含report为2,029,129 bytes（非真实模型压缩率保证）。另对MAIN两真实export执行一次只读build：自有prepare schema、2slots、113bindings，序列化23,924 bytes；没有写prepare/result、加载模型或执行模拟。实际新planning/physical均尚待MAIN。

本侧不创建实际prepare，不执行任何模型或物理；source/test/doc冻结后由MAIN review和实际执行。若上游loader变更接口必须在封存前协调，不能让已执行旧report追认新source。
