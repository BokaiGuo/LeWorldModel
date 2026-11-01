# P3 root-only native progress extension

独占交付 source/tests/innerdoc；旧 local signal/runtime/MAIN 文件不改。代码导入、prepare、verify、consume 只读 saved/NumPy；仅 MAIN 审查后授权 run 加载 HOST CUDA。没有 worker 神经/物理实验，没有训练、拟合、调阈值或 outcome 选根。测试只是 RAM 工程证据。

## 复用与边界

已检查 `pusht_full_packet_compiler_20261027.py` 的 `bounded_ridge`/`spectral_bank`/`compile_arm`，以及旧 horizon residual 和 goal-patch compiler。复用真实 `bounded_ridge`，原 damping=.002、joint L2 cap=sqrt(5)*.75。旧 `spectral_bank` 只接2/10维，旧完整 compiler 要重新获取子节点 J 且用 tanh logits；不冒称它们直接适配原始 raw J。沿用 sealed signal 的 `NativeSignal` 真实模块观察器、数组 blob、链式 Store、Ledger、冻结模型 hash 和 HOST CUDA loader；不改旧 globals、不 monkeypatch NumPy SVD。

上游固定 report：`results/agent_p3_local_signal_cuda_20261028_v1_cuda_MAIN02/report.json`，SHA `5c776ea604f615081b9f50b92648e22e2612ef46d8443bf1949f69341ed4864d`。prepare 必须调用原 strict `signal.consume`；不接受仅声称通过的 status。原两 known tasks、原 identity Q/goal/action normalizer 保留；四个 task×method 决策直接消费，16个候选完整分母。

这是 **一个真实 root extension**，不是 depth3 树。原 program_depth_limit=3 保留为 requested_depth，executed_extension_depth=1。H3 是同一个 root 对15原始动作的联合 J；三个预测包不等于三个已重新规划的 child。未来 child 必须另付本节点 H1/H3 J；缺适配器则停止。reconfiguration/composition 没有真 objective/编译器，明确 blocked/null，不制造假动作。

## 事前固定生成与 replay

每个 task×method 用选定 H 的原始 `J[192,10H]`、其自身 nominal 和原 Q/goal 求一次 bounded ridge。J 是零 raw 处导数，绝不解释为 logits。求解 direction 在 float64 中除以 `max(1,max(abs(direction)))`，再乘固定 `[0,1/3,2/3,1]`，最后一次 cast float32 得四候选。该 raw-domain bounded line-search 是本文件新生成规则，不宣称等于旧 tanh generator 或原 Proposal 的完整搜索。常数/动作映射/源码/全部输入在 prepare 前封存，不依本轮分数换模式。退化 J、相同动作、零/负结果全部保留；不去重、不补预算。

每个候选 whole replay H 包，然后从相同 root 独立 segmented replay H 包（每包从真实 float32 context 重建 CUDA tensors）。完整 rolling latent `[1,3,192]` 与 between-history `[1,2,10]`，每步 raw→原 subtract/divide→AE→predict→predictor→pred_proj。两种 replay **每个边界全 context 精确相等**；原 Q terminal mean-square 决定 winner，tie 按候选 index；无物理 outcome 输入。成功仅表示模型回放工程完成，不给 physical release。

严格 saved consume 重算 ridge/actions/原 Q 成本/winner，验证所有 raw/normalization/AE/predict/predictor/pred_proj float32 buffer、paid operation shape、parent-child 状态/历史闭合、双 replay 全路径、journalchain、attempt-return 资源账、模型与源码前后绑定。它不重新运行神经，因此不是对已保存神经计算的独立数值复算/科学认证。

## 实账与失败

实际本轮 fixed H1、adaptive H3，各 task 四候选，每候选两遍回放。每 task fixed8、adaptive24 prediction；两 task fixed16、adaptive48，共64 API/64 samples/192 context tokens。AE/predictor 各64 calls/64 samples/192 tokens，pred_proj64 calls/192 rows；4次 solve/4次真实数组 SVD、4 program compile、32 replay、16 metric。新增 AD/VJP/J/encoder/projector/physics 为0（实际未调用）；上游已付4J/768VJP 等账另列 historical，不抹掉、不加成新费用。上游 acquisition unknown 仍 null。per_branch 输出各轴实账、nested inclusive wall（不可相加作总时长）、动作碰撞数与全部候选成本；unknown CUDA ns/backward engine/FLOPs 为 null。

不声称两方法等算力或性能优势。事前工程 cap：wall900s、总输出256MiB、report reserve16MiB、disk floor200MiB、CUDA allocated3GiB/reserved4GiB，禁 TF32/autocast，SDP MATH。predispatch debit，失败不退款、无 retry；工程失败保留16分母及未派发/null，不换动作。模型 state_dict 与包括 nonpersistent 的全部 buffers 前后 hash；prepare 完整绑定原 signal 源闭包、模型文件、generator、chart、三独占文件、实际 report/journal/binding。

saved 验证费用不混入 generation native ledger：每次 prepare/verify 调原 strict signal consumer，另做4次 NumPy SVD；完整 run 的前/后 verify 共8次，报告单列 saved_verification。完整 consume 的 verify4次加候选生成重算4次，共另8次；原信号采集的历史4次 SVD、这次生成实际4次、验证附加次数分别列明，不宣称只有4次总 SVD。验证中断时未完成 pass 的实际工作量未知记 null，不填0。这些都不产生新模型调用。

为保持复用 solver 且不修改 globals，array_svd 计数在已封存、恰好调用一次 SVD 的 `bounded_ridge` helper 入口扣账；该计数准确，但它的 inclusive wall 包含该 helper 的投影/二分等数组工作，**不是独立 SVD kernel 计时**。solve/array_svd 两个 nested wall 不得相加。

## MAIN 精确命令

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。本 worker 不执行 run。prepare 也不运行模型；新路径必须不存在，禁止覆盖旧结果。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_p3_native_program_20261028_v1_test.py tests/agent_p3_local_signal_cuda_20261028_v1_test.py tests/agent_p3_runtime_contract_20261028_v1_test.py

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_native_program_20261028_v1 prepare --output results/agent_p3_native_program_20261028_v1_prepare_MAIN01.json
```

取上条实际输出 `sha256` 为 `PREPARE_SHA`（不得编造未来 SHA）。以下仅 MAIN 执行：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_native_program_20261028_v1 verify --prepared results/agent_p3_native_program_20261028_v1_prepare_MAIN01.json --sha256 "$PREPARE_SHA"
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_native_program_20261028_v1 run --prepared results/agent_p3_native_program_20261028_v1_prepare_MAIN01.json --sha256 "$PREPARE_SHA" --output results/agent_p3_native_program_20261028_v1_cuda_MAIN01 --authorization MAIN_APPROVED_HOST_CUDA
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_native_program_20261028_v1 consume --output results/agent_p3_native_program_20261028_v1_cuda_MAIN01 --sha256 "$REPORT_SHA"
```

REPORT_SHA 只能取这次 run 实际返回值。不得复用上游 signal report SHA 作为新 report SHA。全进程已 import torch 时，新测试仍检查精准 import interception 和独立冷 subprocess，不删除 sys.modules。旧 sealed runtime 的全局冷导入断言不修改，三文件联合回归请从新 Python 进程启动。

## 未完成/科学阻断

交付前实际验证：三文件联合回归127 passed /13.33s；新文件在进程已导入 torch（仅 import、未构造或执行模型）后36 passed /5.72s。已对本次真实上游 report 运行原 strict saved consume，并实际用4个 task×method 对应的原始 J 做 NumPy 生成：16候选、各分支4个不同 float32 动作缓冲且全在原动作域；实际新增4 program_compile、4 solve、4 SVD，模型/replay调用0。这不是神经 replay、候选收益或物理结果，未生成研究报告/示例资产/prepare 文件。RAM fixture 包含完整 saved-consume roundtrip、17类变异拒绝、身份/分母/冷导入/失败不退款及reconfiguration阻断；无真实模型测试。三独占文件 whitespace diff-check 无警告。

真实 extension CUDA 尚未由 worker 执行；MAIN 完整审查/prepare/verify/CUDA/strict consume 仍待执行。缺 fresh roots/tasks/history 认证、独立 P1 与适用 P2 admission、实际 child-local signal/generate/reconfiguration/composition tree adapters、完整树资源预算、独立物理后果。P1/P2/P3/freshness/full_C3/scientific_admission/physical_release/优势 flags 全 false。原 PushT 成功定义包括 agent+object；此 latent-Q 模型成本或 object-only 投影不能证明完整任务失败。固定两根结果不支持因果泛化。
