# Moving 两旧根：固定 H2 kinodynamic beam 本地对照 v1

仅新 stem 的实现交付；MAIN 独占审计、准入、prepare、HOST CUDA 和物理。固定已知 moving 两任务，非独立确认，P1/P2=false。本地 bounded search 不是论文官方复现，不声称与 GN/D2P 等搜索域或等总算力。这里没有物理结论。

## 事前冻结合同

合法非零历史装载使用原 moving encoder 的 `load_task`，经 moving planner 的非零 `load_export` 校验。不是旧 blind zero-only schema；不改全局、不借旧根 material/verify。输入只为批准的两 task export（slot0、1 顺序）、native192 context/goal、public normalizer；不打开 sourcepack、blind NPZ、certificate、witness 或物理结果。sourcepack SHA 仅 provenance 字符串，不据此自认其内容验证。

原 encoder report SHA `e7fb554327b3bb5a96190f6291f8450b2b0723cd981817715cfb7e50aa266bf6`；sourcepack provenance SHA `c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6`。两 export 为 `results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_00.json.gz` / `task_01.json.gz`，SHA 分别 `a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b` / `101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74`；固定路径和 SHA，不通过扫描选择替代根。

复用 `d2p_kinodynamic_program_search` 的 bank/context/cost 验证和 `p2_matched_composition_20261027.schedule(16,4)` 固定 H2 语义。不能直接调用旧 `search_programs`：它会跨 H1/H2 选择。新纯内核完整枚举16首段，按 `(H1 native MSE, index path)` 保留4，再枚举4×16后段；仅从64个 H2 终点按 `(terminal MSE, path)` 选择。H1再好也不能成为短赢家然后补零。没有 flat arm，没有自适应省预算/全库优势声明。

旧 P2 的四个位置 bank **不同**。本合同事前固定旧位置0、1的 bank 分别映射 moving opaque slot0、1，无 native-root lookup；literal bank 顺序和值取自 `results/pusht_p2_matched_composition4_v1_20261027_cuda/p2_matched_protocol.json` 的 `/prepared/0/bank`、`/prepared/1/bank`，原文件 SHA `38679b9a7339f822eb4d8cb9af2d5cf794c61badee6213021ac14434a0234720`。每个16项，每项同一原始 float32可表示 raw2 重复5次；不重新采样、不 arctanh/tanh 重建原数值、不裁剪。运行仅 hash 原协议文件，算法只用源码 literals，不读取原协议的旧 context/goal/结果。测试只核 bank 指针原值。两 bank 都包含超出 D2P `.75*sqrt(5)` packet logit L2 ball 的项；contract 输出所有 logit norms/超域数量。D2P 可逐时刻改变动作，旧 bank 为离散 constant packets，故两个方向的域差异都存在，不能称等搜索域。

## 精确预算与三路径

每根 planning=16+64=80 predict；fresh segmented=2；fresh whole=2；共84。两根168 predict、168 action_encoder、168 pred_projector；每次 batch1、3 latent tokens和3 normalized action tokens（每token10坐标），两根各504 incoming tokens及504 pred_projector rows。83个高层 dispatch/根，最后 whole dispatch 含2次真实模型预测，不能当1次。0 AD、0 SVD、0 encoder、0 image projector、0 physics、0训练。继承的原 encoder 8 encoder/8 projector另计，不宣称总流水线视觉成本为零。加载/hash/CPU数组比较并非神经 fallback，FLOPs未知不捏造。

三路径为 search 路径终点、从原 root fresh 两次 `TorchBackend.step`、从原 root 一次 `CudaReplay(raw10)`（内部2步）。三者**完整 context exact** 相等，不只 cost 接近；失败不放宽容差/重试。保留原3 states/2 action历史：raw warmup 前5 `[.1,0]` 后5 `[0,.1]`，float32 public mean/std；每次预测先追加 outgoing token形成3动作，再滚动为后2动作。whole 与 segmented 分别从 root 重建，不能从 search缓存 latent偷开始。

实际 module hook 在每个 action_encoder/predictor/pred_proj 入口先计 attempt，在 return 后计 completed。每资源每根84硬上限；所有神经入口 CUDA/no_grad。保存实际 action3、latent3 输入和 pred_proj native successor，以 gz journal 独立核对，禁止 image encoder/projector 入口。实际底层 predictor入口计为 predict；pred_proj单列，不冒充 image projector。MATH SDP，TF32 false，frozen model before/after tensor hash和源码/任务/模型文件绑定；没有 CPU 神经 fallback。

## Own schema / saved consume

命令 `prepare / verify / run / consume`。prepare NEW单文件≤256KiB；run NEW目录≤8MiB（含 partial/journal/final），200MiB reserve。原 Store 保留256KiB final空间，每写检查；无覆盖/重试。所有资源尝试、失败点及尚未开始的第二根保留。只有所有两根完成、源码/模型未变、gzip正常finalize及资源全分母84/根才释放 commands。partial gzip若无footer仍保留，own consume拒绝。

report rows 各有 `source_slot(int), arm, task_export, task_sha256, task_id, blind_sha256, commands(raw10x2), path, terminal_cost, resources, result_file`（失败未完成字段不伪造）。own consume 外部 report SHA→prepare 重建→文件全 census/hash→完整gzip→独立重新执行纯 fixedH2 schedule，逐dispatch重建成本、所有16/64分母、实际module receipts、非零history滚动、全部三路径。其 `plans` 顺序固定0、1，保留上述身份与 literal commands，schema独立，不冒用旧物理 consumer合同。下游需 MAIN 新接线/明确准入；本 worker不运行。

## MAIN 精确命令

从 `/home/gbk/code/LeWorldModel/le-wm-main` 运行，`<PREPARE_SHA>` / `<ACTUAL_REPORT_SHA>` 由 MAIN 实际输出取得，不预造。目录父路径需存在，输出都必须 NEW。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_moving_kinodynamic_baseline_cuda_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_kinodynamic_baseline_cuda_20261027_v1 prepare --output results/p1_moving_kinodynamic_baseline_cuda_20261027_v1_prepared01.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_kinodynamic_baseline_cuda_20261027_v1 verify --prepared results/p1_moving_kinodynamic_baseline_cuda_20261027_v1_prepared01.json --prepare-sha <PREPARE_SHA>
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_kinodynamic_baseline_cuda_20261027_v1 run --prepared results/p1_moving_kinodynamic_baseline_cuda_20261027_v1_prepared01.json --prepare-sha <PREPARE_SHA> --output results/p1_moving_kinodynamic_baseline_cuda_20261027_v1_cuda_main01
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_kinodynamic_baseline_cuda_20261027_v1 consume --output results/p1_moving_kinodynamic_baseline_cuda_20261027_v1_cuda_main01 --report-sha <ACTUAL_REPORT_SHA>
```

RAM tests仅数组fixture与已有saved任务/bank读取，不创建 tmp/result/prepare，不运行模型/物理；不是CUDA或科学验证。实现交付实跑 **46 passed**，`--help` 正常；包含完整 own consume 内存端到端、10类损坏/partial拒绝、资源/cap/floor/验签重建。没有实际 prepare/模型/物理产物，源码冻结SHA由交付消息提供（不写入自身避免循环）。可执行阻断为 MAIN审查、可用HOST CUDA与原资产/hash不变、空间 floor；之后另行物理与配对评价仍未执行。本实现不解除P1前置或认证强基线公平性。
