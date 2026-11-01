# Moving/off fixed-H2 四策略预算调度 v1

实现范围：已批准外层 prereg `docs/D2P_MOVING_BUDGET_SCHEDULER_PREREG_20261027.md`，SHA256 `8e8cd4091dbaed5991dc119638ffdc00c8aed0852d0ed4c54cc6b1b3a82009bf`。这是 known-root model-only calibration，不是完整 C3、在线反馈或科学 P1/P3 通过。仅新增本 stem 源码/tests/本文；不修改 sealed 文件。MAIN 独占真实 prepare/CUDA/物理授权。

## 固定输入和防火墙

使用 `moving_history_planners_cuda_20261027_v1.load_export` 与 moving encoder 自有 loader，绝不走旧 zero-only gate。两实际 task SHA 固定为 `a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b` / `101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74`。真实 raw 历史前5 `[.1,0]`、后5 `[0,.1]`，逐位 float32，原 mean/std、无再次 `/100`。原 off goal、identity192 Q、逐包 `.75sqrt5` logit ball 不变。

`material` 外部 pin handoff `52c475ed499b077f261517bc592252b208926919909674cebf68fa4029cc8310`，其 projection 回指原 compiler `cd8432c06288577d25d9d9b92c7eb3673de592499ee64cae93238156f3f19aa0`，不能把两者混同。逐 source_slot/arm/task/blind 身份 join 两根全部 GN/D2P；通过原完整 plan 文件验证 selection seal，再核 handoff projection。算法只接收 literal commands/logits，不读取旧 model cost/物理结果选择锚点。原 saved source authority 只核 root/context/goal 身份，privileged sidecar 不进入 backend。

原 encoder report SHA `e7fb554327b3bb5a96190f6291f8450b2b0723cd981817715cfb7e50aa266bf6`、prepare `4386cf3babd40df0eb21abdf8bed52913596f012fb017ddb4986af96ddafa599` 与8个原 native192 NPY逐位核对；原 sourcepack/prepare 与源码、测试、本文、prereg、静态 import closure、weights/config 全部封存。prepare 不复制模型、图片或旧大报告，只存引用与哈希。禁止读新 overlay fidelity、未来 winner 物理/latent 来选择参数。

## 完整四策略和精确预算

先执行两根共同零动作 nominal H2 + fresh joint AD H2；两包间 graph 不 detach。对 raw float32 动作逐行 `torch.autograd.grad`，再乘一次 tanh chain（零点为1），完整 J `[192,20]`。原 transition `_start`/rolling graph 复用 frozen NativeCuda，新增独立 scoped meter。`K=QJ` 薄 SVD，`sigma>sigma_max*1e-6`；`r=Q(goal-nominal)`，chi 为 span 外残差能量比（有限值 clamp[0,1]），`w=(1-chi)*||r||²`。零 residual chi0/w0；零rank非零 residual chi1/w0。SVD异常/非有限直接拒绝，无调 epsilon。

固定顺序 U_uniform、D_geometry、S_swapped、N_no_signal_uniform。U/N `[10,10]` blocks，N allocation 入口禁止传 weights/J。D 为 prereg half-up 配额，S 只交换 weights；sum w0则均分。每策略先 `[0,1]` 交替5轮保障各根5blocks，再轮询剩余配额；不按成本改顺序。每block3 H2，各方法完整60候选。所有候选先冻结规则/池哈希：index0 zero、1原GN、2原D2P、其余42个 PCG64 ball样本；seed `20261027+101*slot`。每随机包依序 draw normal10 再 uniform，float64方向/半径计算，logits只cast一次float32，NumPy float32 tanh。锚点保留原GN float32/D2P float64 logits与literal float32 raw，沿用原 `core.check_prediction` 的 `cap+1e-12` 边界判定（实际D2P约1e-15 roundoff），不是修命令或放宽旧域；random logits严格不超cap。无projection修复；anchor不合法或零方向直接拒绝，不重抽。

每方法 search120predict + 每根 whole H2/segmented H1+H1 两次独立 replay 共8predict =128。四方法512 + shared8 = **520 predict/AE/predictor/pred_proj**；action context tokens1560。**2J、384数学VJP方向、384显式grad API、2SVD、0新solve**。叶raw gradient hook单独记384 backward callback；不是kernel/engine内部总数，内部未知不得填0。新 visual encoder/projector/physics为0；模型eval/frozen/CUDA-only，MATH SDPA/noTF32，绝无 CPU neural fallback。

全局、策略128和每根64/94上限均在dispatch前检查；shared每根4。原8/8编码、历史GN78/根+768方向、D2P66/根+2112方向、总288历史predict独立引用，不加进新520、不删除成本。独立部署逻辑成本 U/D/S136和N128不是实际重复调用，也非walltime实测。无总算力匹配/速度优势结论。

所有候选成本 `mean((Q@(terminal-goal))²)` 保留，stable `(cost,index)` argmin；search/whole/segmented full context精确相等，history以实际float32归一化逐包重算，非receipt重复等价。每paid block10/20仅报告已付费best-so-far，不补调用。D−U为主差、D−S/D−N为次差，保留两根逐根值。失败禁止删根/补根/退款/自动重试，paired summary不可用。

## Durable trace / 限额 / 输出

独占 flat output：`binding.json`、`banks.json.gz`、`events.jsonl.gz`、`panel.json.gz`、`report.json`。journal每条gzip Z_SYNC_FLUSH后fsync文件与目录；先attempt再dispatch。每packet保存actual full context，module hooks保存AE/predictor/pred_proj完整float32输入/输出；raw/J完整精度通过 shape/dtype/raw_sha256/base64 无损字节保存，不降精度。partial packet/J row/module数组即时入journal，不等root finally。原模型tensor hash前后核验。execution首错与final source/model/hook/journal错误分别保留，final异常不覆盖首错。

全panel wall600s含验证/加载；CUDA allocated3GiB/reserved4GiB，实测峰值/elapsed/disk下界写limits receipt。16MiB总output、1MiB final reserve、200MiB diskfloor，开始前需至少216MiB。预计完整产物约8–14MiB（RAM随机全精度trace压缩尺寸测试是工程上界fixture，不保证真实产物一定可装下；触cap需工程停机，不裁数组或调限额）。prepare硬上限256KiB，若真实材料超过即拒绝，不自动提限额。

consume仅saved算术：重建bank/配额/全部候选cost/argmin、J/SVDsignal、8plan seal、实际520输入history与native模块数组receipt、独立replay、完整ledger。own schema `moving_history_budget_scheduler_cuda_20261027_v1.consume`，status `four_policy_commands_verified_not_admitted`，每plan含 `source_slot/root_sha256/task_id/blind_sha256/arm/commands/selection_sha256`，可供未来NEW physical consumer绑定外部实际report SHA。所有输出 `P1_admission=false,P3_admission=false,independent_confirmation=false,compute_advantage=false,walltime_advantage=false`。

## MAIN 命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`，主机独立审读本三文件后执行；下面未在agent端运行prepare/CUDA。

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_moving_history_budget_scheduler_cuda_20261027_v1.py

OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.moving_history_budget_scheduler_cuda_20261027_v1 prepare --handoff-sha 52c475ed499b077f261517bc592252b208926919909674cebf68fa4029cc8310 --anchor-report-sha cd8432c06288577d25d9d9b92c7eb3673de592499ee64cae93238156f3f19aa0 --encoder-report-sha e7fb554327b3bb5a96190f6291f8450b2b0723cd981817715cfb7e50aa266bf6 --output results/moving_history_budget_scheduler_cuda_20261027_v1_prepare01.json

# PREPARE_SHA 由上述真实receipt提供，不能填虚构hash；两个命令使用同一真实值。
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.moving_history_budget_scheduler_cuda_20261027_v1 verify --prepared results/moving_history_budget_scheduler_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.moving_history_budget_scheduler_cuda_20261027_v1 run --prepared results/moving_history_budget_scheduler_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA --output results/moving_history_budget_scheduler_cuda_20261027_v1_cuda_main01
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.moving_history_budget_scheduler_cuda_20261027_v1 consume --output results/moving_history_budget_scheduler_cuda_20261027_v1_cuda_main01 --report-sha ACTUAL_REPORT_SHA
```

RAM/mock tests不用模型/torch autograd/模拟器、不创建tmp或新research结果；这些测试通过仅表示实现合同的数组与计账fixture成立，不是实际520 forward、有效J或科学证据。后续前置仍为MAIN review/prepare/真实CUDA/own consume、新物理与fresh endpoint；原P1/P2及完整P3/独立任务确认不足不能由本任务绕过。

## 本次交付验证（非实际CUDA结果）

限BLAS单线程、禁止pytest cache/bytecode的51项RAM测试通过：完整四方法520账、2J/384方向/2SVD、独立winner双重replay、全部native输入/输出chain、稳定同池前缀、half-up配额及swap、零rank/零residual、SVD/非有限拒绝、partial首错与blocked分母、root/method/global cap、CLI与consume内存端到端。测试不执行torch模型或真实AD。只读实际素材的 `material(...)` smoke成功核156个closure/input bindings、两task、两个45候选bank；prepare序列化35,797 bytes，未创建prepare/output。

RAM独立随机化所有blob（保留全原精度）后的trace+panel/banks压缩fixture为12,535,084 bytes，低于15MiB非final写入阈值；fixture本身仅存RAM，无research结果文件。这不是实际CUDA输出尺寸/模型一致性的证明。生产路径仍硬拒绝16MiB/200MiBfloor/600s越界，保留部分数组与账，后续槽记 `blocked_not_dispatched`，不以工程失败作理论负结果，不裁精度或自动重跑。
