# Constructive blind planners — MAIN审阅后独占HOST CUDA

仅NEW同stem源码/tests/本文。使用experiment-code技能实现RAM数组回归、真实算法内核调用、durable attempts及partial失败保留；本交付不调用模型/物理、不prepare实际任务、不改sealed源。无训练、下载、Q拟合或P1准入。

## 输入与最短接线

只接受MAIN显式批准的Arendt **task export路径+每文件外部SHA**。严格校验已冻结export/contract的字段集合及source_slot一致性、native_root_sha256只能为null；此处确实依赖最终export schema，不放宽校验。算法数组仅由最终 `encode.load_task(path,sha)` / `task_interfaces` 返回，只取 `gn_task/d2p_packet/q/task_id`；source_slot不用于planner join或native root查表。opaque顺序slot + export SHA + task_id为join，native root映射由MAIN evaluator外部持有。绝不调用encoder的build/verify/whitelist/load_blind，也不读取sourcepack/certificate/witness/PNG/NPZ或目录发现输入。

prepare时固定MAIN批准encode源SHA `c3b071de1ad0b322e1fff77a716550f42014f0303a6b5fa09271fd4e55609b4a`，并绑定其tests/doc、全部静态local import闭包、关键本地model/autograd源码、模型/config文件SHA、task export SHA；不携带旧两根material。随后每次verify重新构造全部绑定；encode代码漂移必须重新审批prepare，不动态接受未冻结接口。

task验证真实native192 float32 history/goal、3帧/2 between tokens、identity192 Q、原packet cap和原checkpoint，以及与 `TorchBackend` 隐式使用的public mean/std精确float32相符。保留真实zero raw history，normalized history不为零。不新编码：继承8image成本单列，不宣称pipeline免费。

## 两臂，明确不等预算

每个批准task按固定GN→D2P顺序执行；算法均看同一新task/goal/Q，不共享导数，不利用未来物理标签挑选。**matched仅指task/模型/动作域，不是相同计算或相同目标函数。**

| 每根 | line_search_GN | full10_conditional_D2P |
|---|---:|---:|
| predict / action_encoder | 78 / 78 | 66 / 66 |
| 上述predict已包含ADforward | 8 | 11 |
| Jacobian | 4个192×20 fullH2 | 11个192×10 H1 |
| 数学VJP方向 | 768 | 2112 |
| 实际grad API固定cap | 768 | 11（vectorized Jacobian内部batched API） |
| functional.jacobian API | 0 | 11 |
| SVD | 4 | 12 |

GN另计4 linear_solve/proposal/selection。四更新、五scale、零初始化，复用 `gn.optimize`、`gn.Meter`、`screen.NativeCuda(model,device,task)`。不调用GN/diagnostic旧material/verify。

D2P复用v2 `compile_arm(packet,q,backend,journal)`、`TorchBackend(model,device)`，packet**只有root_context/goal**。固定10prefix，每prefix新suffix J，beta1不调。不能复用旧 `NeuralCap`：它硬编码旧Q arm名和双Q预算；本driver独立actual module hooks和predispatch math ledger。不能调用compiler旧source_inputs/verify/execute_cuda四旧根路径。

两根总新资源：GN156/156，D2P132/132；合计288/288仅供总支出记账，**不是算法等预算对照**。ADforward已在这些数内，不能漏计/再加。backward API调用数与数学VJP数分别保留，不作FLOP/kernel等价声明。

## HOST-only执行与失败保存

只有MAIN run进入HOST CUDA preflight，显式cuda:0，offline loader，frozen eval，参数/缓冲CUDA；MATH SDP、matmul/cudnn TF32均false。模型文件与tensor before/after SHA均验。禁止encoder/projector新调用和未预算AD接口；实际AD API attempt/completion单列。module completion前CUDA同步；错误无退款无重试；任何根/臂失败即不启动后续，未开始分母保留。

8MiB全产物cap含partial和final，启动保留200MiB+8MiB，每写保留200MiB；final预留256KiB。gzip journal每事件Z_SYNC_FLUSH/fsync，底层文件和目录fsync；broken stream不补假footer，partial不冒充success。所有source task/model运行结束再验；退出非零保留partial。wall由MAIN外部timeout控制，不抢占/自动重启。

每臂完整模型选优后先exclusive写result和commands plan，再进行下一臂。plan为 `selected_pending_global_closure`，只有最终complete report及MAIN外部report SHA经consume才可释放给**另一个MAIN物理executor**。本driver无物理命令；不凭已保存partial plan抢跑。consume复核source、模型、所有artifact SHA、gzip EOF、两臂分母、模型选择与commands seal；不是独立saved机制审计。

## MAIN完整命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。MAIN已完成8image编码，下列两export SHA由MAIN外部提供且已只读核验task_id；不传sourcepack/证书。`PREPARE_SHA/ACTUAL_REPORT_SHA` 须替换为后续实际64位SHA。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_blind_planners_cuda_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_blind_planners_cuda_20261027_v1 prepare --task-exports results/p1_constructive_blind_encode_cuda_20261027_v1_cuda_main01/task_00.json.gz results/p1_constructive_blind_encode_cuda_20261027_v1_cuda_main01/task_01.json.gz --task-shas 714f471761278fa654542d4bb0878ce18e3c304df6158005af46d2129533265a 70865d56f9b66f0072b2ad7de01cc09d9a9275047925e188104a13d5182ce1f2 --output results/p1_constructive_blind_planners_cuda_20261027_v1_prepare01.json
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_blind_planners_cuda_20261027_v1 verify --prepared results/p1_constructive_blind_planners_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 timeout 600s .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_blind_planners_cuda_20261027_v1 run --prepared results/p1_constructive_blind_planners_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA --output results/p1_constructive_blind_planners_cuda_20261027_v1_cuda_main01
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_blind_planners_cuda_20261027_v1 consume --output results/p1_constructive_blind_planners_cuda_20261027_v1_cuda_main01 --report-sha ACTUAL_REPORT_SHA
```

若MAIN encode输出目录命名不同，只替换两个实际export路径，外部SHA必须对应。prepare不运行模型，不读取encode report的路径列表。禁止用mock export作为actual任务。

## 证据边界

冻结前最终回归33 passed（0.94s；在MAIN已跑28项基础上补5项严格export envelope拒绝回归）。两个MAIN actual export的外部SHA和task_id均一致；完整prepare build仅在RAM完成，21757 bytes、107 bindings，无prepare/result/tmp落盘，无torch/gymnasium/pymunk导入。最终3文件现在冻结，后续独立audit仅新增文件。

测试仅RAM合成线性backend和故障注入；复用真实optimize/compile_arm可证明接线及记账，不证明CUDA数值、任务成功或P1机制。实际GPU计划、physical成功/失败及native root映射待MAIN执行并另审；source构造已用可达witness不等于planner利用witness。没有科学通过阈值、不作理论失败或强基线排名结论。
