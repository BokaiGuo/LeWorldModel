# Moving-history sampling baselines CUDA v1

## 有界实现交付

将既有 CEM 与 **nominal MPPI 本地参考核**适配到固定旧 moving 两根；不是新 benchmark、独立确认、
论文默认 protocol 或官方复现。MAIN 独占审读、实际 HOST CUDA、物理及科学准入。worker仅实现本
stem source/tests/innerdoc，并运行 RAM/mock tests；不 prepare、不加载模型、不运行物理、训练、
Q/readout fitting或 CPU neural。P1 始终 false，不因本地结果声称等总算力或优胜。

使用 experiment-code skill 的预算预声明、持久化和 mock faults；本轮不采用其超授权的训练/结果
图/正向结果要求。Goal API 已 get_goal，残留已撤销且 blocked 的旧 observed-FD review objective；
create_goal(本实现目标、无token预算)被 unfinished goal 拒绝。未假完成旧目标绕过限制。

## 冻结输入和合法非零接口

唯一模型任务来源：
`results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/report.json`
外部 SHA `e7fb554327b3bb5a96190f6291f8450b2b0723cd981817715cfb7e50aa266bf6`。
build检查该报告完整status、errors[]、原model hash未变、源验证true、encoder/projector实际8/8返回、
0predict/actionencoder/AD/physics；封存原report全部12项artifact（binding/journal/8latents/2tasks）。

按原 source_slot0/1加载 task_00/01，task SHA、task_id、blind SHA、encoder prepare SHA、MAIN
manifest SHA及每根四image receipt严格相连。仅调用冻结的
`p1_constructive_moving_history_encode_cuda_20261027_v1.load_task`，保留其 task export/schema验证。
新 own interface要求原mean/std、identity192Q、原cap，以及真实 raw history前5[.1,0]、后5[0,.1]
float32；两包 token=(raw-mean)/std，**不再除100**。来自实际的三帧 latent不填充、不重编码。

原 sourcepack authority SHA
`c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6`
仅以 MAIN 已批准 provenance token写入新plan；本 planner不打开sourcepack、state、certificate、
witness、physical outcomes，也不独立核验sourcepack。source/root在此是 blind task identity，
不是通过隐藏物理state进行规划或 native-root映射。继承encoder的manifest-only admission边界。

不调用旧 `blind_planners.interface/load_export/build`（要求zero）、旧sampling/MPPI build/run/
consume；不改全局whitelist、不 monkey-patch数学模块。只复用已封存的纯数学函数与NativeCuda。
本新 own load_export/build/plan_all/run/consume和 model_plan schema负责完整接线。

## 预声明算法及精确调用预算

每根按 CEM→nominal MPPI执行；也可在 prepare前固定只运行其中一个 arm，仍含两根、相同seed/
参数/历史/goal，不是按结果选择arm。run不得覆盖prepare的arm。

| arm | 固定算法 | fullH2计算/root | predict/action_encoder/root | 两根总计 |
|---|---|---:|---:|---:|
| CEM | 4轮×16，elite4，初mean0/std1，无incumbent injection | 64候选+3赢家replay=67 | 134各 | 268各 |
| kernel_faithful_mppi_nominal | 2轮×16，temp1，noise .5 | 1初始+32提案+2更新nominal+1独立replay=36 | 72各 | 144各 |
| 两arm一起 | 不padding、不替换 | 206 total | — | **412各** |

CEM复用 `p1_constructive_sampling_cuda_20261027_v1.optimize/choice/elite_update`：NumPy PCG64
seeds[20261028,20261129]；高斯logit经每packet球投影，elite按(cost,index)稳定排序，投影后的
elite均值/std(ddof0)，std clip[.05,1]、无smoothing；所有64候选稳定global argmin。赢家新算
nominal/whole/segmented三次完整H2，终态/原Q成本/动作/contexts exact才封plan。不是修改采样预算。

MPPI复用 `p1_constructive_mppi_cuda_20261027_v2.optimize/choice` 和完整未改
`d2p_sampling_baselines.mppi`：seeds[20261029,20261130]，float64参数box[-cap,+cap]^20，
协方差.25I的高斯先boxclip；权重exp(-(J-minJ)/1)，**参数坐标**加权更新nominal，而非平均投影
logits。各次回调完整[1,16,1,16,1]保留，共35H2；实际dispatch最后nominal index34，不用best-ever
sample替代，即使best sample的成本更好。最终nominal再独立整H2一次，**不声称MPPI也有三次replay**。
原kernel best/receipt只作诊断并在consume中完整回放，不能借best return变成shooting。

两个arm同一动作图：每packet logit球cap=.75sqrt5，NumPy float32 tanh后reshape(10,2)；每次
H1由真实三state/两incoming-action context开始，outgoing token追加后滚动，第二包保留旧最后
incoming token+第一包新token。不在prefix重新取zero。完整H2末态使用原native192 identity Q
mean-square，不加入control/intermediate/cross term，不拟合新metric。

MPPI是本地heuristic：无importance/projection-density correction，boxclip/球投影改变分布，
不是density-corrected或author-pinned官方MPPI。CEM与MPPI forward预算本就不同；更不与GN/D2P
按全资源等配。AD/J/VJP/SVD/solve/newencoder/projector/newphysics均0。NumPy CEM elite4次/
稳定全局选择1次和MPPI两次softmax更新、kernel best比较是实际算法CPU开销，保留receipt但不是
“数学开销0”或FLOPs等配。模型load一次/源码模型hash/编译guard/选择/replay也不能被predict计数替代。

actual module entry/return计数覆盖predictor和action_encoder；batch=1，完整context3tokens；两arm
共412个各类module batch-sample，predictor输入latent tokens=1236。pred_proj包在predict路径里，
未独立计量，不能把此账说成全部模块/FLOPs计量。MATH SDP、matmul/cudnn TF32=false与原moving
encoder相同。原模型/config/normalizer和完整源import闭包在prepare及run前后重hash。

历史编码成本**一次性单列**8encoder+8projector（每root4+4），共享两arm不重复计两遍，绝非
整个pipeline免费。若分两次独立invocation报告各会引用同一历史receipt；跨run汇总须按reportSHA
去重，不得将它们误算成16次实际新增编码。模型载入/文件hash/host时间不纳入encoder次数。

## 输出/partial/own consume

prepare<=256KiB，不复制大protocol或模型。NEW run总cap8MiB，final reserve256KiB，启动需
208MiB可用；每写保留200MiB floor。复用已验证 lossless gzip level6 +每事件Z_SYNC_FLUSH/
fsync的Store，所有native packet、候选、kernel参数/更新、赢家replay上下文不薄化。
attempt在module prehook记，completed在返回及CUDA synchronize成功后记；失败不退款/不重试，
inference_mode且NativeCuda guard禁止encoder/projector入口。只有CUDA设备，不CPU fallback。

每arm逐候选保存，result/plan独占输出；错误时保留先前events、failed_partial与未开始行；后续
model/source/hash/journal失败另记，不冒充complete。最终写盘失败不能保证report存在；async
CUDA failure可能已返回但计completed=false，这是确认完成口径，不是假装未执行。
run成功才commands_released=true，status=`moving_sampling_plans_sealed_not_admitted`。
本协议无新增wall/RAM/GPU-memory cap（继承前向runner的allocation preflight）；需要MAIN外层
作业限时，不声称predict cap能限制单次kernel hang或总耗时。

own consume要求外部完整report SHA、prepare/model/source前后绑定、精确artifact集合、两根/arm
顺序、task/blind/goal/action identity、全量gzip结束和每实际module attempt/return。CEM重建全部
proposal RNG/elite链及67条H2；MPPI重放完整kernel、35评价/两nominal更新及独立replay；每条saved
path检查真实两incoming tokens滚动、终态成本/动作及final seal。它是本runner的可执行消费闭包，
不是另派审计agent，不调用旧白名单consumer或任意payload shortcut。

公开接口 `consume(...)["rows"]` 每行含 `source_slot/arm/task_id/blind_sha256/task_export/task_sha256/
task_contract_sha256/commands/logits/prediction/model_terminal_cost/selection_sha256/whole_run_report_sha256`，
以及moving encoder/sourceauthority SHA、with_targetFalse/history语义。MPPI额外带选中index34/parameter/
replaypolicy。**本轮不写physical consumer**；后续统一moving adapter必须独立复原真实非零warmup、
task/root/goal/frame provenance，不能用旧zero branch执行。未做任何新物理效果声明。

## MAIN 命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。worker只运行首项RAM/mock测试，无实际prepare：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_moving_sampling_baselines_cuda_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_sampling_baselines_cuda_20261027_v1 prepare --arm both --output results/p1_moving_sampling_baselines_cuda_20261027_v1_prepare01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_sampling_baselines_cuda_20261027_v1 verify --prepared results/p1_moving_sampling_baselines_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_sampling_baselines_cuda_20261027_v1 run --prepared results/p1_moving_sampling_baselines_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA --output results/p1_moving_sampling_baselines_cuda_20261027_v1_cuda_main01
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_sampling_baselines_cuda_20261027_v1 consume --output results/p1_moving_sampling_baselines_cuda_20261027_v1_cuda_main01 --report-sha ACTUAL_REPORT_SHA
```

单arm先行时，prepare改 `--arm CEM` 或 `--arm kernel_faithful_mppi_nominal`，使用分别全新prepare/run
路径；run不传--arm，由prepare固定。两根都保留，不需其他worker基线齐套，不择根/seed/参数。
tests纯RAM数组/假backend，不是模型实际结果；冻结后MAIN独占review/准入/实际CUDA与物理。
