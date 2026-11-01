# Moving：CEM / local nominal-MPPI 等 prediction 调用实验

本文件只定义一个新、固定、已知两根的校准实验，不回改旧结果。MAIN 审读并独占 HOST
CUDA 执行；作者本轮只实现源码与 RAM/mock 测试，不运行 prepare、模型或物理。
不读取 witness、sourcepack 内容、native outcomes 或其他物理结果来选择计划。

## 固定分母与本次算法

输入是原 off moving encoder report 与两个 task00/01，继续使用其原 goal、identity192
Q、非零 moving raw history、normalizer、冻结模型及原 packet 球 `0.75*sqrt(5)`。
loader 是 `p1_constructive_moving_history_encode_cuda_20261027_v1.load_task`，不是旧零历史
loader。原 source authority SHA 仅为继承的外部身份 token，planner 不打开 sourcepack。

原 `p1_moving_sampling_baselines_cuda_20261027_v1` 完成资产每根 CEM 为134 predict、
nominal-MPPI 为72，不满足新合同，不能重标为 matched。新 runner 不调用它的 optimize、
execute 或 run，仅复用冻结的 public task build/loader 与纯函数依赖。

固定 root-major 四行：root0 CEM、root0 kernel_faithful_mppi_nominal、root1 CEM、root1
kernel_faithful_mppi_nominal。每行 **3 iterations ×24完整H2候选=72候选=144 predict API**，
每次实际 predictor batch1，因此每行也是144 predictor forward samples。全次各576；
另列 AE576、pred_proj576，不能相加成2304次 world-model 调用。AE/projector各处理3个
context rows，所以每行各432 rows，不冒充432 samples。没有初始/updated nominal评分、
winner replay、nominal replay、padding 或失败补跑。

| 项 | CEM | local nominal-MPPI |
|---|---|---|
| 两根PCG64 seeds | 20261028 / 20261129 | 20261029 / 20261130 |
| 初始化 | mean0、std1 | nominal parameter0、noise std0.5 |
| proposals | Gaussian logits，按packet投影 | Gaussian围绕nominal，先boxclip parameter，后packet投影 |
| 更新 | 每批稳定top4，projected logits mean/std(ddof0)，std clip[.05,1] | temperature1，exp(-(cost-min)/1)；加权**box parameters**，boxclip |
| 发布 | 全72中稳定argmin(cost,index) | 全72中稳定argmin(cost,index) |

CEM 保留旧 elite4、方差与无平滑/无incumbent注入。MPPI 保留旧噪声、温度、参数坐标
加权及投影方式，但因新合同禁止额外 nominal 评分，不运行旧完整 optimizer：用新有限
24×3 loop，并发布已评分 sample winner，**不发布未评分最后nominal**。第三轮分布更新
仍保存但不再评分。这是 nominal-guided MPPI sampling 的新有界变体，不是旧版
“final updated nominal dispatch”，更不是官方或 density/importance-corrected MPPI。
保留旧 arm 名用于明确算法族 join，必须同时携带本新 schema/contract，不可混称旧配置。

## 真实路径与预算拦截

每候选只运行一次 fullH2：每步 concatenate 两条真实 incoming normalized packets 和
当前 packet，AE 一次、`model.predict` 一次，滚动完整3state/2action history。raw10为
NumPy float32 tanh 的精确输出；normalizer减除一次，无二次 /100。只按terminal原
`core.model_cost` 计算 mean192 identity Q，没有control/intermediate penalty。

使用旧 NativeCuda 的只读 fullH2 rollout基础类，但重写自身 `_step` 与 guard，不修改模型
方法、模块 globals 或任何旧源码。API在实际调用前记 attempted；predictor/AE/pred_proj
用真实 forward pre/post hooks记录。prehook严格batch1/context3 shape；禁止encoder和
image projector进入。模块与API返回前CUDA synchronize，返回账不提前。保存每个实际
模块/API输入输出的float32 shape/dtype/完整values。predictor和pred_proj处于同一活跃
API ticket，AE在API前独立计量。每packet开始检查四项144 cap，各dispatch仍独立检查；
失败不退款、不重试、不通过空转补预算。

Pauli的 `moving_prediction_budget_contract_20261027_v1.declaration` 固定旧六方法/12组；
不能声明本新四组。按MAIN明确批准，本runner使用自己 prepare contract的四组144 cap，
不调用/修改Pauli模块，不伪造其他八组或旧sealed configuration。当前会话没有线程通信
工具，因此未声称已直接发送给Pauli。预算匹配的主张仅是**本次两种采样方法之间的
actual API calls与batch1 predictor samples**，不是与GN/D2P、AD方法同FLOPs、同墙钟或
equal total compute。

原 acquisition 与原8encoder/8image-projector成本只继承一次，单列
`inherited_encoding_once`；本runner新encoder/image-projector/AD/J/VJP/SVD/physics均0。
不将共享编码重复乘四，也不称整个pipeline编码免费。

## 封存、失败与 saved consume

独立 NAME/schema/prepare/source closure；绑定冻结旧helper、原moving encoder完整产物、
原task、模型文件、安装的官方LeWM源码以及本stem三文件。官方LeWM SHA为
`ccc2ac2880880f85ea71a5baa2152ba97e3f6dc5cb546d27570795aa6020d3d2`。
不依赖其他正在编辑的新stem。prepare小于等于256KiB，output目录必须NEW。

64MiB包含全部候选结果、计划、gzip journal、partial与report；其中512KiB预留final。
每次write检查200MiB磁盘floor；run启动要求64MiB+200MiB。900秒从Store创建开始，
覆盖load/search及dispatch/write检查点，不承诺中断阻塞中的native调用；final cleanup/hash/
report尽力保存不受超时停止。HOSTCUDA-only、MATH、TF32off、禁autocast，0CPU neural
fallback，无训练/拟合/Qfit/阈值调整。失败保留首错及secondary、预置四行和真实attempt/
return；未开始行保留。可能无法写入磁盘时不保证最后report可落盘；stdout保留错误。

每轮bank在首次候选前durable，所有72候选完整prediction/contexts/raw/logits/parameters/
cost、更新、winner seal保留。实际module dispatch attempted在执行前flush/fsync；每次
returned在同步后durable，journal使用单流gzip Z_SYNC_FLUSH，不因压缩丢字段。
完整成功正好10个非report产物：binding+journal+四份result+四份plan。

consume输入外部report SHA，严格固定四行/产物全集、prepare重建与final TOCTOU；纯数组
重新生成所有RNG proposals、CEM/MPPI更新、packet-ball/raw映射、full-H2 history与Q，
并检查完整因果journal、每packet实际API/module输入输出桥、144×4实际返回计数和
每组144 predictor samples。选择只能引用已评分候选；不调用模型重新评分，也不从native
结果重选。输出 `.consume.rows` 提供 `source_slot/arm/task_export/task_sha256/blind_sha256/
task_contract_sha256/commands/logits/prediction/model_terminal_cost/selected_index/
selection_sha256/whole_run_report_sha256`，供MAIN另建物理selection使用。

## MAIN命令

在 `/home/gbk/code/LeWorldModel/le-wm-main`。先完整审读三文件和RAM测试；下面的prepare/
run只由MAIN执行，本次作者没有执行它们。所有路径须NEW，SHA由MAIN从实际文件取值，
不能填本文件尚无的实际结果。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_moving_prediction_matched_sampling_cuda_20261027_v1.py

.venv/bin/python -B -m experiments.spectral_reachability.moving_prediction_matched_sampling_cuda_20261027_v1 prepare --output results/moving_prediction_matched_sampling_cuda_20261027_v1_prepare01.json

.venv/bin/python -B -m experiments.spectral_reachability.moving_prediction_matched_sampling_cuda_20261027_v1 verify --prepared results/moving_prediction_matched_sampling_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA

.venv/bin/python -B -m experiments.spectral_reachability.moving_prediction_matched_sampling_cuda_20261027_v1 run --prepared results/moving_prediction_matched_sampling_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA --output results/moving_prediction_matched_sampling_cuda_20261027_v1_cuda_main01

.venv/bin/python -B -m experiments.spectral_reachability.moving_prediction_matched_sampling_cuda_20261027_v1 consume --output results/moving_prediction_matched_sampling_cuda_20261027_v1_cuda_main01 --report-sha REPORT_SHA
```

RAM测试只证明程序合同的mock路径，不证明真实模型预算、实际控制改善、泛化或P1通过。
即使MAIN将来完整576/576返回，也只是已知两根的预算匹配校准，不是独立确认样本。

本轮交付验收为48项RAM测试：含四臂576API/576samples完整mock saved-consume，所有72候选
与历史/RNG/更新重建，非法batch、receipt篡改、故障不退款、同步/日志故障、hook注册中途
失败清理与原始错误保留。无磁盘mock目录，无实际prepare、CUDA或物理产物。
