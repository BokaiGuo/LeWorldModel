# Constructive matched-ridge GN：独立校准 variant

只新增本 stem source/tests/doc；旧 planner/GN/D2P、exports和结果均不改。只接受 MAIN 已批准两个真实 task exports，不读取blind NPZ、sourcepack、证书、witness、physical结果或旧candidate结果。作者只做NumPy/RAM mock，MAIN独占prepare与HOST CUDA。不是已知修复成功、P1准入或等预算对照。

## 先确认算术与边界

GN原实现对 `A/sqrt(192),r/sqrt(192)` 做SVD，再分母加 `.002`，因此其方向求解
`min mean((Aδ+r)^2) + .002||δ||²`，等价未归一化λ=.384。

本variant显式 `lambda_mean=.002/192=0.000010416666666666666`，未归一化λ=.002。D2P ridge解的目标是 `||Ku-r||²+.002||u||²`，报告ownV为整个目标除192；两者的**正则系数单位一致**。无界线性解可由 normal equation `(A.T A+.002I)δ=-A.T r`独立验证。

这不等于两个算法求同一个问题：GN为当前joint H2的192×20导数及增量δ，随后endpoint投影+非线性五尺度选择；D2P是各prefix处H1的192×10导数，在以zero suffix为中心的球内做精确有界ridge，并按stage+ownV选prefix。相同系数不能消除线性化点/动作维度/球约束/目标/计算量差异。不得把λ匹配当作理论或物理收益。

## 明确实现

新 `optimize(task, backend, meter, *, lambda_mean, updates, scales)` 接显式参数并限定冻结值。不调用旧 `gn.optimize`，不patch gn.DAMPING/SCALES/AD globals；复用旧 `gn.Meter`、`screen.NativeCuda`、`line_trials`、`single/verified`和浮点packet-ball投影。复用 scoped ADTrace（同GN API cap）只在HOST run内计量实际AD接口并finally恢复，不改sealed源码/常量。

两批准export均zero2×10起点，四更新，每轮五scale `[1,.5,.25,.125,.0625]` 全部新fullH2评价；incumbent参与稳定minimum选择，每轮选中后新nominal/whole/segmented三重replay，不提前停、不调参。原identity192 Q、独立goal、native完整latent/history、rawfloat32 tanh、normalizer和模型不变。

每根78predict+78actionencoder（已含8ADforward，不能再加），4J(192×20)/768数学VJP与768grad API、4SVD/solve/proposal/selection。两根156/156+8J/1536VJP。新增0image/physics/training；历史8图成本单列，不宣称免费pipeline。相对原GN调用预算相同，但相对D2P66/root不等预算。

## 来源、资源和失败

prepare显式绑定两个批准export path/fileSHA、真实taskID、source_slot/blindSHA、冻结旧helper/静态import闭包/本新三文件、模型/config/normalizer；prepare/run/final重验。无outcome输入。source_slot只是公开元数据配对，不伪造native root SHA。HOST cuda:0、offline、MATH SDP、TF32 off、frozen eval；禁止新encoder/projector，无CPU neural fallback。

复用封存planner Store的durable gzip Z_SYNC_FLUSH/fsync及8MiB cap，final预留256KiB，启动需208MiB且每写保留200MiB。这里不承诺2MiB：保存全部J原始行、chained J、所有contexts/调用/失败，不能为压缩预算丢审计资产。两根GN-only输出预计约1--4MiB，8MiB是硬上限而非预先已认证actual大小；mock容量只证明工程mock。prepare<256KiB，只hash，不复制大source/model/PNG。

所有两根行先创建。失败保留first/secondary、已付费attempt/return、partial gzip/未启动分母和先前完整plan；不重试，不凭partial计划启动physical。模型未load时model_unchanged=None。complete闭包后才commands_released=true；consumer必须外部reportSHA并验证own schema/完整artifact/hash/源封存/完整gzip/两根counts/choice/plan seal，拒绝partial。consume属于本runner的释放检查，不是独立saved科学审计。

## 给MAIN physical的接口

最终report schema=`p1_constructive_matched_ridge_gn_cuda_20261027_v1.report`，status=`matched_ridge_model_plans_sealed_not_admitted`；须errors=[]、source_verified/model_unchanged=true、commands_released=true及MAIN外部SHA。rows顺序source0,1，每行含 `source_slot, arm="matched_ridge_GN", blind_sha256, task_id, commands, selection_sha256, result_file, plan_file, resources, AD_API_resources`。未成功commands=null，不能dispatch。新plan和result stem=`root_{slot}_matched_ridge_GN`，与旧plan完全分离。

可直接用 `/rows/0/commands`、`/rows/0/source_slot`、`/rows/0/blind_sha256`、`/rows/0/arm`（root1对应rows1）作为physical显式selection pointers。需要新selection并MAIN绑定actual reportSHA；旧两arm handoff模块硬编码原ARMS，不可伪装复用。本模块consume返回schema自身`.consume`、status=`own_model_commands_verified_not_admitted`，rows为自身封存plans，不执行物理。

## MAIN命令

项目根 `/home/gbk/code/LeWorldModel/le-wm-main`，先review/tests，然后MAIN独占prepare：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_matched_ridge_gn_cuda_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_matched_ridge_gn_cuda_20261027_v1 prepare --task-exports results/p1_constructive_blind_encode_cuda_20261027_v1_cuda_main01/task_00.json.gz results/p1_constructive_blind_encode_cuda_20261027_v1_cuda_main01/task_01.json.gz --task-shas 714f471761278fa654542d4bb0878ce18e3c304df6158005af46d2129533265a 70865d56f9b66f0072b2ad7de01cc09d9a9275047925e188104a13d5182ce1f2 --output results/p1_constructive_matched_ridge_gn_cuda_20261027_v1_prepare01.json
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_matched_ridge_gn_cuda_20261027_v1 verify --prepared results/p1_constructive_matched_ridge_gn_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 timeout 600s .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_matched_ridge_gn_cuda_20261027_v1 run --prepared results/p1_constructive_matched_ridge_gn_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA --output results/p1_constructive_matched_ridge_gn_cuda_20261027_v1_cuda_main01
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_matched_ridge_gn_cuda_20261027_v1 consume --output results/p1_constructive_matched_ridge_gn_cuda_20261027_v1_cuda_main01 --report-sha ACTUAL_REPORT_SHA
```

测试只用RAM合成backend/压缩，验证算术等价、λ原始尺度、完整预算/raw J/history/所有scale/失败停止/ownplan/零sealed globals漂移。actual任务prepare可以只在RAM build核验，无作者GPU运行；任何实际收益必须MAIN后续报告。
