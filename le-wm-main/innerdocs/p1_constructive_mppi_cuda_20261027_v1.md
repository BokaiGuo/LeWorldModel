# Constructive blind MPPI-inspired shooting v1

新独占源码/tests/本说明；不改sampling或旧审计交付。MAIN独占prepare/verify/CUDA/consume/未来物理。agent仅编写和NumPy/RAM mock，无模型、physics、下载、训练、Q拟合。使用同两份批准blind latent exports及原normalizer、192维Q/goal、3状态2动作历史、H2；不读witness/sourcepack或任何真实效果。

## 首次run前固定的算法

slot0/1 seed分别20261029/20261130（20261027+slot*101+2），PCG64 default_rng。温度 **1.0**，logit噪声尺度 **0.5**，初始nominal零 `[2,10]`，固定两轮、每轮16个候选。无CLI温度/迭代/种子旋钮，不根据结果改参数。

每轮float64 Gaussian(nominal,.5)后，复用 sealed screen.projected_logits，逐packet L2 cap=.75sqrt5，保留既有float32 inward rounding；raw命令为NumPy float32 tanh(logits).reshape(10,2)，不额外clip或/100。logit球不是均匀raw box。全部16个候选成本均为完整H2相同identity192 Q terminal mean-square；不加隐含control penalty。

复用现有 `d2p_sampling_baselines.softmax_weights`：`exp(-(cost-mincost)/1)`/sum，float64稳定计算；极差下溢为零允许，最小cost权重保证分母非零。按权重平均**实际已投影logits**，再作原packet-ball projection，作为下轮nominal；不是平均未投影noise后再裁剪。保存全部weights与两轮更新（末轮更新仅留存，不执行未评分nominal）。

从全部32个真正评分候选按(cost,index)稳定全局argmin；不选未评分weighted nominal。先封存winner，再一次独立完整H2 replay，比较rawcommands/terminal/同Q成本精确一致。没有三重/segmented replay证据，不能制造相应字段。后续consumer需支持该明示单replay合同，不能把它冒充gn.verified。

## 为何不是官方MPPI/不能直接调用旧kernel

已审查现有纯NumPy `d2p_sampling_baselines.mppi`：它额外评分initial与每轮updated nominal，并采用box clipping。两轮×16会是1+2×(16+1)=35次H2，另winner H2共72predict，**不符合66预算**。本实现仅忠实复用其稳定weights数学，明确采用新预算/packet-ball/selection合同，不宣称exact旧kernel复用或官方论文实现。

没有proposal-density importance correction或自动control-energy/cross term，且best-evaluated selection不同于标准path-integral nominal dispatch，所以命名 `mppi_inspired_shooting`。其负结果不排除faithful MPPI。

## 资源、封存与失败

每根32候选×2=64predict/action_encoder，winner一次H2再2，共 **66各/root，两根132各**。无初始/nominal额外评分、无padding、无batch折算。零AD/VJP/Jacobian/SVD/solve，新encoder/projector/physics零；继承blind编码成本另列，不能称等sampling64或D2P总算力。

复用NativeCuda实际forward pre/return hook、CUDA synchronize与Meter predispatch66 cap；frozen model hash前后相等，MATH SDP、TF32关闭，无CPU fallback。搜索/replay都处于inference_mode。不调optimizer、不训练。

build只读原approved exports和source/model byte hashes，不调用sampling的optimize/consume、也不读其实际report。自身、tests、doc和weights helper并入prepare绑定；verify完整重建。来源source_slot/blindSHA/taskID直接列入plan/report。每轮bank在cost前封存，candidate逐项持久化，update在全部16cost之后，winner在全部32后/replay前。实际调用与partial数组保留gzip Z_SYNC_FLUSH/fsync；执行首错与final-model/source/journal/hash次错分开，不自动重试、不替补根。

Store独占新目录，总 **8MiB**硬cap含partial，**200MiB**floor，256KiB终态预留。prepare<256KiB，不复制旧大报告或模型。mock大小不保证真实压缩率；超限fail-closed，不能放宽续跑。consume读外部真实report SHA，重建全部RNG/weights/更新/selection及逐event全H2 history/成本/ledger/单replay，整体闭合后才交付commands；不执行物理。

## MAIN命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。agent不创建prepare，不runCUDA。MAIN独立review后：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_mppi_cuda_20261027_v1 prepare \
 --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_mppi_cuda_20261027_v1_prepare_main01.json
```

核实际prepare SHA后赋值 `$MAIN_PREPARE_SHA256`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_mppi_cuda_20261027_v1 verify \
 --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_mppi_cuda_20261027_v1_prepare_main01.json --prepare-sha "$MAIN_PREPARE_SHA256"
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_mppi_cuda_20261027_v1 run \
 --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_mppi_cuda_20261027_v1_prepare_main01.json --prepare-sha "$MAIN_PREPARE_SHA256" \
 --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_mppi_cuda_20261027_v1_cuda_main01
```

真实terminal report成功后独立核SHA，再：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_mppi_cuda_20261027_v1 consume \
 --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_mppi_cuda_20261027_v1_cuda_main01 --report-sha "$MAIN_REPORT_SHA256"
```

## 仅工程验证

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -s -p no:cacheprovider tests/test_p1_constructive_mppi_cuda_20261027_v1.py
```

测试为全尺寸192维NumPy解析数组替身与RAM store，无CPU神经、tmp大fixture、physics或网络；覆盖精确66/132计数、两轮因果更新、权重稳定/shift invariance/underflow、packet投影/动作绑定、单replay错配、全saved consume、篡改拒绝和失败partial。旧kernel复用范围仅weights，预算不偷换。

本轮实际 **32 tests passed**；两根完整RAM journal/results/plans产物645,934 bytes（约0.62MiB）。只读build通过，113项source/input/model绑定，prepare预计22,318 bytes，未创建prepare或真实结果。真实CUDA/replay相等与未来物理仍待MAIN运行；个人goal保持active等待MAIN验收，不将工程交付当科学成功。

已按experiment-code及两份required refs预声明run与错误留存；用户只允许本stem，所以不生成额外图表/通用run目录，也不采纳“结果必须非零/按效果择优”的模板。mock成功不表示真实CUDA/物理成功、MAIN验收或P1准入。交付后冻结，任何修改另建版本。
