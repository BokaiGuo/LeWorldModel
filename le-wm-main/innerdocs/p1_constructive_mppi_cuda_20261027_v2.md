# Constructive v2：project-kernel-faithful MPPI数学与末轮nominal派发

仅NEW v2源码/tests/本说明，v1完整保留为探索性best-evaluated shooting，不混入论文MPPI主表。此次任务是落实本地参考kernel和nominal dispatch；**不是宣称论文要求的official/强MPPI baseline已完成**。无训练/Q拟合、网络、神经/物理执行；MAIN审查后独占prepare/CUDA/物理。

按experiment-code与两份required refs，先固定算法/成本/seed/预算，再实现日志和失败保留；不按效果调参，不产生额外图表/通用run目录。

## 1. 确切参考与官方边界

生产代码直接调用完整 `experiments/spectral_reachability/d2p_sampling_baselines.py::mppi`，SHA256：
`e1978044c81c6313b4b0a18fb7417cca4c46578860cc382909770641b4aafc04`。

不是只借名字或softmax函数；实际kernel负责initial评分、Gaussian噪声、box clipping、权重、updated nominal及每轮nominal评分。其原return为best-ever；**本runner不以该return派发**，而是保存为diagnostic，取最后updated nominal的已评分参数派发。这是相对于参考API输出策略的明确改动，但不改变kernel内部采样/数学更新/评分流程。

本地已审阅kernel自身与tests（含weighted nominal公式测试）；这些文件明确标注非official reproduction，未提供可认证的作者实现/commit或密度修正合同。本轮未联网取得新官方来源，不编造来源认证。这里的kernel-faithful只证明对该固定项目参考数学忠实，不证明faithful Williams/其他作者官方MPPI复现，也不自动补齐论文强基线缺口。

## 2. 参数域、原合法动作域、proposal变形

原blind两任务保持192维CLS、3状态2动作历史、相同identity192 Q/goal、原normalizer与冻结checkpoint。输入仅固定approved task_00/01 latent exports，不读sourcepack/witness/physical或sampling结果。

为了完整使用原box-kernel，明确区分两个坐标：

1. **kernel参数** x `[2,10]` float64，每坐标界限[-c,c]，c=.75sqrt5。initial=0。原kernel生成 `boxclip(nominal + epsilon)`，epsilon~N(0,.25I20)。
2. **合法模型logits** `l=P_ball(x)`：逐10维packet按原sealed screen.projected_logits投影到半径c的球，含原float32 inward rounding。
3. **真实命令** `u=tanh_float32(l).reshape(10,2)`；无额外clip、/100或normalizer替换。完整保留x、l、u，不能将三者混为同一随机变量。

所有候选、initial、updated nominal和fresh replay都走同一map；故原packet动作域严格保持。kernel更新**x**：`x_nom=boxclip(sum(w_i*x_i))`，不是平均投影后l_i；成本是复合函数J(x)=J_model(tanh(P_ball(x)))。这样不悄悄改变kernel更新数学。对应差异由测试明确覆盖。

box clipping会产生边界质量；球投影进一步多对一，实际l/u proposal不再是原Gaussian，不能将高斯原密度直接当其密度。无importance/proposal-density correction，无理论exact path-integral声称。这是有界参数化的普通项目kernel MPPI nominal baseline，不是官方density-corrected MPPI。

## 3. 首次run前固定参数与控制代价

- 两根seed20261029/20261130，NumPy default_rng/PCG64；两轮×16，不重启、不挑根。
- 温度1.0；logit-parameter噪声std0.5（裁剪前cov=.25I20）；不按结果改温度、噪声或iteration。
- J(x)=原Q fullH2终态mean-square；显式control-cost系数 **0**，无中间state项、无control-energy、无隐含自动cross term。单位为原latent MSE；1.0温度与其同单位。
- 权重是原kernel稳定 `exp(-(J-minJ)/temperature)`/sum，float64；有限极差下溢成0可接受，最小cost提供非零分母。
- 派发固定为 **index34：最后updated nominal**，即使比best-ever差也不能替换。kernel best return及其index/cost仅diagnostic。

## 4. 完整预算（不迁就66）

| stage | fullH2数 | predict/action_encoder各 |
|---|---:|---:|
| initial | 1 | 2 |
| 两轮proposal | 32 | 64 |
| 两轮updated nominal | 2 | 4 |
| 末轮nominal独立fresh replay | 1 | 2 |
| 每根总计 | **36** | **72** |

两根共144predict+144action_encoder。kernel自身35次cost、5批callback；kernel批量“attempted evaluations”若中途失败会计整批，**不等于真实forward已执行**，实际以模块predispatch/return ledger为准。没有空跑padding。只有一次独立fullH2 replay，不能伪造三重/segmented证据。

零AD/VJP/Jacobian/SVD/solve、新encoder/projector/physics；继承blind编码成本另列。与sampling64/D2P均不声称等总预算。MATH SDP、TF32关、inference_mode、模型参数冻结，原tensor hash前后相等且search/nominal replay精确相等才可封存commands。

## 5. 持久化与消费

每个callback在首次模型调用前封存完整x bank，逐次保存真实完整H2路径/参数投影/rawcommands/成本。updated nominal进入评分callback时核其与原kernel权重更新精确相等并保存weights；保留5批bank、35条evaluation和原kernel完整receipt。末轮nominal派发seal在fresh replay前。执行失败保留partial及kernel失败receipt，模块attempt/return分账；首错与final-model/source/journal/hash次错分别保留，无替补/重试。

prepare/verify绑定原blind exports、source/model/sealed kernel和NEW v2文件，不读取或绑定v1 MPPI/其他仍在写stem。Store新独占目录、8MiB总cap含partial、200MiB floor、256KiB终态预留。consume只读实际report外部SHA，按保存成本无神经重放完整参考kernel，逐callback/模块/packet数组核算、检查selection=last_updated_nominal_not_best及plan/row/commands哈希链。只读重放不产生模型调用。

## 6. MAIN命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。agent不创建prepare、不CUDA。MAIN review通过后：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_mppi_cuda_20261027_v2 prepare \
 --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_mppi_cuda_20261027_v2_prepare_main01.json
```

MAIN核实际prepare SHA再填变量：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_mppi_cuda_20261027_v2 verify \
 --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_mppi_cuda_20261027_v2_prepare_main01.json --prepare-sha "$MAIN_PREPARE_SHA256"
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_mppi_cuda_20261027_v2 run \
 --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_mppi_cuda_20261027_v2_prepare_main01.json --prepare-sha "$MAIN_PREPARE_SHA256" \
 --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_mppi_cuda_20261027_v2_cuda_main01
```

真实run terminal成功后：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_mppi_cuda_20261027_v2 consume \
 --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_mppi_cuda_20261027_v2_cuda_main01 --report-sha "$MAIN_REPORT_SHA256"
```

## 7. 工程测试/科学状态

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -s -p no:cacheprovider tests/test_p1_constructive_mppi_cuda_20261027_v2.py
```

纯NumPy/RAM mock，无CPU神经/物理/网络/tmp大fixture。包括完整kernel所有参数/receipt一致、独立权重公式、projection与参数坐标差异、故意构造initial零成本但仍派发较差末轮nominal、72实际预算、saved consume、失败partial和篡改拒绝。mock不是实模型效果；真实CUDA/物理、官方实现来源与论文强MPPI准入仍未完成。v1不改，任何v2 freeze后修改须另建版本。

本轮实际v2测试 **33 passed**，完整两根RAM journal/results/plans为719,808 bytes（约0.69MiB）。真实两份approved export只读build通过，113项source/input/model绑定，prepare预计22,752 bytes；未创建prepare、未访问任何v1结果。源码/tests/doc工程交付和MAIN验收、实际模型/物理验证、论文强基线要求是不同状态。
