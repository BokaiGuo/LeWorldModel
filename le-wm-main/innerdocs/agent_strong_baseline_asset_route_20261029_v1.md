# GRASP / LeWM / PlanWAM：已有实物与下一可运行入口

2026-10-29；本次仅新增本文，冻结后不改。以下路径除特别注明外相对 `/home/gbk/code/LeWorldModel/le-wm-main`。只读本地源码、JSON与SHA；未运行模型/物理、网络、下载或统计拟合，不新增另一套registry、consumer或批准文件。

## 结论与选择

**优先复用已有 saved consumer，而不是再跑强基线。** 已有GRASP模型→真实物理→RGB比较，也已有LeWM同身份beam/GN/D2P/CEM/nominal-MPPI十行比较；后者已经保存，不需要重复实现。PlanWAM目前只有作者协议/表值登记，所检查登记未提供可认证本地官方驾驶执行入口，不能拿同名PointMaze脚本替代。

选取入口不基于收益：固定两known roots、固定全部方法及原资源，负结果不删。两个比较不是同一cohort，不将它们合并排名。本文不将任何作者值与本地值算提升率，不授权新运行，不开启P1/freshness。

|方法|作者报告（独立证据栏）|本地实际产物（独立证据栏）|下一入口与精确限制|
|---|---|---|---|
|GRASP|`results/p2_reported_baseline_table_20261027_v3/export/report.json`及`docs/p2_grasp_official_contract_20261027_v1.md`；原论文/表locator沿原registry，不重新转录数字|`p1_moving_grasp_baseline_cuda_20261027_v1`有模型报告和own consume；extended physical/RGB已进入comparison v3的固定六方法×两根分栏|可立即saved consume；是LeWM-backbone bounded virtual-state/soft-dynamics适配，不是官方DINO-WM表级复现，`official_equivalence=false`。原表checkpoint/config/data/seed/完整预算配对仍缺|
|LeWM及其本地规划器|同registry内LeWM原文登记，原表版本依现有登记为2603.19312v3；不能用旧v1摘要补表级协议|已有sampling/gradient/GN/D2P、beam、物理和RGB；matched comparison有两identity partitions、每组完整五臂|可立即消费已有十行比较；仅beam/CEM/nominal-MPPI各144 forward/root，GN78/D2P66另有AD。不同搜索support/数值验证边界保留，不称等总算力或完整官方baseline|
|PlanWAM|同registry的2610.11382v1条目；缓存协议见外层`/home/gbk/code/LeWorldModel/docs/D2P_REPORTED_BASELINES_LEWM_PLANWAM_20261026.md`|本轮沿这些明确登记未定位官方本地执行闭包；本地效果unknown，不填0|没有可据此认证的NN运行命令。缺immutable官方推理source revision、checkpoint/config/许可、4camera×4history与ego输入、NAVSIM-v2/HUGSIM split及evaluator绑定。其驾驶指标与PushT raw10/native success不同，comparability=false|

GRASP官方缓存已有 `results/p2_grasp_official_contract_20261027_v1/sources`；现有协议登记给出官方GRASP commit `64f988aebf04ad9f674f9821bc6d2b9e22879105`及le-wm-grasp commit `94be3c42aa2f0867c6cb1e38bda498b293b6921e`。这不补齐论文表到本地checkpoint/任务的配对。旧GRASP H1 saved audit与本次moving历史3帧适配不是同一协议，不能混称重复确认。

## 本次实际读到的封存锚点

以下SHA在本轮对实际文件重算；读取报告不是重新执行其strict闭包。

|相对路径|字节数|SHA256|
|---|---:|---|
|`results/p1_moving_grasp_baseline_cuda_20261027_v1_cuda_main01/report.json`|8150|`7593940c1503416f42ce87a5d8135e8512763121e027c0ef98dbe614bfb31f91`|
|`results/p1_moving_grasp_baseline_cuda_20261027_v1_consume_main01.json`|27512|`e6a23d0ca7f48bc782d0187673af84ea9cf4ade3fedb985be6771e5d4e4f7fbf`|
|`results/p1_moving_baseline_comparison_20261027_v3_main01/comparison.json`|239641|`96cf27b143bb1a839edf7bc1e4de02f4a97e433c6475c4c731c78af76300d32e`|
|`results/agent_baseline_matched_rgb_comparison_20261028_v1_MAIN01.json`|450199|`eb661bfd89b75b4865b37ba517de4412f015472928a9ed3c2b76ec15367ed128`|
|`results/agent_p2_equal144_baseline_cuda_20261028_v1_cuda_MAIN01/report.json`|7060|`7eeaa059d58895545340458455ddcaabceda4a813a41051e1a2c60f22bc10228`|
|`results/p2_reported_baseline_table_20261027_v3/export/report.json`|1464049|`52e687811e6d6f728ab3eda60836ea05eff11f36da2bb4af975d1dd19ef380f1`|
|`experiments/spectral_reachability/agent_baseline_matched_rgb_comparison_20261028_v1.py`|21366|`bc61c9f70ccd89cb6c0316773c528741cec4a618a3034775cf398fbef39193b5`|

GRASP模型报告status=`moving_grasp_model_plans_sealed_not_admitted`、errors=[]；原consume为`own_moving_grasp_commands_verified_not_admitted`。旧comparison v3保12模型行/12物理行/24RGB图，`official_reproduction=false`、`equal_total_compute=false`。matched comparison保10行、原生0成功/10失败、同身份两组完整，`equal_FLOPs=false`、`equal_total_compute=false`、`scientific_admission=false`。这些是既有报告字段，不是本轮新实验或本轮独立strict认证。

更新的equal144模型报告也已存在且status=`equal144_plans_sealed`、errors=[]，故不应再提出重写144预算调度器。其现成源/doc是`agent_p2_equal144_baseline_cuda_20261028_v1`；物理handoff也已有`results/agent_p2_equal144_physical_handoff_20261028_v1_physics_MAIN01/report.json`。本文未重新消费该模型/物理闭包，不据此猜测终态RGB或新增比较已完成。其576predict账含两根×两方法；谱臂额外2J/384VJP/4SVD/2solve，API/sample匹配不等于AD/FLOPs匹配，也不是GRASP等预算对照。

## MAIN下一条可执行命令：不重跑NN/物理

cwd为上述inner仓库。已有consumer重建原全部saved输入、身份、指标和资源，只stdout，不写receipt。不是新对照实验，也不需要重跑CUDA。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_baseline_matched_rgb_comparison_20261028_v1 consume --output results/agent_baseline_matched_rgb_comparison_20261028_v1_MAIN01.json --report-sha eb661bfd89b75b4865b37ba517de4412f015472928a9ed3c2b76ec15367ed128
```

如MAIN需要先闭合GRASP本身，已有独立真实命令（不伪装为上述五臂之一）：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_grasp_baseline_cuda_20261027_v1 consume --output results/p1_moving_grasp_baseline_cuda_20261027_v1_cuda_main01 --report-sha 7593940c1503416f42ce87a5d8135e8512763121e027c0ef98dbe614bfb31f91
```

两个命令的CLI及saved-only分支已读；本轮未执行，避免重复MAIN已完成的全量消费。如需新实验，应先明确与GRASP共同任务/动作support/预算轴的前瞻设计，而非事后对异构报告评分；现有`agent_p2_strong_baseline_saved_summary_20261028_v3`已有matched合同，不重复实现。family/algorithm不是必须相等的实验控制字段。

## 原始process停点：不能拟合代替

`eval.py:87–98`在运行时建立process，逐列`StandardScaler().fit(col_data)`，goal列复用对应processor；104行还有自动CPU设备分支。**本任务不调用eval.py，不fit，也不把源码可重新fit等同取得原始frozen process。** 原始资产需包括与对应checkpoint/data/eval provenance绑定的实际mean/scale、flags、dtype、字段顺序与字节hash，不能借公共常数或另一次训练统计。已有本地adaptation所用normalizer及已完成物理记录原样保留；缺官方process阻塞的是官方原协议认证/新inverse authority，不反写成“旧物理从未执行”。

最近实际metadata路线为`innerdocs/agent_hilewm_normalizer_asset_route_20261029_v2.md`：镜像快照仅列README/config/weights或大训练archive，没有列独立process/scaler，取得的小config也无相应mean/scale。范围限定该快照，不自证全历史穷尽；镜像revision声明不能独立签官方authority。Hi已有真实owned加载/RGB→high-FIRST→low-CEM，不沿用旧readiness中“未取得Hi对象”的过期结论。本次不再重跑Hi诊断，不下载大包寻统计。

PlanWAM的输入/evaluator也不能用这里的process补齐。作者值仅context；完整官方运行仍待其本身资产，不造启动命令。PushT native success须沿原agent+object及angle判定，object-only或latentQ不能替代。

## 验证与冻结范围

本次实际完成：读取上述JSON状态/分母与现有源码CLI；stdlib逐文件重算上表7个文件SHA/长度；核对既有两份比较doc、equal144 doc、官方协议缓存登记及eval.py具体fit位置。只新增本文；无新source/tests、无pytest、无全量consumer、无权重反序列化、无NN/物理/网络执行。文档路径与SHA表一致性另作离线检查，交付时给本文SHA；测试/元数据读取不等于科学通过。所有旧结果、dirty改动及冻结代码保留，unknown不填0。
