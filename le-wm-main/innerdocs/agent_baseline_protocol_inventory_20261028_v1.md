# 基线协议与可复用产物清单（saved-only）

2026-10-28，Jason独占新增。只读本地记录；MAIN审代码并调度实际CUDA/物理。本文不修改任何旧source/results/共享主表，不产生新效果数值，不下载、不加载神经模型、不执行物理。依据documentation-and-adrs技能，重点记录可执行入口及不能跨越的协议边界，而不是再实现一个readiness工具。

## 1. 当前可采取的动作

- **先消费现成结果，不重跑known-root beam。** 新matched-domain beam的两根CUDA和物理均已完成；旧beam16x4不应覆盖它，也不能把新结果塞进旧六法comparison。
- **GRASP、LeWM-backbone sampling与gradient已有可调用的冻结runner。** 本次离线readiness重核三工作流139/129/121项closure及report artifacts，通过的是MAIN dry preflight，不是新模型运行授权、官方复现或科学准入。
- **Hi-LeWM、PlanWAM没有通过当前本地完整执行前置。** 它们的作者表存在，但作者数字不补齐本地checkpoint/数据/evaluator；缺项是unknown，不是效果0。
- **不另写saved checker。** 已有[Pauli离线readiness](/home/gbk/code/LeWorldModel/le-wm-main/innerdocs/agent_baseline_launch_readiness_20261028_v1.md)覆盖GRASP/LeWM资产与失败诊断；beam已有独立saved consumer。下面提供实际入口。

本文是[原执行矩阵](/home/gbk/code/LeWorldModel/docs/D2P_BASELINE_EXECUTION_MATRIX_JASON_20261028_v1.md)的协议/产物增量索引，不替换其冻结SHA，也不修改[Pauli覆盖表](/home/gbk/code/LeWorldModel/docs/D2P_BASELINE_COVERAGE_PAULI_20261028_v2.md)。10月26日faithfulness审查中的“当时缺实现”不能用作10月28日moving runner仍不存在的证据。

## 2. 作者原表与本地证据分栏（禁止据此跨协议排名）

作者记录权威为[238-cell registry](/home/gbk/code/LeWorldModel/le-wm-main/results/p2_reported_baseline_table_20261027_v3/export/report.json)，SHA `52e687811e6d6f728ab3eda60836ea05eff11f36da2bb4af975d1dd19ef380f1`。238项为236数值+2原文缺失，不是238个本地实验；按论文计GRASP112/Hi-LeWM106/LeWM8/PlanWAM12，包含各论文的对照方法。完整locator/原表quotes沿registry，不挑最好cell代替整体覆盖。

|方法|作者报告栏：固定代表cell，仅原协议|本地证据栏|阻止主表直比的关键差异|
|---|---|---|---|
|GRASP，2602.00475v1|Table1 p7，`grasp:S4.T1.3:5:1`，PushT H40 learned-model success 59.0%，不是实际物理SR|本地bounded virtual-state/soft-dynamics适配已完成model→physical→RGB；值沿独立comparison，不赋予作者59%|DINO-WM与本地LeWM、时域、初始化及预算不同；没有官方表级checkpoint/config配对|
|Hi-LeWM，2607.12547v2|Table4 p13，`hilewm:A3.T4.3:15:7/8`，staged d50 Hi-LeWM42.0±2.8%、-C64.0±3.3%，跨seed SD|完整本地pipeline效果unknown；已有text contract与资产失败记录|高低层对象、数据/normalizer、-C经验bank与本地moving任务未闭合|
|LeWM，2603.19312v3|Table10 p26，`lewm_v3:A7.T10.6:1:1`，CEM96.0±2.83%；±定义unknown，不继承别表定义|本地LeWM-backbone CEM/nominal-MPPI/GD有真实结果，不代表论文该cell复现|论文H5/data goal、逐solver优化与预算绑定不同；本地H2构造任务|
|PlanWAM，2610.11382v1|Table1 HTML，`planwam:S4.T1.6:16:11`，NAVSIM-v2 EPDMS90.9；PDF页unknown|本地官方驾驶pipeline效果unknown|4camera×4history、waypoints、NAVSIM/HUGSIM评分器不同于PushT raw action/native success|

这些作者值不是本轮新增数值；来源缓存位于registry同级`sources/`。本地值请读取下列原JSON，不另做跨协议排序表。

## 3. 原moving两根的共同合同

以下共同合同只适用于原off moving cohort，不自动涵盖zero-history、on overlay、真实反馈suffix、CEM8或其他根。

|项目|已冻结内容 / 仍缺内容|
|---|---|
|task00|[task_00.json.gz](/home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_00.json.gz)，SHA `a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b`|
|task01|[task_01.json.gz](/home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_01.json.gz)，SHA `101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74`|
|source/history|原sourcepack SHA `c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6`。真实RGB0/5/10；float32前5raw=[.1,0]，后5=[0,.1]；三latent状态与两incoming normalized packets。必须moving loader，不是旧padding/zero loader|
|模型|[weights.pt](/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm/weights.pt) SHA `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`；[config.json](/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm/config.json) SHA `2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09`。224RGB/patch14，257tokens、projectedCLS192，不是197tokens|
|normalizer|沿task完整float32常数，mean约[-.0087,.0068]、std约[.2019,.2002]；只做一次(raw-mean)/std，不再/100，不现场拟合。`training_provenance_verified=false`不因本地共用而变true|
|动作/目标|每packet五步、每步2raw；原packet logit球cap=.75*sqrt(5)，tanh映射；H2=10raw。各方法必须核自己的实际support，旧constant-packet bank与joint→packet投影不能冒充同域。原构造goal/with_target=False，不用默认轮廓代替goal|
|指标|模型及真实图像latent均用identity192 mean-square Q。native state7 distance包含velocity；success用agent+object前4维norm<20与原角度判据<pi/9，角度实现无mod的风险保留。object-only投影非正不能被说成完整任务失败。Q与native success分列|
|科学范围|仅2known roots，非独立fresh confirmation；P1未准入。相同predict/API样本不等于等FLOPs；AD/J/VJP/SVD、image acquisition、physics分别计量，不重复相加|

## 4. 本地实现与产物入口

下表module均位于`/home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/`。简称必须与完整stem配套，不能将旧结果换arm重标。

|方法 / 实际module|保存产物及状态|预算 / 最小剩余缺项|
|---|---|---|
|GRASP bounded：`p1_moving_grasp_baseline_cuda_20261027_v1`|[model report](/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_grasp_baseline_cuda_20261027_v1_cuda_main01/report.json)，`moving_grasp_model_plans_sealed_not_admitted`；extended physical/RGB同批含该arm|两根3604predict/AE、1800scalar backward。保留virtual states/soft dynamics，不是GD改名；H3指3历史帧。官方表级复现资产绑定unknown，不需重跑补清单|
|GD：`p1_moving_gradient_baseline_cuda_20261027_v1`|[model report](/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_gradient_baseline_cuda_20261027_v1_cuda_main01/report.json)；baseline physical/RGB已保存|两根272predict/AE、64scalar backward；ADforward128是predict子集。开环gradient，不是feedback MPC，不等于论文三solver全部复现|
|GN/D2P：`moving_history_planners_cuda_20261027_v1`|[model report](/home/gbk/code/LeWorldModel/le-wm-main/results/moving_history_planners_cuda_20261027_v1_cuda_main01/report.json)；[physical](/home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_moving_history_physical_20261027_v1_main01/report.json)；[RGB](/home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_moving_history_rgb_cuda_20261027_v1_cuda_main01/report.json)|每根GN78predict、4J/768数学VJP、4SVD/solve；D2P66predict、11J/2112数学VJP、12SVD。API AD调用与数学VJP不互换。该四arm cohort不在旧六法comparison中|
|旧sampling：`p1_moving_sampling_baselines_cuda_20261027_v1`|六法comparison已收旧CEM/nominal-MPPI，不能追认为新matched配置|两根旧CEM268predict、旧nominal-MPPI144，不公平的forward预算必须明确；nominal kernel不冒充完整官方MPPI|
|新matched sampling：`moving_prediction_matched_sampling_cuda_20261027_v1`|[model](/home/gbk/code/LeWorldModel/le-wm-main/results/moving_prediction_matched_sampling_cuda_20261027_v1_cuda_MAIN01/report.json)；[physical](/home/gbk/code/LeWorldModel/le-wm-main/results/moving_prediction_matched_sampling_physical_20261027_v1_MAIN01/report.json)；[RGBv2](/home/gbk/code/LeWorldModel/le-wm-main/results/moving_prediction_matched_sampling_rgb_cuda_20261027_v2_MAIN01/report.json)|每根每算法3×24完整H2=144predict API/samples，四组576；无额外nominal/winner replay。RGB14encoder/proj=12checkpoint+2unique goals，不抄28图。只匹配CEM/nominal-MPPI的forward轴|
|旧beam16x4：六法comparison内原kinodynamic arm|已有model/physical/RGB，保留旧身份与动作域|两根168predict，constant-packet rawbank；不能当同packet logit球逐时刻primitive基线|
|新beam：`agent_kinodynamic_matched_domain_20261028_v1`|[model](/home/gbk/code/LeWorldModel/le-wm-main/results/agent_kinodynamic_matched_domain_20261028_v1_cuda_MAIN01/report.json)；[physical](/home/gbk/code/LeWorldModel/le-wm-main/results/agent_kinodynamic_matched_physical_20261028_v1_MAIN01/report.json)；[saved consumer](/home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/agent_kinodynamic_matched_physical_consume_20261028_v1.py)|zero+19预固定球内点，逐时刻primitive；20→beam6→120，加whole2+segmented2=144predict/root，两根288。各node有context/actions见证；不是RRT，不是论文所有kinodynamic search。两根物理完成0native success；本次未登记经核的新beam RGB评价，不能用旧beam RGB顶替；**不重跑模型/物理**|

原六法共同保存视图：[comparison](/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_baseline_comparison_20261027_v3_main01/comparison.json)，SHA `96cf27b143bb1a839edf7bc1e4de02f4a97e433c6475c4c731c78af76300d32e`，12model/12physical/24RGB，`comparison_recorded_not_admitted`，native全部false。它不包含GN、新matched sampling或新beam。完整resource向量见[budget_matrix](/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_baseline_budget_matrix_20261027_v1_main01/budget_matrix.json)，SHA `721d978d29516f8ecf1d1056e159736ef231723e6849490a83678bd76baa1ebc`。shared原8encoder/proj继承一次，物理后RGB单列；无hardware/FLOPs计量处null而不是0。

GRASP及no-composition真实端点共同保存于[extended physical](/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_extended_baseline_physical_20261027_v1_main01/report.json)与[extended RGB](/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_extended_baseline_rgb_cuda_20261027_v1_cuda_main01/report.json)。不把no-composition packet映射后的support说成原joint ball完全相同。

## 5. 尚不可执行的官方路径：具体收什么材料

|方法|可复用的本地来源|阻断真实推理的最小材料 / 禁止替代|
|---|---|---|
|GRASP官方对齐|[官方contract缓存](/home/gbk/code/LeWorldModel/le-wm-main/results/p2_grasp_official_contract_20261027_v1/sources)，grasp commit `64f988aebf04ad9f674f9821bc6d2b9e22879105`，le-wm-grasp commit `94be3c42aa2f0867c6cb1e38bda498b293b6921e`；本地bounded机制可运行|作者逐表checkpoint/config/data/seed/完整预算仍未闭合；不影响读取本地adaptation，但不能宣称官方等协议效果|
|Hi-LeWM|[33份text/config](/home/gbk/code/LeWorldModel/le-wm-main/results/p1_hilewm_text_contract_20261027_v1/texts)；官方hierarchical eval真实high-subgoal→low；[资产恢复文档](/home/gbk/code/LeWorldModel/docs/D2P_HILEWM_ASSET_RECOVERY_PAULI_20261028_v1.md)|缺可信完整`package/checkpoints/pusht/main/pusht_hi_lewm_epoch15_object.ckpt`，应122424517B/CRC32 d59f865b，成员SHA尚未认证；可信加载/高低层配对/normalizer及依赖闭包另需认证。-C还需dataset identity、macro encoder、生成代码/indices与bank。不可现场fit或普通proposal bank改名|
|Hi-LeWM归档失败|[MAIN归档report](/home/gbk/code/LeWorldModel/le-wm-main/results/hilewm_archive_MAIN_20261028_v1/report.json)，SHA `19c526da4a8e3da6b340addd23289df2bd3a57e5a18169290f9bd13dd4f0783b`，`failed_preserved`|whole archive curl未完成，不能partial解pickle或称已恢复。期待归档2299863924B/MD5 aaf78b8b4fd6f5d1f5e720927951152f只为验收线索，CRC/MD5不替代可信pickle来源。本轮不重试网络|
|PlanWAM|registry缓存HTML与[协议文档](/home/gbk/code/LeWorldModel/docs/D2P_REPORTED_BASELINES_LEWM_PLANWAM_20261026.md)|需官方immutable推理revision、权重/config与许可、4camera/ego数据splitmanifest、NAVSIM-v2/HUGSIM evaluator。当前未认证本地完整入口；不猜造`planwam.run`，旧同名PointMaze脚本不算官方实现|
|原LeWM eval|[eval.py](/home/gbk/code/LeWorldModel/le-wm-main/eval.py)与[PushT配置](/home/gbk/code/LeWorldModel/le-wm-main/config/eval/pusht.yaml)|原eval会StandardScaler.fit且有CPU fallback；原数据/goaloffset/H5预算不是上述moving合同。本轮不可直接运行；不因本地有权重而绕过冻结normalizer与HOSTCUDA约束|

## 6. MAIN现在可安全直接执行的既存入口

全部cwd `/home/gbk/code/LeWorldModel/le-wm-main`；以下是saved metadata/fixture或own consume，不启动模型/物理。前提是保留全部原prepare/任务/NPY/PNG/模型文件字节，原闭包不漂移；任何失败保留并停止，不能改seal求通过。readiness不替代strict consume。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_baseline_launch_readiness_20261028_v1 --method GRASP
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_baseline_launch_readiness_20261028_v1 --method LeWM
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_baseline_launch_readiness_20261028_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_grasp_baseline_cuda_20261027_v1 consume --output results/p1_moving_grasp_baseline_cuda_20261027_v1_cuda_main01 --report-sha 7593940c1503416f42ce87a5d8135e8512763121e027c0ef98dbe614bfb31f91
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.moving_prediction_matched_sampling_cuda_20261027_v1 consume --output results/moving_prediction_matched_sampling_cuda_20261027_v1_cuda_MAIN01 --report-sha 60872ad114700731d82d293a616a4dede54a53b7be6514ad0cdf231d838d3ed8
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.moving_history_planners_cuda_20261027_v1 consume --output results/moving_history_planners_cuda_20261027_v1_cuda_main01 --report-sha cd8432c06288577d25d9d9b92c7eb3673de592499ee64cae93238156f3f19aa0
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_kinodynamic_matched_physical_consume_20261028_v1 --output results/agent_kinodynamic_matched_physical_20261028_v1_MAIN01 --report-sha 01d5485208fe40a71a53ae20bb2920740894d6ef7c114c139a991f4707e5ebe7
```

beam physical v1本身没有consume子命令，必须调用上面独立consumer，不改producer补接口。只需查看已有六法指标时可用：

```bash
jq '{status,denominator,rows:[.rows[]|{arm,source_slot,model_Q,realQ_terminal10,native_success,physical_returned_steps}]}' results/p1_moving_baseline_comparison_20261027_v3_main01/comparison.json
```

如MAIN另有新实验目的，readiness可生成实际`verify/run`命令（见其innerdoc），但本文**不推荐新run，也不授权重跑**。现有planner的prepare→verify→run→consume可复用，但需NEW输出、MAIN审读及HOSTCUDA资源检查。GRASP cap64MiB/wall600s，matched sampling与新beam cap64MiB/wall900s，floor200MiB；计时边界以各冻结源码为准，不声称覆盖整个进程。官方Hi-LeWM/PlanWAM无可复制的安全模型启动命令。

## 7. 本次实际检查与未运行项

本次实际执行已有readiness默认全方法：exit1为Hi-LeWM/PlanWAM blocked的预期；GRASP/LeWM三个工作流均`ready_for_MAIN_dry_preflight`。191个唯一hash读取、134008791 bytes；作者registry/comparison/budget三个证据SHA匹配。没有反序列化权重。已有readiness RAM测试实际 **13 passed in 0.16s**；不新增重复源码/测试。

另逐文件实际重核：GN/D2P model report `cd8432…`；matched sampling physical `8ce8634eddcd3e2e5b25d5b55572d80d06da1fd5d3513ee06aecec89001efa41`、RGBv2 `63c48c16b91d16b8f20c5f6ba08010d366544558f6e7eb9dd7a9542a5e50fa2f`；新beam model `d4976aeca357b5324c74be8b2b7e3bfc254e82b834b3acfe1b0ab90a1ef4614c`、physical `01d548…`；Hi-LeWM失败report `19c526…`。上文完整SHA作为命令参数或明确记录。metadata状态核对不等于重做完整journal/PNG/Q审计。

本次未执行上述own consume、prepare/verify、模型、物理、网络；历史MAIN完成状态与本次元数据检查分开。最小未决依赖为Hi-LeWM完整可信对象及配对闭包、PlanWAM官方驾驶资产、各作者表级协议身份，以及新beam尚未在本清单登记的独立RGB评价。后者若需补齐应先复用已有44PNG与原task建立严格saved输入桥，不先重跑模型或物理。所有本地known-root结果保留负结果，不能据测试、readiness或作者数字开启P1科学准入。
