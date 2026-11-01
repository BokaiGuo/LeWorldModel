# 强基线执行矩阵：现有入口、冻结资产与下一条命令

2026-10-28，Jason独占新增本文。MAIN负责代码审读和HOST CUDA/物理调度。没有修改共享主表、封存源码/结果，没有后代、网络/下载、神经CPU/CUDA、physics、训练、normalizer拟合或阈值修改。本文不生成新性能数值、不排序跨协议结果、不采样新根。

## 结论：下一步无需先重跑已完成基线

- **本地可消费**：原moving两根上的GRASP bounded、LeWM-backbone CEM/nominal-MPPI/gradient等已有真实model→physical→RGB产物。先消费现存结果，不因重新整理矩阵而再跑一轮。相同task/动作域/目标可作描述性配对，原六法预算不同，不能称等总算力。
- **同predict采样比较已完成**：新CEM/nominal-MPPI四计划各144 predict API和144 predictor samples；整次576。它们彼此匹配这两个轴，不是GRASP/GN/D2P等FLOPs基线。原nominal MPPI不是具名官方完整实现。
- **Hi-LeWM当前不能执行完整本地比较**：已取得源码/config文本及定位main object，但已知对象probe失败，不能把partial加载成权重；-C还缺已认证数据/normalizer/macro bank。应收资产，不重复探针或写一个冒名planner。
- **PlanWAM当前不能执行本地同PushT比较**：官方方法是驾驶pipeline，未建立本地完整模型/数据/评测入口。不是给PushT优化器换名就能运行，也不能将驾驶分数与PushT native success排名。
- **仓库原LeWM `eval.py` 不能直接作为本轮冻结normalizer执行入口**：其源码会对数据执行StandardScaler.fit，且默认device允许CPU回退；本轮不运行它，不通过临时参数掩盖协议差别。现有moving封存runner才是本地可执行的冻结合同。

本文补的是“该调用哪个入口／什么条件下不该运行”，不重复Pauli完整覆盖表或238-cell导出。原10月26日文档的“当时未取得原文”不覆盖后来已存在的缓存。

## 1. 作者原表严格另栏

权威输入：[238-cell registry](/home/gbk/code/LeWorldModel/le-wm-main/results/p2_reported_baseline_table_20261027_v3/export/report.json)，实际重核SHA `52e687811e6d6f728ab3eda60836ea05eff11f36da2bb4af975d1dd19ef380f1`。238＝236数值＋2原文缺失，**不是238次实验，也不全是论文自己方法**。计数按paper_key/论文，而不是method字符串：GRASP112、Hi-LeWM106、LeWM8、PlanWAM12。每cell原表/行列/token、协议quotes与HTML/PDF绑定仍沿registry，不新增未经核实页码或版本。

以下只转录固定代表cell便于定位，不从sweep选最好；它们没有本地效果含义。

|方法 / 原版本 / 全部选定cells|作者报告值专栏（仅原论文协议）|本地执行入口 / 本地值专栏|不可横比的最小原因|
|---|---|---|---|
|GRASP / 2602.00475v1 / 112|Table1 p7 `grasp:S4.T1.3:5:1`：H40，59.0%；是learned-world-model success，不是真实物理SR。原表成功trial计时与其余表仍见registry。|`p1_moving_grasp_baseline_cuda_20261027_v1`；本地值只读后文comparison，不把59%挂给此臂。|原DINO-WM、时域/动作与初始化/完整预算未对齐；bounded adaptation不是官方完整复现。|
|Hi-LeWM / 2607.12547v2 / 106|Table4 p13 `hilewm:A3.T4.3:15:7/8`：Hi-LeWM42.0±2.8%、-C64.0±3.3%，跨seed SD；原staged d50配置。|完整本地pipeline结果 **unknown**；已有text contract及失败资产probe，非0%效果。|高低层对象/坐标、normalizer/bank与任务反馈预算未闭合。|
|LeWM / 2603.19312v3 / 8|Table10 p26 `lewm_v3:A7.T10.6:1:1`：CEM96.0±2.83%；±确切定义unknown，不继承别表。其余solver仍见完整registry。|原LeWM-backbone的moving sampling/gradient；本地结果在独立saved列，不挂作者96%。|论文逐solver seed/checkpoint/预算绑定未建立；原论文H5/数据goal与本地H2构造目标不同。|
|PlanWAM / 2610.11382v1 / 12|Table1 HTML `planwam:S4.T1.6:16:11`：NAVSIM-v2 EPDMS90.9；PDF页unknown。其余PDMS/HUGSIM原值在registry。|官方本地驾驶结果 **unknown**；未建立可dispatch的本地官方入口。|4相机×4历史、驾驶waypoints/评分器、训练模块与PushT raw action/native判据不同。|

来源与版本细节已有[Pauli覆盖表](/home/gbk/code/LeWorldModel/docs/D2P_BASELINE_COVERAGE_PAULI_20261028_v2.md)、[原表定位](/home/gbk/code/LeWorldModel/docs/D2P_REPORTED_BASELINE_NUMBERS_PRIMARY_LOCATORS_20261027.md)。不声称本轮检查了在线最新版本、DOI或发表状态。

## 2. 本地共同底座：什么已固定，什么仍unknown

以下只适用于原moving/off两根，不把zero-history、overlay/on、observed suffix或CEM8混为同cohort。

|轴|本地已核合同 / 实际资产|仍不能宣称|
|---|---|---|
|模型|`/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm/weights.pt` SHA `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`；`config.json` SHA `2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09`。本轮只读hash，不加载模型。|对应作者每一表cell的immutable revision仍unknown。|
|架构|实际config：224 RGB、tiny ViT patch14、3frame predictor、projected CLS192、action encoder input10、pred_proj192；224/14形成16×16 patches，不能硬抄197 tokens。|未在本轮运行encoder测shape；已有实际module receipts才是运行时依据，不能只凭config认证所有第三方模型。|
|原tasks|`results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_00.json.gz` SHA `a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b`；task01 SHA `101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74`。对应moving-specific loader，不能用旧padding loader。|只有2known roots，非fresh独立确认或P1。|
|历史/目标|真实RGB0/5/10和raw前5次[.1,0]、后5次[0,.1]，两normalized incoming packets；原构造goal、with_target=False。|不能换zero history、默认轮廓goal或目标可视化condition；不能按native outcome选根。|
|normalizer|task_contract已冻结float32 mean≈[-.0087,.0068]、std≈[.2019,.2002]；实际完整数值以task为准。raw已是public relative单位，仅做一次(raw-mean)/std，无第二次/100。|`training_provenance_verified=false`仍保留。它是同本地合同可比，不是训练数据统计已溯源。不得现场拟合更新。|
|动作/评价|每packet10维logit球半径`.75*sqrt(5)`，tanh→raw5×2，H2=10raw；native192 terminal mean-square Q；native完整任务判据与Q分列。|不是作者原动作域或完整成本统一。native位置含agent+object，object-only非正不等于完整任务失败；state7距离与success阈值也不互换。|
|反馈/预算|原root开环fullH2，与真实prefix反馈suffix分层报告；API、predictor samples、AE/proj、AD/backward/J/VJP/SVD、image编码及physics分别列。|相同sample数不等于等FLOPs；无独立计量的轴unknown，不填免费0、不把子模块与API重复相加。|

模型资产已存在，无需再下载。共享原8次encoder/projector成本沿原任务继承一次，物理后RGB单列；不把旧编码扣掉，也不在每法重复摊成免费。原task正常化谱系未认证不应悄悄被论文数据normalizer替换。

## 3. 实际入口与执行缺项矩阵

|本地入口|已存在的实际状态 / 固定预算|MAIN下一项可执行工作|不能执行或不能升级的内容|
|---|---|---|---|
|`p1_moving_grasp_baseline_cuda_20261027_v1`|实际report `moving_grasp_model_plans_sealed_not_admitted`；2根合计3604 predict/AE、1800 scalar backward，API账一一对应不相加。750 joint＋每50步sync10；virtual state/soft dynamics保留。|先own consume既有report，或verify既有prepare；后续仅在新问题/新协议授权时用NEW输出run。|`official_equivalence=false`；H3是3历史帧，不是未来15raw。不能把restart-Adam叫GRASP，也不能把本bounded当作者同协议复现。|
|`p1_moving_sampling_baselines_cuda_20261027_v1`|旧CEM/nominal-MPPI均已跑，原CEM与MPPI预算不等。|读取六法comparison和budget矩阵；不为了补表重跑。|旧配置不能追认为新144 cap；nominal kernel不等于所有官方MPPI。|
|`moving_prediction_matched_sampling_cuda_20261027_v1`|4计划，各3轮×24完整H2=144 predict API/samples，整次576；无额外winner replay/nominal补算。新sampling→physical→RGBv2均已有保存产物。|own consume精确modelreport，结合其独立physical/RGBv2 cohort；需要新模型运行时先verify，再MAIN独占CUDA。|只允许CEM与nominal-MPPI彼此匹配该forward主轴，不能与GRASP/GN/D2P称等算力。|
|`p1_moving_gradient_baseline_cuda_20261027_v1`|原moving开环gradient实际已保存，合计272 predict/AE，64 scalar reverse sweeps；完整上下文和own consumer存在。|读取已保存六法结果；如要闭环gradient，须另预声明并实现新协议，不重标旧臂。|`feedback_MPC=false`；不是LeWM论文Table10三solver的直接逐项复现。|
|Hi-LeWM `p1_hilewm_text_contract_20261027_v1` / 官方归档eval文本|33份文本及main config存在；`eval/hierarchical.py`将high rollout真实subgoal交给low，机制接口已定位，不是普通proposal bank。|接收可信完整对象后，按资产恢复文档做纯字节验收，再由MAIN安排可信加载/共同embedded-low认证。|当前完整object、配对normalizer和-C bank未认证；没有本地可安全直接执行的高低层runner闭包。|
|PlanWAM|原论文HTML缓存/238-cell registry可读；本地官方驾驶pipeline未建立。|只消费作者原表；外部交付明确官方推理代码revision、checkpoint+config、数据/评测manifest与许可后再设计新合同。|不提供猜造的`planwam.run`命令；旧PointMaze实验名包含planwam不算官方方法。|
|原 `eval.py`＋`config/eval/pusht.yaml`|入口真实存在：原配置50eval、goal offset25、H5/actionblock5/evalbudget50，CEM300×30/topk30；源码有dataset StandardScaler.fit。|本轮仅只读配置；不调用原eval，不创建转换object。|在本轮“不拟合normalizer、无CPUfallback、固定moving任务”约束下不可直接dispatch。数据缺项、episode selection和原表逐solver配对亦未闭合；不能把这些默认值全部配给Table10。|

官方GRASP小文本所在 `results/p2_grasp_official_contract_20261027_v1/sources/`，已保存grasp commit `64f988aebf04ad9f674f9821bc6d2b9e22879105`、le-wm-grasp commit `94be3c42aa2f0867c6cb1e38bda498b293b6921e`、MIT许可及solver/pseudocode；代码版本线索不等于作者实验checkpoint/配置全等证明。

## 4. 可直接复制的命令：先saved，不重跑

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。以下为MAIN可执行清单；本worker只执行第A项中的轻量测试与小文件读取，没有执行consume/verify或模型。

### A. 作者表与本地资源列，只读

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_d2p_baseline_reported_values_20261027_v1.py
jq '{counts,cross_paper_ranking_allowed,local_reproduction}' results/p2_reported_baseline_table_20261027_v3/export/report.json
jq '[.rows[]|{id,paper:.record.paper,version:.record.version,value:.record.value,unit:.record.unit,uncertainty:.record.uncertainty,table:.source.table_id,page:.source.pdf_page_1based}]' results/p2_reported_baseline_table_20261027_v3/export/report.json
jq '{denominator,status,rows:[.rows[]|{arm,source_slot,model_Q,realQ_terminal10,native_success,physical_returned_steps}]}' results/p1_moving_baseline_comparison_20261027_v3_main01/comparison.json
jq '{methods,model_invocations,physical_cohorts,RGB_cohorts,inherited_encoding_shared_once}' results/p1_moving_baseline_budget_matrix_20261027_v1_main01/budget_matrix.json
```

### B. 已有真实model计划的own consume（非独立审计／非模型运行）

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_grasp_baseline_cuda_20261027_v1 consume --output results/p1_moving_grasp_baseline_cuda_20261027_v1_cuda_main01 --report-sha 7593940c1503416f42ce87a5d8135e8512763121e027c0ef98dbe614bfb31f91
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.moving_prediction_matched_sampling_cuda_20261027_v1 consume --output results/moving_prediction_matched_sampling_cuda_20261027_v1_cuda_MAIN01 --report-sha 60872ad114700731d82d293a616a4dede54a53b7be6514ad0cdf231d838d3ed8
```

只在当前依赖闭包仍与封存一致时通过；漂移应failclosed，不能修改seal适配。consume只验证已保存选择，不重新优化，不是新的实验证据。

### C. 既有prepare verify / 后续MAIN-only run接口

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_grasp_baseline_cuda_20261027_v1 verify --prepared results/p1_moving_grasp_baseline_cuda_20261027_v1_prepare01.json --prepare-sha d1f6a9ac66774759ff47aa02cbfb00132399da5971f0458500deec359b2cca87
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.moving_prediction_matched_sampling_cuda_20261027_v1 verify --prepared results/moving_prediction_matched_sampling_cuda_20261027_v1_prepare_MAIN01.json --prepare-sha 4e4eededb3f8d498ff0c7993e47e7f8935129d99ebdd02270f30de9aae0dbf06
```

两个module都支持 `prepare/verify/run/consume`。新prepare用各自innerdoc的原task参数（GRASP）或无新参数的fixed build（matched sampling）；prepare只能写NEW文件，不能覆盖上面已有文件。若MAIN有明确新实验目的并通过预检，run语法为：

```text
.venv/bin/python -B -m experiments.spectral_reachability.<上述实际module> run --prepared <实际NEW或已验证prepare> --prepare-sha <实际SHA> --output <MAIN指定NEW目录>
```

这是待MAIN补入新目的/真实NEW输出的接口，不是已授权立即重跑的默认命令。GRASP cap64MiB/floor200MiB/wall600s；matched sampling cap64MiB/floor200MiB/wall900s，后者pre-dispatch meter保4组144 cap；不能改预算以绕过gate。两者HOSTCUDA-only、MATH/TF32off、冻结参数，不CPUfallback。模型规划、物理执行和RGB评价是不同入口/计量，不从planner modelQ填真实endpoint。

## 5. 不能执行的官方管线：要哪份最小材料

### Hi-LeWM

当前known入口状态以[Pauli资产恢复交接](/home/gbk/code/LeWorldModel/docs/D2P_HILEWM_ASSET_RECOVERY_PAULI_20261028_v1.md)为准：

- 正式目标成员 `package/checkpoints/pusht/main/pusht_hi_lewm_epoch15_object.ckpt`，完整展开122,424,517 bytes；现有partial不是它。
- 供件任选完整官方package.zip，或该完整对象＋MAIN可信来源的expected SHA及许可/来源说明。既有恢复文档提供不解pickle的size/CRC/SHA字节验收命令；这里不重复下载/续传协议。现有probe仍`failed_preserved`，不能拿失败probe运行full。
- main config已齐，不需重新下载。内嵌低层若能认证为共同坐标，可新做共同backbone比较；不必凭空要求作者未给出的外部flat历史SHA。但若复用本地旧endpoint，仍须证明等价。
- 原dataset normalizer与-C empirical bank未认证。`Hi-LeWM-C`需数据身份、原统计/处理、macro encoder、row索引及生成合同；不能用本地proposal bank代替。当前禁拟合/训练，不临时生成替代统计。
- 本轮仅检查默认候选 `/home/gbk/.stable_worldmodel/pusht_expert_train.h5` 不存在；不声称全盘没有数据，未检查其他cache。真实cache路径/数据digest应由MAIN/资产方明确提供。

### LeWM官方表级复现

已有flat模型并不等于官方Table10全部solver配置已齐。需要每solver的优化器参数、停止/重放范围、原checkpoint/config、数据/episode manifest、预处理/normalizer、seed与误差定义。原eval现场fit与当前冻结统计约束冲突，故不能直接复制README的eval命令开跑，更不允许训练或在线拟合来补表。

### PlanWAM

需要官方完整推理code revision/环境、已训练权重及config、4相机/ego数据与NAVSIM/HUGSIM明确split/评测器、许可和资源合同；当前未建立本地实际入口，不能为表格完整而造命令。PlanWAM训练LoRA/规划模块的事实不授权本轮训练。其驾驶pipeline不能在原PushT任务上“公平原样执行”；如研究要新adaptation，需新协议并明确不叫官方结果。

## 6. 本轮实际核对、未运行项与证据SHA

实际：仅按明确路径读取registry、prepare/report/config/task元数据；核模型两个文件SHA（只读权重字节、不反序列化），核入口CLI声明及既有innerdoc。既有轻量测试 **20 passed in0.34s**，不是神经实验，也不承担MAIN代码审计。未新增源码/测试/结构JSON；不重复原已有覆盖工具。本文初稿完成后的本地链接/命令源码存在性检查见交付消息。

|已实际复核的保存输入|SHA256|
|---|---|
|六法comparison v3|`96cf27b143bb1a839edf7bc1e4de02f4a97e433c6475c4c731c78af76300d32e`|
|六法budget_matrix|`721d978d29516f8ecf1d1056e159736ef231723e6849490a83678bd76baa1ebc`|
|moving GRASP actualreport|`7593940c1503416f42ce87a5d8135e8512763121e027c0ef98dbe614bfb31f91`|
|matched sampling actualreport|`60872ad114700731d82d293a616a4dede54a53b7be6514ad0cdf231d838d3ed8`|
|Hi-LeWM v2失败probe|`669bf41043dfae01a9bca183fe230d61baad29234c5cc73038c80409844e0b90`|
|moving GRASP source|`52d3157574e4e2726f6d88b43bd7e5dc804ce79730b65512888e63e0dd7c7106`|
|matched sampling source|`72195df2a7f1d10948c2bf5c494dd26e66400eaa2e269737332cbd876e599d40`|

未运行：新prepare、完整consumer/verify、原文全量重建、任何神经/物理/网络/官方checkpoint加载、环境安装、normalizer fit；未独立审全部模型/物理/PNG/journal。不把前人/MAIN实际结果当本worker新跑结果。完整官方可比性、Hi-LeWM资产配对、PlanWAM本地pipeline、独立fresh根/P1仍未完成。

下一动作顺序：先B消费已存在计划、A读取分栏资源；若研究需要Hi-LeWM，向供件方索取上述完整对象/来源及冻结数据统计，不再probe；PlanWAM保留作者背景引用，直到有明确可执行资产合同。没有充分理由，不重跑当前已完成的moving基线。
