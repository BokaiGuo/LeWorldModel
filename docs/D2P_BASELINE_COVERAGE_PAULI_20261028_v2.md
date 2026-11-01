# D2P baseline覆盖表：作者原表与本地保存结果分栏

核对日期：2026-10-28，Asia/Shanghai。本轮仅新增本文；不改旧源码、封存结果或共享文档。无后代、网络、下载、训练、CUDA、神经或physics执行。本表是现有证据覆盖清单，不是全论文引文认证、官方复现认证或P1准入。论文身份/数值以**固定版本本地原文缓存**为准；未核在线最新版本、DOI当前解析及发表状态，不猜会议。

## 1. 结论与使用口径

四篇论文的作者结果已存在可定位本地原文，不能继续沿用10月26日“原文未取得”的旧快照。实际离线重建238个选定单元格，与封存export逐项相同：GRASP论文112、Hi-LeWM论文106、LeWM论文8、PlanWAM论文12；其中236个数值、2个原文缺失。计数包括对照方法与配置，**不是238次本地实验**。

本地moving六基线已有12 model/12 physical/24 RGB完整综合，不能写成“尚无结果”。但它们是两根校准、不同资源预算的本地变体；Hi-LeWM完整高层管线及PlanWAM官方驾驶管线的本地结果仍为unknown。另有新同predict预算CEM/MPPI四计划及CEM8/teacher_full八anchor配对，不与旧六基线或作者值拼成一张统一排行榜。

作者报告值只作 `paper_reported/context_only`；本地值只作 `saved_local/calibration`。缺表号、协议、误差定义、完整资产或计量，写unknown，**不填0、不从另一篇论文/另一张表继承、不拿modelQ填actualQ/native success**。旧版及失败全部保留。

## 2. 固定来源入口与哈希

以下均实际读取并复算SHA。项目根为 `/home/gbk/code/LeWorldModel/le-wm-main`；来源代号用于后续表格，不省略真实路径。

|代号|现有产物路径|SHA256|
|---|---|---|
|R|[四论文封存原表registry](/home/gbk/code/LeWorldModel/le-wm-main/results/p2_reported_baseline_table_20261027_v3/export/report.json)|`52e687811e6d6f728ab3eda60836ea05eff11f36da2bb4af975d1dd19ef380f1`|
|L|[六基线comparison v3](/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_baseline_comparison_20261027_v3_main01/comparison.json)|`96cf27b143bb1a839edf7bc1e4de02f4a97e433c6475c4c731c78af76300d32e`|
|B|[六基线budget_matrix](/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_baseline_budget_matrix_20261027_v1_main01/budget_matrix.json)|`721d978d29516f8ecf1d1056e159736ef231723e6849490a83678bd76baa1ebc`|
|P|[资源—结果逐根Pareto saved02](/home/gbk/code/LeWorldModel/le-wm-main/results/moving_baseline_resource_outcome_pareto_20261027_v1_saved02/pareto.json)|`838b1916a7da99240a4593d994f4db6edd88779cbf8567190b20236273456e6e`|
|Gm|[GN/D2P overlay模型](/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_overlay_recompile_cuda_20261027_v1_cuda_MAIN01/report.json)|`2b2e5064f74f8aca712ae0d0415fcbfb64a2c8e6325bdffe24ae4fdc3b373381`|
|Gp|[GN/D2P overlay物理](/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_overlay_recompile_physical_20261027_v1_MAIN01/report.json)|`56962e9370fcf249a2ed88f004a400414b491f503a4c3a9db1bd951cf984a521`|
|Gr|[GN/D2P overlay RGB](/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_overlay_recompile_rgb_cuda_20261027_v1_MAIN01/report.json)|`18d2f192a75e397c01e6c7eb7f6422e9bb073af865107ef469debe5b8ae32d1e`|
|Sm|[同predict预算sampling模型](/home/gbk/code/LeWorldModel/le-wm-main/results/moving_prediction_matched_sampling_cuda_20261027_v1_cuda_MAIN01/report.json)|`60872ad114700731d82d293a616a4dede54a53b7be6514ad0cdf231d838d3ed8`|
|Sp|[该四计划物理](/home/gbk/code/LeWorldModel/le-wm-main/results/moving_prediction_matched_sampling_physical_20261027_v1_MAIN01/report.json)|`8ce8634eddcd3e2e5b25d5b55572d80d06da1fd5d3513ee06aecec89001efa41`|
|Sr|[该四计划RGB v2](/home/gbk/code/LeWorldModel/le-wm-main/results/moving_prediction_matched_sampling_rgb_cuda_20261027_v2_MAIN01/report.json)|`63c48c16b91d16b8f20c5f6ba08010d366544558f6e7eb9dd7a9542a5e50fa2f`|
|C8|[Hegel严格CEM8配对分析](/home/gbk/code/LeWorldModel/le-wm-main/results/agent_CEM8_physical_analysis_20261028_v1_MAIN01.json)|`a3f874328acec265428c95c209aed3dc3fb5fcaebb2f822a3e6daeec5ac4d370`|
|C8T|[CEM8/teacher_full长表](/home/gbk/code/LeWorldModel/le-wm-main/results/agent_CEM8_baseline_table_20261028_v1_MAIN01.json)|`8683200b0a9fd46d8d8b0e4155e828222d63f04c78af5b01397637afbbcca8ac`|
|H|[Hi-LeWM资产probe失败回执](/home/gbk/code/LeWorldModel/le-wm-main/results/p1_hilewm_bounded_object_fetch_20261027_v2_probe_main01/report.json)|`669bf41043dfae01a9bca183fe230d61baad29234c5cc73038c80409844e0b90`|

R每个cell包含完整title/version/source_url、figure/table/行列ID、raw_row/reported_token、原HTML SHA与protocol_quotes；前三篇另有PDF SHA/页和页文本SHA。原始缓存位于[既有sources目录](/home/gbk/code/LeWorldModel/le-wm-main/results/p2_reported_baseline_table_20261027_v3/sources)。原始HTML文件名为 `grasp_html.html`、`hilewm_html.html`、`lewm_v3_html.html`、`planwam_html.html`；不把解析后的合并嵌套表当原文。

辅助定位：[逐页/协议核对](/home/gbk/code/LeWorldModel/docs/D2P_REPORTED_BASELINE_NUMBERS_PRIMARY_LOCATORS_20261027.md)、[完整作者报告值](/home/gbk/code/LeWorldModel/docs/d2p_baseline_reported_values_20261027_v1.md)、[既有分栏交接](/home/gbk/code/LeWorldModel/docs/chat_baselines_20261027_evidence_handoff.md)。本轮复用这些既有来源链，不覆盖旧“未取得原文”文档，也不把历史失败写成当前论文不存在。

## 3. 四篇论文覆盖：作者值与本地值严格分栏

示例固定配置用于引用定位，不从全sweep逐格选最优；完整选定单元格继续保存在R。所有跨列直接差值、提升百分比、等预算名次均不成立。

|方法／固定原文版本|作者报告值、确切表号／协议|本地结果及来源，绝非作者值|覆盖边界／unknown|
|---|---|---|---|
|GRASP，*Parallel Stochastic Gradient-Based Planning for World Models*，2602.00475v1|Table1 p7 / `S4.T1.3`，Push-T H40：59.0%与成功trial中位8.5s；500 trials/setting，learned-world-model success。R cell `grasp:S4.T1.3:5:1`及时间相邻列。Tables3/4 p8另有任务/H sweep。|L的`GRASP_H3_bounded`两根：mean modelQ=0.1626559966364103，mean actual terminalQ=0.236275930840675，native=0/2；不是原文59%。|本地bounded机制变体，非官方同协议复现；H3是历史3帧、未来H2/10动作，非15步。原表H到本地primitive/model-step精确映射、权重digest、完整总费用unknown。Table4时间分母unknown。|
|Hi-LeWM，*Mind the Gap: Promises and Pitfalls of Hierarchical Planning in LeWorldModel*，2607.12547v2|Table4 p13 / `A3.T4.3`：staged、d50、(high_h,high_replan,low_h,low_r)=(2,5,2,1)，PushT Hi-LeWM42.0±2.8%、-C64.0±3.3%；R cells `hilewm:A3.T4.3:15:7/8`，跨seed SD。C.2为每配置50episodes、通常3seeds；Table5 p14给预算，d为环境步offset而非horizon。|本地完整Hi-LeWM结果unknown/null；H状态`failed_preserved`，首错`TransferError: Truncated range`，`unpickled=false`，不是0%成功。|原文冻结低层，但训练macro encoder/high predictor，-C还需empirical macro bank。完整官方高低层/normalizer/bank配对digest未认证；不训练补齐，不把普通proposal bank冒充learned prior。|
|LeWM，*LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels*，2603.19312v3|Table10 p26 / `A7.T10.6`，PushT LeWM列：CEM96.0±2.83、SGD26±4.32、RMSProp67.33±2.49、Adam84±7.12%；R `lewm_v3:A7.T10.6:1:1`至`:4:1`。此Table10不是v1；±定义及各cell具体seed/episode绑定unknown。|L的LeWM-backbone本地CEM：mean modelQ=0.30700590568169106、mean actual terminalQ=0.24477232582649805、native=0/2。C8另有suffix CEM64，不能替换此行。|骨干来源≠整任务官方复现。通用CEM300samples×30轮、elite30，H5blocks/25步、执行budget50；未证明逐个Table10 cell都绑定此协议，尤其梯度solver预算unknown。不能从Table5数值相同推定Table10的±定义。|
|PlanWAM，*Planning-Shaped Future Representations for End-to-End Autonomous Driving*，2610.11382v1|Table1 `S4.T1.6` NAVSIM-v2 EPDMS90.9；Table2 `S4.T2.1` v1 PDMS93.8；Table3 `S4.T3.1/.3` HUGSIM RC Avg53.7、HDScore Avg38.7。PDF页unknown；HTML原子表及协议可核，非整项unknown。|本地官方驾驶pipeline及分数unknown/null。旧PointMaze实验名含planwam不算该方法结果。|驾驶分数不是PushT native success；4历史帧×4相机、64候选×8waypoints/4秒，含训练LoRA/规划评分模块。HUGSIM436scenarios的Avg按场景数加权，不取4难度等权均值。官方权重digest/完整推理资源unknown。|

R的原文限定可以直接定位，而不是转引内部评价：GRASP `S4.SS2.p2.1`：“All reported metrics measure task success under the learned world model.”；Table1 caption明确“successful trials only”。这直接禁止把59.0%写成本地物理成功率。LeWM Table10 ±、PlanWAM PDF页等未建立的字段继续unknown，不补造表页或误差解释。

## 4. 算法族覆盖：已有本地对比，不冒充外部成绩

下表L的modelQ/actualQ均为**两个固定moving roots的描述性均值**；native列是实际计数，不是总体成功率。实际Q为自身RGB经冻结native192 latent对goal的MSE，与原生state7距离不同。各方法完整两根、负结果和未计量项保留。

|算法族／实际arm|可验证作者原表值|本地已保存结果／来源|实际新规划资源（两根合计，分轴）／最小缺口|
|---|---|---|---|
|CEM，`CEM`|LeWM v3 Table10 CEM96.0±2.83；仅对应其原协议，不能给本地挂此分数。|L：modelQ0.30700590568169106、actualQ0.24477232582649805、native0/2、两根均10步；Sm/Sp/Sr另有新144 cap四计划。|L predict268、AE268，backward/J/SVD显式0；不能与旧MPPI144称同predict。最小同predict比较已存在于Sm，不需重跑旧配置。|
|nominal MPPI，`kernel_faithful_mppi_nominal`|本次未绑定具名官方MPPI论文表号/协议，作者值unknown。|L：modelQ1.2488806768324243、actualQ1.2282096618213263、native0/2；新同predict版本另列下节。|L predict144、AE144，backward/J/SVD显式0；local nominal kernel，不等同所有强MPPI实现或官方benchmark。|
|gradient MPC族，实际`moving_gradient_open_loop`|LeWM Table10三种gradient solver有上节原值；此本地梯度器与它们不是已建立一一对应，原算法值unknown。|L：modelQ0.16666522166766978、actualQ0.2506878873727779、native0/2。|predict272、AE272，ADforward128已包含其中；64 scalar reverse sweeps/grad API描述重叠工作，无J/SVD。它是open-loop differentiable MPC optimizer，`feedback_MPC=false`，**闭环gradient MPC结果unknown**，不能仅改标签完成覆盖。|
|GN，`line_search_GN`|未绑定一篇具名GN控制论文的原表/协议，作者值unknown。|Gm/Gp/Gr存在完整model/physics/RGB；Gp固定root0/1×off/on的GN共4branch，native0/4、每branch own_returned10。不是六基线L里已有的第七法，也不是4独立根。|Gm每GN branch predict78、AE78、J4、VJP方向768、SVD4、solve4；AD/backward API无独立字段不补0。比较必须保留condition、task、goal、原prefix身份；不直接拼到L的off-only两根均值表。|
|无组合SRA，`joint_H2_spectral_packet_mapped`|本地无组合对照，不是已认证具名外部SRA论文benchmark；作者值unknown。|L：modelQ0.6050982669483915、actualQ0.4437224331512617、native0/2。|predict176、AE176、ADforward4为子集，J2、数学VJP方向384、SVD4、bounded solve2；独立backward API unknown。joint bank后独立packet投影改变支持域，不能称精确双约束ridge或等搜索域。|
|kinodynamic，`fixed_H2_kinodynamic_beam16x4`|算法类别，未绑定具名官方论文及原表，作者值unknown。|L：modelQ0.4753335862086381、actualQ0.4256731828264687、native0/2；不是“完全没有搜索实现”。|predict168、AE168、pred_projector168；backward/J/SVD显式0。每根16首段＋beam4×16后段80规划＋4 replay；constant-packet离散库与逐时刻动作球不同，不称自适应预算优势。|

GRASP的L资源为predict3604、AE3604、ADforward3600子集、1800 scalar loss.backward；不是1800×192输出VJP。B/P保留不同导数语义、计量单位与完整成本缺口：P的30个同根方法对均为unknown，不能把“未证明被支配”当作“已证明Pareto前沿”。**这个历史all-axes Pareto合同也不否定另行预声明的同predict调用/样本比较**；后者不要求wall/J/AD逐项相等，不宣称总算力相等。

## 5. 新预算／后缀cohort必须单列

### 5.1 CEM/MPPI同predict调用与forward samples

Sm是**新配置**：每root每method 3iter×24 candidates×H2=144 predict API、144 forward samples，AE/proj各144；4计划各144，整次各576。原六法配置不能追认成这个预算，也不能补造另8组。Sp四计划均10步完成、native0/4（每法0/2）；Sr为`encoded_sealed`，每branch自己的checkpoint0/5/10，2 goal＋12 checkpoints=14图的encoder/projector账单独计。

Sr实际terminal10 Q：root0 CEM0.5307681356555619 / nominalMPPI0.4074763050455723；root1 CEM0.1381070809062569 / nominalMPPI0.906007014744056。完整四分母保留，不按root择优。它们不能与L的旧modelQ/actualQ互换。相同predict调用/样本不代表相同FLOPs、wall、encoder或全历史开销。

### 5.2 CEM8 versus teacher_full suffix

C8严格保存分析与C8T长表已存在；另有[MAIN独立复算](/home/gbk/code/LeWorldModel/le-wm-main/results/recompute_CEM8_pair_MAIN_20261028_v1_MAIN01.json)，SHA `dd546eb2dd9c127b9d2a95961ed7be21e961f5f3fd0d7c33ff50f1231929288e`。完整8 anchors＝原2根×off/on×两种prefix，不是8独立样本。双方各1/8 native成功，但索引分别CEM[1]与teacher_full[3]；平均CEM-minus-teacher joint XY为+22.711274897484124，仅已知根描述，非普遍优胜。

C8T有32指标行（8×2methods×endpoint/H10）和16唯一resource_rows。CEM每anchor65 API/65samples，teacher_full13 API/13samples＋自身J/VJP/SVD，**不是同predict或等总算力**。earlystop的实际后缀步数与H10缺失分列；每预算在endpoint/H10展示两次，必须按resource_row_id去重，不重复收费。不得再运行8anchor物理“补齐覆盖”。

另有[GRASP-H1保存核算](/home/gbk/code/LeWorldModel/le-wm-main/results/p2_grasp_h1_saved_outcome_audit_20261027_v1_main_verify01/report.json)：8branch/4root pairs、4完成/4 raw越界拒绝、只有1 full-path pair，其余3 unmatched；保留该负结果和分母，不并入L、Sm或C8，不把4完成解释为完整4对成功。本轮未重复运行。

## 6. 可直接执行的最小缺口

优先已有saved核验，不为了表格完整启动新训练/神经/物理。以下命令均从 `/home/gbk/code/LeWorldModel/le-wm-main` 执行，无输出文件改写。

1. **作者原表的可核查性已可立即复核，无需联网补抓。**

   ```bash
   PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.d2p_baseline_reported_values_20261027_v1
   jq '[.rows[] | {id,method,paper:.record.paper,version:.record.version,value:.record.value,table:.source.table_id,page:.source.pdf_page_1based,protocol:.source.protocol_quotes}]' results/p2_reported_baseline_table_20261027_v3/export/report.json
   ```

2. **六法完整分母与分轴预算直接读取，不重建模型。**

   ```bash
   jq '{denominator,status,rows:[.rows[]|{arm,source_slot,model_Q,realQ_terminal10,native_success,physical_returned_steps}]}' results/p1_moving_baseline_comparison_20261027_v3_main01/comparison.json
   jq '{methods,model_invocations,physical_cohorts,RGB_cohorts,inherited_encoding_shared_once}' results/p1_moving_baseline_budget_matrix_20261027_v1_main01/budget_matrix.json
   jq '{rows:[.rows[]|{anchor_index,method,endpoint_kind,actual_suffix_returned_steps,native_success,joint_xy_error,model_budget_axes}],resource_groups}' results/agent_CEM8_baseline_table_20261028_v1_MAIN01.json
   ```

3. **Hi-LeWM当前最小阻塞是完整资产，不是再写一个planner。** 立即执行 `jq '{status,first_error,unpickled}' results/p1_hilewm_bounded_object_fetch_20261027_v2_probe_main01/report.json` 可复核失败。下一输入应由MAIN/资产提供方交付完整可信高层checkpoint、原低层配对、config/normalizer及-C所需bank和实际hash/许可；先只读核验，不自动反序列化不可信pickle。禁止训练补齐、续传失败probe或反复换URL。资产未到，本地结果保持unknown。

4. **PlanWAM/LeWM原表缺字段先向现有材料补证，不跑替代实验。** PlanWAM要PDF页须提供确切v1 PDF与SHA（或继续只引用已核HTML子表，不需要为HTML可引用性强补页码）；官方本地驾驶分数还缺完整合法pipeline/权重/评测合同。LeWM Table10要逐solver迭代、学习率、停机、seed/误差定义和checkpoint绑定；未取得前保留unknown，不把300×30分给gradient solver。本轮不发网络请求，若后续上游403/权限错误即报告并停，不反复重试。

5. **闭环gradient MPC／官方MPPI／官方GN或kinodynamic成绩不是现有local行的别名。** 本轮最小可执行工作是上面saved核对及保留准确标签；若研究确需新闭环或官方复现，先由MAIN预声明固定原根/任务、实际新prefix完整恢复、反馈、动作域、失败attempt/重放范围，以及predict API或forward samples预算mode与batch限制。只有MAIN审查后HOST CUDA才可运行；本worker不提供“改名即跑通”的命令，不挑根/调阈值、不以模型CPU回退，不把缺论文协议当作已认证。

6. **独立freshness/P1是单独前置。** 当前所有cohort均为已知根校准；新增文档、RAM测试、同predict预算或saved consume不签新根证书、不解除P1。没有独立认证时科学准入继续关闭，而已有校准范围工程报告可以如实列出。

共享继承encoder/projector只计一次，evaluation RGB另计；teacher8是all24子集，不再把104加到312。predict API、predictor samples、AE/proj、AD-forward、scalar backward、J、数学VJP方向、SVD、physics/public steps、wall各有自己的scope/单位。未知项不是免费项；不相加成“总compute”，不退费、不均摊整个cohort，也不强求不同算法导数工作逐项同数才能作任何比较。

## 7. 本轮实际验证与未完成项

- 旧来源consumer实际离线运行：`reported_source_archive_consistent`，238 selected/236 numeric/2 missing，network/neural/physics=0；显式禁止其urlopen，原HTML/PDF-text重建等于R，不只是抄旧说明文档。
- 实际运行：`OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_d2p_baseline_reported_values_20261027_v1.py tests/test_agent_CEM8_baseline_table_20261028_v1.py`，**47 passed in0.90s**。这些是软件/保存数据检查，不是科学效果证据。
- 本文只读一致性检查通过：19个绝对本地链接均存在，13个来源表SHA逐文件一致；六法各两根均值/完整分母、5个代表原表cell ID与protocol_quotes、4份原HTML路径均核对通过。未增加测试文件；首次从外层调用不存在的`.venv`未执行，改用项目内既有解释器后通过，没有安装依赖或环境回退。
- 本轮只复验所列JSON/来源哈希及描述性分母/算术，不重复MAIN完整代码审计、所有consumer/NPY/大权重闭包、物理reset或模型编码；没有新增物理/神经结果。
- 仍未完成：各官方方法到本地的全协议/资产映射；Hi-LeWM完整配对资产；PlanWAM本地驾驶pipeline/PDF页；LeWM Table10梯度预算和±定义；具名MPPI/GN/kinodynamic外部原表与闭环gradient MPC本地评测；独立freshness和P1。unknown表示本轮未认证，不断言原作者没有材料或该方法无效。

本文的完成只指覆盖清单与缺口交付；不升级为官方baseline复现完成、全算力公平性、跨论文排名或研究goal完成。
