# GRASP、Hi-LeWM、LeWM与PlanWAM：作者报告与本地观测

本表把文献结果与本地观测分开呈现：**它们回答不同协议下的问题，不能据此计算跨论文或跨cohort提升。** 文献中的学习模型成功率、环境成功率、驾驶综合分，以及本地latent误差不是同一指标。unknown表示缺少可认证值或未执行，不表示0。日期：2026-10-29。

## 1. 如何阅读与引用

作者表复用现有[reported-only registry/table report](/home/gbk/code/LeWorldModel/le-wm-main/results/p2_reported_baseline_table_20261027_v3/export/report.json)，以及其中每项`source`的paper locator：版本化URL、HTML子表ID、原始行列、token、PDF页和SHA。下文R/C均为**原解析表的0-based行/列索引**，不是试验编号；PDF页为1-based物理页。原表数值均有现存官方HTML缓存及SHA定位，本轮四份HTML重新核SHA相符，标为`paper-reported/source-bound`，**不是本地复现或本轮重新验证作者实验**。未认证的checkpoint、seed IDs、process和逐项成本仍写unknown；若离开这些原表锚点，不能将其它附件数字升级为已认证值，应标`reported-unverified`。

范围按方法和原表预先明确，不按成绩选行：GRASP原Table 1/3/4的GRASP完整行；Hi-LeWM原Table 4全部23配置×4列；LeWM原Table 10两模型全部四solver；PlanWAM已登记的两NAVSIM主指标与HUGSIM完整难度列。没有声称穷尽所有论文ablation；Hi论文自己复现的flat LeWM Table 3不与原LeWM Table 10混为一次实验，仍可在原registry查看。

### 逐项适用的输入权限与预算

后续作者数值表每行继承相应协议码G/H/L/P；本地表继承A/B/I。预算为原单位，不把candidate数、模块调用、API数、samples或wall互换成FLOPs。

|码|任务/输入权限与模型|预算、时域与统计范围|尚未知或不可替代|
|---|---|---|---|
|G|GRASP：DINO-wm学习动力学中的Push-T/PointMaze/WallSingle目标规划；优化虚状态不等于访问真实中间状态。原记录未完整认证observation ABI/输入权限清单，保留unknown|每setting 500 trials；按环境/H sweep后选超参；DINO-wm每次输入5动作、预测5步。H按原表报告，primitive/model-step映射unknown|实际forward/backward/FLOPs、seed IDs、原表checkpoint及精确权限unknown；不是原生物理成功率。Table1仅成功trials计时；Table4计时分母unknown|
|H|Hi-LeWM：当前图像+同一专家轨迹未来目标图像；目标图像是任务输入，不是中间未来观测oracle。低层冻结，已训练高层macro encoder/predictor；-C还使用训练动作构造的empirical macro bank|50 episodes/config，通常3 seeds；本表±为跨seed SD。d=25/50/75是原环境目标偏移；各配置(hh,hreplan,hl,rl)见表，不能把d当horizon；CEM预算见下表H25/H50/H75|逐episode IDs/checkpoint配对/全流水线FLOPs unknown；训练资产/数据权限不等于本地无训练方法|
|L|LeWM Table10：Push-T图像目标规划；通用协议224×224、5动作一block，offline轨迹抽start/goal；原文未据此完整认证每cell的输入权限/具体checkpoint|通用PushT：H5 blocks=25原步，执行5blocks后重规划，goal offset25、环境预算50；CEM通用300 samples×最多30iterations、elite30。不能当作每个Table10 cell的完整运行绑定|SGD/RMSProp/Adam优化预算unknown；seed数与±统计定义unknown，不称SD/SEM/CI；精确split/process/逐cell资源unknown|
|P|PlanWAM：部署4历史帧×4camera并含ego信息；未来多视角posterior仅训练，部署history prior。DINOv2 base冻结但rank32 LoRA与规划模块训练|NAVSIM navtest；64条候选、每条8waypoints/4s；HUGSIM zero-shot 436scenarios。不是64次WM predict的证明|测试全pipeline成本、seed IDs/误差条及本地checkpoint/evaluator闭包unknown；训练未来权限不能写成测试oracle|

|Hi预算码（Table5 p14，`A3.T5.3`）|d（环境步）|环境执行预算（步）|高层CEM samples×iterations / top-k|低层CEM samples×iterations / top-k|
|---|---:|---:|---|---|
|H25（原R2）|25|50|900×20 / 10|300×30 / 150|
|H50（原R3）|50|100|1500×40 / 10|900×30 / 150|
|H75（原R4）|75|150|1200×60 / 10|1200×30 / 150|

上表是每次高/低层搜索参数，不是episode累计成本；还需真实重规划次数和模块账，不能相乘后声称整段等算力。

## 2. 作者报告值（不含本地结果）

### GRASP — 2602.00475v1，协议G

[原文Table1](https://arxiv.org/html/2602.00475v1#S4.T1)，PDF p7；子表`S4.T1.3`，GRASP原R5。成功率越高越好，时间越低越好；两者来自同一setting，不是两组独立样本。

|任务|原H|学习模型成功率（%）|成功trials时间中位数（秒）|原C（成功率/时间）|
|---|---:|---:|---:|---|
|Push-T|40|59.0|8.5|1/2|
|Push-T|50|43.4|15.2|3/4|
|Push-T|60|26.2|49.1|5/6|
|Push-T|70|16.0|79.9|7/8|
|Push-T|80|10.4|58.9|9/10|

[原文Tables3/4](https://arxiv.org/html/2602.00475v1#S4.T3)，PDF p8；子表分别`S4.T3.3`/`S4.T4.3`，均原R5。Table4只明确completion time中位数，不能套用Table1的成功trials分母。反馈细节在所用登记中unknown。

|任务|原H|学习模型成功率（%，Table3）|completion time中位数（秒，Table4）|两表原C|
|---|---:|---:|---:|---:|
|Push-T|10|100.0|1.5|1|
|Push-T|20|89.2|2.1|2|
|Push-T|30|75.2|9.1|3|
|PointMaze|10|100.0|0.7|4|
|PointMaze|20|100.0|1.8|5|
|PointMaze|30|99.8|2.1|6|
|WallSingle|10|95.2|1.6|7|
|WallSingle|20|91.8|1.8|8|
|WallSingle|30|95.0|2.1|9|

### Hi-LeWM — 2607.12547v2，协议H

[原文Table4](https://arxiv.org/html/2607.12547v2#A3.T4)，PDF p13，子表`A3.T4.3`。下面全部数值为环境成功率%，mean±跨seed SD；列依次原C7/8/9/10，标准/约束变体分别保留，不挑最优配置。元组顺序严格为(hh,hreplan,hl,rl)，预算逐行由d对应H25/H50/H75。

|原R|d|模式|配置元组|PushT Hi-LeWM（%）|PushT Hi-LeWM-C（%）|OGBCube Hi-LeWM（%）|OGBCube Hi-LeWM-C（%）|
|---|---:|---|---|---|---|---|---|
|2|25|online|(1,5,2,1)|89.3±4.1|88.7±4.1|67.3±1.9|67.3±3.4|
|3|25|online|(2,5,2,1)|28.0±8.5|28.7±5.2|69.3±2.5|54.7±0.9|
|4|25|online|(1,5,3,1)|70.7±4.1|75.3±0.9|64.0±0.0|64.7±0.9|
|5|25|online|(1,5,5,1)|44.0±1.6|40.7±0.9|64.0±3.3|61.3±0.9|
|6|25|online|(1,5,5,5)|79.3±4.7|80.7±3.4|66.0±2.8|68.0±4.3|
|7|25|online|(2,5,2,2)|23.3±3.8|24.0±8.6|70.7±0.9|52.7±2.5|
|8|25|staged|(2,5,2,1)|90.7±6.2|70.7±5.2|65.3±2.5|60.7±2.5|
|10|50|online|(1,5,2,1)|38.7±9.0|48.7±3.8|52.0±3.3|54.7±3.4|
|11|50|online|(2,5,2,2)|27.3±1.9|32.0±2.8|52.0±0.0|44.0±2.8|
|12|50|online|(2,5,2,1)|35.3±6.6|34.7±3.4|50.7±3.4|48.7±2.5|
|13|50|online|(2,5,5,1)|16.0±5.9|22.0±5.9|48.0±1.6|42.7±2.5|
|14|50|online|(2,5,5,5)|20.7±5.7|18.7±4.1|48.0±1.6|44.0±1.6|
|15|50|staged|(2,5,2,1)|42.0±2.8|64.0±3.3|46.7±2.5|48.0±0.0|
|16|50|online|(2,3,2,1)|34.0±4.3|38.7±6.2|52.0±1.6|48.0±2.8|
|17|50|online|(3,5,2,1)|24.0±4.3|20.7±5.7|47.3±0.9|29.3±5.2|
|19|75|online|(1,5,2,1)|12.7±3.8|19.3±0.9|66.7±6.2|69.3±6.8|
|20|75|online|(2,5,2,2)|13.3±5.0|18.7±4.1|56.7±4.1|43.3±5.2|
|21|75|online|(2,5,2,1)|15.3±3.8|32.7±3.8|58.7±4.7|42.7±4.1|
|22|75|online|(2,5,5,1)|10.0±4.3|16.0±3.3|42.0±1.6|37.3±2.5|
|23|75|online|(2,5,5,5)|10.7±4.1|17.3±0.9|46.7±0.9|42.0±5.9|
|24|75|staged|(2,5,2,1)|15.3±4.1|22.0±3.3|49.3±0.9|50.0±4.9|
|25|75|online|(2,3,2,1)|13.3±0.9|25.3±8.2|58.7±4.7|41.3±2.5|
|26|75|online|(3,5,2,1)|14.0±0.0|12.7±2.5|46.0±5.9|40.0±3.3|

0.0是原报告的SD，不是补零；也不能据此推断没有实验随机性。没有在表中另标winner。

### LeWM — 2603.19312v3，协议L

[原文Table10](https://arxiv.org/html/2603.19312v3#A7.T10)，PDF p26，子表`A7.T10.6`。任务均Push-T，成功率单位%；LeWM/PLDM分别原C1/C2，四solver全保留。**±统计定义unknown**，不改写为SD；v1没有此Table10。

|原R / solver|LeWM SR（%）|PLDM SR（%，同篇对照模型）|cell优化预算|
|---|---|---|---|
|1 / CEM|96.0±2.83|78.0±5.0|300×最多30、elite30仅为通用协议；逐cell绑定unknown|
|2 / SGD (β1)|26.0±4.32|4.67±0.06|unknown|
|3 / RMSProp (β2)|67.33±2.49|49.33±8.26|unknown|
|4 / Adam (β1,β2)|84.0±7.12|80.0±3.27|unknown|

### PlanWAM — 2610.11382v1，协议P

PDF页unknown；一手版本化HTML子表/原R/C明确，缺PDF页不等于数字没有原文出处。原分数均0–100、越高越好，**不是PushT成功率%**。本地驾驶结果全部unknown。

|任务|指标/单位|作者值|table-row locator|
|---|---|---:|---|
|NAVSIM-v2 navtest|EPDMS，0–100分|90.9|[Table1](https://arxiv.org/html/2610.11382v1#S4.T1)，`S4.T1.6` R16 C11|
|NAVSIM-v1 navtest|PDMS，0–100分|93.8|[Table2](https://arxiv.org/html/2610.11382v1#S4.T2)，`S4.T2.1` R19 C6|

|HUGSIM指标（0–100分）|Easy C1|Medium C2|Hard C3|Extreme C4|Avg C5|table-row locator|
|---|---:|---:|---:|---:|---:|---|
|Route Completion|87.6|48.3|51.9|37.2|53.7|[Table3(a)](https://arxiv.org/html/2610.11382v1#S4.T3)，`S4.T3.1` R6|
|HD-Score|75.3|32.9|35.6|22.0|38.7|[Table3(b)](https://arxiv.org/html/2610.11382v1#S4.T3)，`S4.T3.3` R6|

436场景；Avg沿原场景加权值，不把四难度做等权平均，不从已舍入列反推未舍入值。

## 3. 本地实际结果（不与作者值并列排序）

### 本地输入权限及评价合同

|码|任务/输入权限/版本|执行与预算边界|
|---|---|---|
|A|原两moving known roots、off；LeWM frozen224RGB→CLS192，真实3latent/2incoming packet历史、原goal192/identityQ；planner只用模型可见输入，不用物理outcome调参。GRASP为本地bounded机制适配，不是官方DINO-wm复现|固定H2×5=10raw步、开环，各方法两根均完整执行；表A保原六方法×两根。GRASP名字中的H3指历史长度3，不是执行15步。预算按两根合计见每行|
|B|同原off两任务的matched-domain比较，包含beam/GN/D2P/新CEM/nominal-MPPI；原目标/历史/Q不变，所选动作不同；planner无原生reward/state oracle|固定10raw步；两identity partitions各五臂。表B预算按每根，不与表A旧采样配置混并；同数学admissible ball不等于有限搜索support、数值验证epsilon相同|
|I|Hi owned checkpoint真实RGB root/goal→高CEM→官方FIRST waypoint→低CEM，固定两known roots×off/on四槽；不是A/B同一backbone等价证书。低层singlecontext z_init、zero incoming a_hist，与moving full3历史不同|模型内H2×5=10个normalized primitive动作输出，**未执行这10步物理**；每槽samples8、3iterations、topk3；seed_base1729。无原始inverse-process authority，raw actions和物理成功率unknown|

表A/B的Q为192维identity mean-square误差，单位latent²、越低越好；模型预测Q与真实终态PNG编码Q分列。native success沿原agent+object XY及angle判定，不以object-only、velocity距离或Q代替。下表每对数按slot0 / slot1排列，保留逐根值、不计算跨cohort均值或提升；显示6位小数，原始精度留在引用JSON。所有负结果保留。

### 表A：原六方法、完整12物理行

来源为[本地comparison v3](/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_baseline_comparison_20261027_v3_main01/comparison.json)，本地版本20261027-v3；row locator=`rows[0..11]`，顺序如下。共同原rootQ为1.154187 / 1.388805；原先数据集/normalizer provenance未认证，不能冒称论文表级复现。

|本地方法 / rows索引|模型终态Q（0 / 1）|实际终态RGB Q（0 / 1）|native成功/分母|predict总账（两根）|额外导数/线代总账（两根）|
|---|---|---|---|---:|---|
|开环gradient / 0,1|0.261319 / 0.072011|0.454880 / 0.046496|0/2|272|64 scalar backward；AD-forward128为predict子集|
|旧CEM / 2,3|0.474961 / 0.139051|0.364321 / 0.125224|0/2|268|0 backward|
|旧nominal-MPPI / 4,5|1.156129 / 1.341633|1.106069 / 1.350351|0/2|144|0 backward；不是完整官方MPPI|
|旧beam16×4 / 6,7|0.596706 / 0.353962|0.558420 / 0.292927|0/2|168|0 backward；constant-packet bank|
|GRASP bounded / 8,9|0.256188 / 0.069124|0.378680 / 0.093872|0/2|3604|1800 scalar backward；虚状态/soft dynamics适配|
|joint-H2谱 / 10,11|0.678326 / 0.531871|0.343007 / 0.544438|0/2|176|2J/384数学VJP；4SVD/2bounded solve；部分API账unknown|

这不是等预算表；含GRASP所有两根，不以0/2外推方法理论不可能。原source8encoder/8projector共享一次；实际post-physics评价共24encoder/24projector，单列而不重复算成搜索predict。constructor等旧物理计量边界不同，未知轴保null。

### 表B：新matched-domain完整10行（不能替换表A的旧CEM/旧beam）

来源为[同身份saved比较](/home/gbk/code/LeWorldModel/le-wm-main/results/agent_baseline_matched_rgb_comparison_20261028_v1_MAIN01.json)，本地版本20261028-v1。root0的rows0..4与root1的rows5..9按以下方法顺序对应；每行均native0/2，每根实际10步。

|本地方法 / rows索引|模型终态Q（0 / 1）|实际终态RGB Q（0 / 1）|predict API/每根|其他资源/每根|
|---|---|---|---:|---|
|beam20→6 / 0,5|0.575988 / 0.665642|0.854124 / 0.701696|144|samples144，0AD|
|line-search GN / 1,6|0.600330 / 1.200545|0.482175 / 1.374904|78|4J/768数学VJP/4SVD/4solve；单独samples账unknown|
|conditional D2P / 2,7|0.654757 / 0.166067|0.182659 / 0.198952|66|11J/2112数学VJP/12SVD；单独samples账unknown|
|matched CEM / 3,8|0.452012 / 0.093734|0.530768 / 0.138107|144|3×24完整H2；samples144，0AD|
|matched nominal-MPPI / 4,9|0.619275 / 0.703472|0.407476 / 0.906007|144|3×24完整H2；samples144，0AD|

只匹配beam/CEM/nominal-MPPI的predict API/samples轴，不是全五臂等FLOPs/AD/总成本。实际RGB三cohort编码分别beam8、sampling14、GN/D2P8，均另有对应projector；shared source8/8继承一次。物理与采集成本不退款。完整资源向量及数值support差异保留在原报告，本文不归并成一个“算力”标量。

### 表I：Hi-LeWM真实神经接线结果，不是环境baseline成功率

来源为[真实low-CEM报告](/home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_low_cem_20261029_v1_cuda_MAIN01/report.json)及[完整消费回执](/home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_low_cem_20261029_v1_consume_MAIN01.json)。模型原值不变，四槽均returned、errors=[]，本次low-CEM阶段wall17.9644秒；不能与GRASP成功trials中位时间比较。

下列成本是**末轮top3候选的terminal-to-FIRST-waypoint平方误差和，再对三个elite取平均**，单位latent²，属于CEM原始logging `outputs.costs`。不是192维mean-Q，不是最终分布均值动作的成本；原runner明确`final_mean_evaluated=false`。不再预测或重算神经结果。

|root / condition|原report rows索引|末轮elite平均成本（latent²）|native物理成功率|原始inverse/process|
|---|---:|---:|---|---|
|0 / off|0|12.602968|unknown（本阶段未执行）|unknown / release关闭|
|0 / on|1|36.573673|unknown（本阶段未执行）|unknown / release关闭|
|1 / off|2|25.348715|unknown（本阶段未执行）|unknown / release关闭|
|1 / on|3|40.080776|unknown（本阶段未执行）|unknown / release关闭|

本阶段真实资源：12 low-cost calls、96 candidate rows、24 action_encoder/24 low_predictor/24 low_pred_proj；predictor input_rows192/context_tokens288，projection_rows288；0新encoder/0新projector/0AD/0新physics是**本阶段成本**，不是整个资产链从未采集物理。继承RGB-high-low阶段实际8encoder/8projector、12high-cost/56high-predictor、4selected-high-rollout及4low-cost诊断等原资源另列于`inherited_resources`，不能隐去或重复加入低层24calls。没有common-backbone equivalence、官方搜索预算或完整闭环baseline声明。

### 本地尚不可填写的列

|方法|官方原协议本地成功率|原因|
|---|---|---|
|GRASP|unknown|已有bounded本地物理0/2，但非官方DINO-wm/500-trial/原表checkpoint与H合同|
|Hi-LeWM / Hi-LeWM-C|unknown|真实高低层神经接线不是物理成功率；缺认证原process不能inverse-release；-C empirical bank合同另缺，不能用连续CEM冒充|
|LeWM Table10各solver|unknown|已有A/B适配值，但原表checkpoint/split/process、逐solver预算与统计绑定不足；不能用原eval现场fit来补认证|
|PlanWAM|unknown|缺认证官方本地驾驶pipeline、checkpoint/config、4camera/ego数据split和evaluator闭包；不以PushT或旧同名PointMaze结果替代|

因此作者→本地、A→B、I→A/B的数值直比均`comparability=false`；不报告跨cohort提升，不从两个known roots作独立泛化或P1通过结论。

## 4. 出处、字节绑定与本轮检查

论文原文缓存位于 `/home/gbk/code/LeWorldModel/le-wm-main/results/p2_reported_baseline_table_20261027_v3/sources/`。原table report SHA256为`52e687811e6d6f728ab3eda60836ea05eff11f36da2bb4af975d1dd19ef380f1`；本地A为`96cf27b143bb1a839edf7bc1e4de02f4a97e433c6475c4c731c78af76300d32e`，B为`eb661bfd89b75b4865b37ba517de4412f015472928a9ed3c2b76ec15367ed128`，I原report为`68d23f06b88114d3f3a3df227cefe7ca247cb399ba452defbf5456dbda9f32d0`，I strict receipt为`96be2b84aadabc4ca23a8e84f2f54f889b86367d41d4d344e0d7d56b46e3b43e`。这些历史strict记录存在；本轮未重跑任何consume。

|论文|官方缓存HTML文件|实际SHA256（与各paper locator相同）|
|---|---|---|
|GRASP|`grasp_html.html`|`2a17a21d05d1792b0fa7b7bd9442be973af6ab37236263994a59a4eb4e6fa899`|
|Hi-LeWM|`hilewm_html.html`|`c4c7e725d862017334cbfd72c8e3e6dcdadb1ebff4aae45b3af847ba1e76bc7b`|
|LeWM v3|`lewm_v3_html.html`|`49feca90079f39674b3fe7f23de7cb6e76562c4d4ff5efd0a96743ca668e70b3`|
|PlanWAM|`planwam_html.html`|`67be22fee283aa4bad78578c91d83e6a3085bc8927c63597ee7c9721073bd873`|

机器定位可沿registry按`paper_key + source.table_id + row_index_0based + column_index_0based`找到每个作者值、原始token/row与协议quotes；不新造平行registry。A/B的上述row索引与精确key、I的`rows[i].outputs.costs`→NPY均给出可直接定位的值来源。

本轮只读JSON/原源码与缓存、重核绑定SHA、读取4个132B的saved costs NPY并核文件SHA/数据SHA/shape/dtype（`allow_pickle=False`），进行文档转录一致性检查。无新模型调用、NN导入/运行、物理、训练、拟合、下载或网络；没有重新执行旧consumer，没有改共享论文/封存源码/旧结果。数值表的验证仅证明转录/来源一致，不证明原研究因果或baseline公平。独占新增本文，冻结SHA随交付给MAIN。
