# 后续 P 阶段实际推进（MAIN，2026-10-28）

## 最新终态覆盖（后续推进）

### P2八midpoint实际CUDA、strict及MAIN独立数组复算完成

修正config派生token数及数值守卫后，新prepare SHA `1db6db94128dc794e98b0610780fe691c7dc68cd725cff0bdfac06f3ddfa54d6`。固定全四known根八臂主机CUDA实际完成，12.87269秒、encoder/projector各8次attempt/entered/returned/saved，errors=[]、model unchanged/source verified，MATH-only、两TF32-off/autocast-off前后及逐模块守卫均满足。无predict/AD/physics/训练。报告 `results/agent_p2_midpoint_teacher_20261028_v1_cuda_MAIN01/report.json` SHA `7b0b5c00735a7326326a4619c86539fdb51b1bf99f384fe8794a71a6b4e6321c`。

完整strict消费 `results/agent_p2_midpoint_teacher_20261028_v1_consume_MAIN01.json` SHA `4e61d9251f86550d0830c33d5607cfe58d4b308b97865f7fc64de0013d25d135`。MAIN另新增不导入producer/model/环境的8数组独立Q/MSE与全部四对复算，产物 `results/audit_p2_midpoint_MAIN_20261028_v1_MAIN01.json` SHA `cff183a8726af0e9f19a16f6b68534f60651edf289bdbf7331b95f1d83ca66ae`。root0相同literal prefix5、midpoint预测及实际Q差均0，终态反转；root2在step5成对排序一致、step10反转；root3在step5已反转；root1两边界均同向。不能凭两个packet边界确定具体子步或因果根因。八臂仅七张unique图、四known根，不是八个独立样本，也未改变旧0/4成功。

下一最小隔离实现已派：全八臂固定原suffix5，以实际midpoint latent替换旧预测midpoint后做单packet条件预测，保目标/Q/动作/历史，其余不变；复用既有真实终态编码，不训练、不筛根、不新增物理。此任务交付和实际运行前不得声称排序修复。

P3完整verify+plan亦实际完成，plan `results/agent_p3_child_scheduler_contract_20261028_v1_plan_MAIN01.json` SHA `4582d301d2a209bdc153bc7ea5aa9970524a8bf92d9802b323dbae44e2297557`：24child/48方法，46缺新paid child signal、2not_reached；预算拒绝数保持null，不把缺信号计作预算负结果。新的child CUDA接口继续实施。

### 新beam真实RGB桥已完成；P3身份接线修复实测通过

后续更新：P2现已实现从绑定config推导token数、MATH-only/cuDNN及matmul TF32-off/autocast-off前后与逐模块守卫；MAIN审读修改并复跑53项测试通过（5.82秒），不是CUDA证据。P4 saved consumer与P5 contact-source合同经MAIN完整审读，联合48项回归通过（6.26秒）。P4严格检查显式保留缺逐步witness actual-action/goal-pose遥测的阻塞，已要求新producer/consumer版本补齐而非改出口假通过。P5固定可接触几何仍是静态OOD候选，新native生成器/strict实现继续推进；没有观测接触或物体响应时不宣布前置满足。

P3实际saved prepare已完成：`results/agent_p3_child_scheduler_contract_20261028_v1_prepare_MAIN01.json` SHA `8669fb219c6a17e05d5c1b8dc557b401651a2bf3fc759529d64231e5d317f7fd`，1,725,702bytes；输入manifest SHA `7d4446da2a9038ab1251e3c53188b0eaa0af398384f80bf6dd58503f222f4b18`。完整verify/plan随后提交，终态须查实际产物，不能将prepare代替child CUDA/动作或C3结论。

beam新增八张原封存PNG的主机CUDA encoder/projector各8次完成，0 predictor/AD/physics，errors=[]，冻结模型未变。报告 `results/agent_kinodynamic_matched_rgb_20261028_v1_cuda_MAIN01/report.json` SHA `52e6d7c025dcb836aed3ec28aefc9b249d449be2eb19618c8e6ebdae7ea4db49`；完整严格消费 `results/consume_beam_rgb_MAIN_20261028_v1_MAIN01.json` SHA `9d4321544e5cdb1f4b83b83b20ad493820f25d3a41fdf03d32f8c1a8f2df487b`。两根root及goal重新编码均bitwise exact；step5/10的预测Q→真实RGB Q分别为root0 1.017835→1.079480、.575988→.854124，root1 1.263677→1.131222、.665642→.701696。两根native成功均false。仅known-root matched-domain beam，不是官方kinodynamic/RRT，也不证明等FLOPs或独立泛化。

P3 child合同真实saved回归先前捕获 `literal SHA256` 拒绝。MAIN核原task export：`native_root_sha256=null`，model SHA为合法字符串；故不能把null填入runtime根绑定。当前实现显式沿original_manifest→sourcepack取得同slot根SHA，并核blind payload身份；export原null与严格validator均不改。MAIN完整重读当前主体及新增真实回归，39项测试通过（5.24秒）。这是代码/metadata接线修复，不是child信号或C3收益证据；46个observed方法槽缺真实paid信号，2槽未到达不能填造。

P2 midpoint桥审读另发现输出token数硬编码197，而当前绑定checkpoint为224RGB/patch14、含CLS应257；数值执行还需补显式MATH/cuDNN-TF32/autocast守卫。已交独占实现者修复，未据此启动实际CUDA或记录科学负结果。修复、冻结与MAIN回归后才能prepare/run/consume。

本轮并行继续P1 teacher共同库、P2 midpoint、P3 child信号、P4 fresh-source严格消费、P5可接触校准源及matched基线并栏。主线程承担代码审计和实际执行，worker不运行神经/物理；预留fresh四槽保持未打开。交付或派发不算阶段完成。

### P2全四根旧新接线与两类排序失配已复算

MAIN完整审读新analyzer、45项回归通过，实际分析产物 `results/agent_p2_native_composition_gap_20261028_v1_MAIN01.json` SHA `fc9d1ae5ff7751821380ea92b1e07c85c374534abb1e7d8cc249d85c9dedd612`。精确核旧终态编码与新物理raw10/整条轨迹/图片/模型与Q身份，复用既有八个真实CUDA terminal vectors，没有重复请求终态encoder。

composition在预测Q/真实图片teacher Q/native7/jointXY/objectXY的距离胜出数分别2/1/2/3/1（分母均4，非success）；两臂native成功均0/4。root0/2/3的预测→真实图片Q成对排序反转；root0/1/3的teacher Q→native7排序反转。七维平方误差分解另将位置、原始角度、agent速度分列，非因果贡献。下一个最小增量已独占派发为全八臂step5真实图像编码及原midpoint预测对齐，不重跑物理、不事后挑根或改指标。

### P5全后缀像素/原生状态分析完成

MAIN审读308行saved分析及文档，26项RAM回归通过，实际结果 `results/agent_p5_word_physics_analysis_20261028_v1_MAIN01.json` SHA `3a60c94ad7c3c4f2590dc06c482c85f52c18435d4292eda0d95db72893b0b370`。保留12baseline、120后缀行、60同帧配对和三根模型端点特征并栏，完整raw RGB先int16相减。全六对中途与终点block位姿均相同、RGB有差；推杆最大位置差XY/YX约27.21122、cancel/zero约19.24124。这不是物体非交换性；全图差不能归因给物体。

### 后续实施与保留样本保护

P1八真实teacher anchor共同对称suffix库编译器、P2八midpoint图像桥、P3真实child-local调度接口、基线beam已存PNG评价桥已独占并行实施；交付不预算完成。P4四槽source adapter已审读，当前无合格历史认证。**不以calibration_only提前运行原四个预留fresh槽**，避免为了工程运行先打开确认样本；实现与strict消费可继续推进，真实确认运行须满足原证书前置。

### P3预算与真实指标分离核账完成

MAIN完整审读289行新saved analyzer和说明，37项回归通过；实际完整consumer-linked分析完成，产物 `results/agent_p3_cost_comparison_20261028_v1_MAIN01.json` SHA `89c6b67c579d82565d559b77f8a3b3042bcaa782a6b0aefa2dd2a7698be75cff`。

每臂实付signal8+search24+prefix1=33 predict API/sample、2J/384VJP，fixed/adaptive主要forward/AD轴完全相同；减少的是两根合计32个replay和16个metric计数，不能称神经算力/FLOPs或端到端时间节省。adaptive−fixed平均native7 −196.51751，而jointXY +16.12327、objectXY +1.91983、velocity error −245.51187。四臂native成功均false。因此native7下降不能冒充位姿或成功率收益；完整C3 child-local闭环树仍未验证。child-local调度接口实施已独占分派，不把同known-root核账写成独立确认。

### P5新原生物理与strict已完成

全三known根×四固定程序的原生物理重放完成，12/12完整后缀10步、global_error=null，执行7.6478秒。报告 `results/agent_p5_action_words_physical_20261028_v1_physics_MAIN01/report.json` SHA `796d757c6ffcf623f4e307f094dba9e39eda2a4810a140e5ab3cf0897b3c73a5`；完整strict回执 `results/consume_p5_physical_MAIN_20261028_v1_MAIN01.json` SHA `2248899fb03439bc4c52ee741dc0e2b497e15ce692d1d4d306f32fee7d476883`，状态all12_saved_native_replays_verified。252帧原生文件、逐帧前缀字节门、实际动作/返回/计量均保留。

MAIN新增不导入producer/环境的实际saved审计，核全部120后缀返回done/reward和132帧完整精度body位姿。全部12臂的block pose在frame10..20逐位不变，120个control-step返回的contact计数均0；不冒称观察了全部内部接触配置。XY/YX推杆终距均约1.56321、速度差约52.56803；X−X/zero推杆终距约1.10536、速度差约37.17121，block pose均相同。回执 `results/audit_p5_word_motion_MAIN_20261028_v1_MAIN01.json` SHA `54382af4fd0c4835ca0c1e315b90fff89a0344ebbb896acbf16b4f1358582201`。

这解释了为何本组action-word特征差异不能算物体重配置能力：实验实际变化的是推杆而非block；不能把本组无物体变化归为理论不可能，也不能用agent惯性证明物体非交换控制。任务目标/接触前置认证仍缺，task_success=null，P5科学准入不打开。后续saved像素/预测并栏分析已独占分派，不重跑模型或物理。

### 新增同时间signed对照已完成

`agent_mixedmode_zero_signed_comparison_20261028_v1` 的273行源码与文档经MAIN审读；固定全部四层五探针，真实assemble成功，产物 `results/agent_mixedmode_zero_signed_comparison_20261028_v1_MAIN01.json` SHA `7dff71c3977758da3174f2b4650860ddbdea972685b96127fd0fc765779368d6`。它核两份已验收strict receipt、原始journal及动作/时间/根目标身份，不重跑物理。

MAIN另写不导入producer/环境的独立算术复算全部112个response/secant变化，通过；回执 `results/audit_zero_signed_MAIN_20261028_v1_MAIN01.json` SHA `35e1c434e9e8db7b5a276f93a1d164dbf92f643f8250f4107b9fe1ee688312b9`。两份实现联合RAM回归47 passed in1.98s。

固定原root→goal方向的objectXY有符号secant变化（selected-prefix5减zero5；每held raw command）：root0/off x−103.50889、y+1.43542；root0/on x−105.96433、y−6.45799；root1/off x−.42916、y+88.39027；root1/on x−2.29898、y+85.47091。全部四层保留。列符号本身不能表示能力好坏（动作可以反向），响应/角度/agent与joint分栏完整保存。所有层两臂start-state/snapshot均不同；仅同elapsed，不是中介隔离、不证明C2或理论反例。

P5全12固定程序原生物理adapter经MAIN完整审读，补严native `done=False` 校验，38项RAM回归通过。实际prepare成功，SHA `d33b539c77b35b4256a5e7099e39cdd1da619d8e8c3466d6f753ca3fded351d9`；已显式主机提交真实物理重放。此处无神经运行，不是CPU神经降级；terminal仍以实际report及strict消费为准，不预填成功。

下文保留执行过程；本节覆盖其中“已提交/待核”的历史状态。

- P1 zero5同时间控制已实际完成20/20分支，errors=[]，400 public/4080 physics/1320 renders、40 resets；无神经执行。严格saved消费完成，回执 `results/consume_zero_and_rolling_MAIN_20261028_v1_zero_MAIN01.json`，SHA `7c22af94f7aebaabbe4857cb1797c1b7fe857020145829273fb2349b45ea8e19`，状态 `zero_time_control_responses_verified_not_admitted`。它控制elapsed time，不控制状态/速度/接触中介，不是P1准入。
- P4新增独立rolling visual接缝消费通过，回执 `results/consume_zero_and_rolling_MAIN_20261028_v1_rolling_MAIN01.json`，SHA `182d5ce14a6363b287607c511d3bc4dde85769618e8c8082029f090fb19c1232`。重建实际动作历史，逐个检查R的真实滚动latent及I的预测后继接线；不替代已有native/resource审计，不新增科学样本。
- P5原生action-word主机CUDA与strict消费已完成，终态见本页末节；不得再按“终态待核”计数。

本轮并行新任务已派：zero与selected-prefix的全固定signed探针对比；P3全根资源/终点评价分栏核账；P4独立确认根前置认证；P5全12固定程序原生物理adapter。仅实施与saved算术由工作者承担，代码审计和真实执行仍由MAIN承担。交付之前不计作完成；不把前置未通过改成科学准入。

本轮不把 P1 未闭合解释为所有下游工程均停工。科学准入仍严格遵守前置；已知根工程实验、独立接口实现、资源核算可以并行推进。所有神经执行仅主机 CUDA，冻结模型，不训练、不拟合、不放宽阈值。

## 当前独占工作线

| 阶段 | 具体任务 | 分工与边界 |
| --- | --- | --- |
| P1 | zero5 同时间控制，warm10→zero5→原五探针 | Poincare；20个新物理分支由MAIN审读后执行，不能声称状态匹配因果隔离 |
| P2 | 核查并补组合程序到实际执行的缺口 | Bohr；新 `agent_p2_composition_execution_20261028_v1`，不得冒用单程序或beam作为组合机制 |
| P3 | 固定执行5步的真实反馈与终点评价 | MAIN定位合同哈希方言错误，Hegel仅实现v2最小修复；不改旧封存证据 |
| P4 | 5/10/15三个真实反馈边界，20步控制，I/R对照 | Nash已交新driver，MAIN已完整审读源码与文档，准备真实输入验证 |
| P5 | 独立冻结架构可执行前置 | Pauli；优先已有JEPA-WM原生接口，不重复manifest或旧fidelity记数 |
| 强基线 | 同动作域beam真实raw10物理与严格saved消费 | Jason实现，MAIN审读并运行；非RRT或官方论文复现 |

主线程不委派代码审计，继续亲自检查各实现的实际输入边、失败分母、动作历史、预算及科学结论边界。各工作者不得运行神经/物理/网络或写MAIN产物；不得创建后代。共享旧文件与产物保持不动。

## 主线程本轮已验证

kinodynamic物理适配与P4多反馈driver联合RAM回归：**51 passed，58.96秒**。这仅是代码回归，不是CUDA/物理或科学证据。

P3 v1 prepare真实失败归因：原task_contract由 `codec.core.digest` 默认空格JSON序列化；新桥接误用compact JSON的 `fixed.digest`。两根用原函数精确复得封存权威SHA；保留v1失败，v2只修该边界，不修改权威SHA或绕过校验。详见 `D2P_SIGNED_RESPONSE_AND_KINO_MAIN_20261028_v1.md`。

新beam物理manifest已经MAIN排他新建，选择完全来自之前封存模型计划，未读物理结果挑根：`le-wm-main/results/agent_kinodynamic_matched_physical_20261028_v1_selection_MAIN01.json`，SHA `6ddecf6514a84504c8b4ca4dd8bfdbf1e6e21f3769200d61712b4c74f4703028`。固定2已知根，不是独立确认集。

## 新增真实执行与主线程验收

beam物理实际完成两根各raw10：errors=[]、无工程失败，40 public/408内部physics/44PNG，原生成功0/2。报告 `results/agent_kinodynamic_matched_physical_20261028_v1_MAIN01/report.json` SHA `01d5485208fe40a71a53ae20bb2920740894d6ef7c114c139a991f4707e5ebe7`。prepare SHA `2430db7624f3d24ba80af78313b32ac2be168d45383a484f3aed91d5270fab43`。

MAIN新增不导入producer/环境的独立算术审计，核1186事件完整链、全部实际intent/return与原receipt、44PNG字节及像素、20条plan native reward/stop和两终点距离。回执 `results/audit_kinodynamic_native_MAIN_20261028_v1_MAIN01.json` SHA `9b1a6abe186d1ab3d74a66832dbad89d4cb7ef88a31791d3f8bad98e3ac3df7b`。终态native7距离186.60936/135.08162，object position误差24.63567/94.35089；这些是两个已知根的实际结果，不等于优于D2P或独立泛化。新严格producer-linked物理consumer仍另行推进。

P3 v2 MAIN回归96 passed（21.77秒），实际prepare已成功，SHA `51837e4f8a97d6bed08031ceea70b549ea1bf893351c77e6061b698dc5763e85`；已提交主机真实物理执行。新版本仅修task_contract所属哈希函数，未重选计划。

P4 MAIN02准备前，MAIN01真实run因源码准备后变化被输入闭包正确拒绝。逐项比较仅source SHA变化（实现者尚在追加consume），未运行模型/物理，不是科学负结果。MAIN要求冻结三文件，完整审读新增consume，再运行28项回归（全部通过，58.98秒）；保留旧prepare，MAIN02新prepare SHA `8528e8d7f9b688b1c93be154305ccc35b8ed6218de63e736b1ae567691c432a2`，已提交主机CUDA+物理。今后修改必须新版本。

P5明确材料更正：Cube原生五维动作统计与source-native P0实际上已存在；旧工具allowlist未覆盖不是“全库无原生入口”。JEPA PointMaze当前缺原训练口径的既存冻结normalizer数值认证；不得用tail重拟合来解阻。详见 `le-wm-main/innerdocs/agent_p5_next_checkpoint_prerequisite_20261028_v1.md`；不是P5复制成立。

## P2 与 P4 已有本轮终态

P2四个原known根八臂实际完成，每臂warm15+raw10，40 reset/200 public（120warm+80selected），errors=[]，全部完整，原native成功两方法均0/4。报告 `results/agent_p2_composition_execution_20261028_v1_physical_MAIN01/report.json` SHA `ccb790aeb65313b391abaf9fcb671c24b8246f9e27a1e1bb162a3c233b1654b5`。strict saved回执 `results/agent_p2_composition_execution_20261028_v1_consume_MAIN01.json` SHA `2afd4879481fb236514e3d18dd3d219cbc12b3b29295e1f204b9cbe9d48fbb79`。使用原封存480候选库，不重跑神经；不是P1/P2科学放行。内部physics/render/constructor账未知，未补0。

P4 MAIN02真实主机CUDA+物理完成：252.824秒，errors=[]，model before/after相同。8臂中7臂完整20control，`e01_R`在11步native成功即停止，未补步；全部8臂和24编译槽保留（23已编译，1未达）。实际115predict/23J/4416VJP、30encoder、231public/2342physics/247render，不能称各臂实际资源相等。

报告 `results/agent_p4_moving_multiboundary_20261028_v1_MAIN02/report.json` SHA `60489b7db3d4eff70d1be46d47d92209e6adcf333ae1d0a0e389b259ec1f6084`；原driver strict消费回执 `results/consume_downstream_MAIN_20261028_v1_P4_MAIN01.json` SHA `29c760cc10a28748789303796e38b49efad6d786fe16e39bd077227216b4e83e`。MAIN另独立审计10040事件、247真实RGB、全部151control返回reward/native终止与完整原生7维终点评价通过；回执 `results/audit_p4_native_MAIN_20261028_v1_MAIN01.json` SHA `4f2f62029ef5e4209fcda4dc853a6404189fa820f9f8b4a35c4024ff16e9c60e`。MAIN新增独立audit回归4passed。

| 固定层 | I原生成功 | R原生成功 | I object终距 | R object终距 |
| --- | --- | --- | ---: | ---: |
| root0 GN | 否，20步 | 否，20步 | 49.38088 | 8.71860 |
| root0 D2P | 否，20步 | 是，11步早停 | 16.92510 | 9.21375 |
| root1 GN | 否，20步 | 否，20步 | 81.87737 | 123.53658 |
| root1 D2P | 否，20步 | 否，20步 | 47.35529 | 43.28384 |

该表完整保留四层，不挑成功层；R不是处处改善。object终距仅诊断，不可替代含agent的native成功；早停臂没有actual_Q20，不补终态20。只有两个known根，不能作泛化或机制正式证据。

beam新完整producer-linked物理strict消费亦完成，回执 `results/consume_downstream_MAIN_20261028_v1_beam_MAIN01.json` SHA `fda1644c2fbb035401c1e0ef02d11d2817f5fc731a13abc418517c879826a8eb`，与MAIN独立native审计分开保存。

P3 strict消费第二个错误由MAIN定位为warmup0 requested角未取模、native观测角按2pi取模。真实物理结果已完整，不重跑；独立新saved consumer仅在该边界复算原生投影并同时严格核snapshot未取模角/原bits，43项回归通过；完整原producer/prepare闭包消费仍在核终态，未提前当通过。

## P3完整严格消费完成；P5新接线的实际错误

P3不重跑物理，使用新独立 `agent_p3_fixed_execution_saved_20261028_v1` 修复placement角观测边界。MAIN完整审读312行、43项回归通过；实际完整调用冻结v2.verify及CUDA producer.consume、原物理所有204PNG/四臂/60公共步/native metrics/原始动作/源码闭包均通过。完整新回执 `results/consume_p3_native_saved_MAIN_20261028_v1_MAIN01.json` SHA `0ece90fd652a171cb6dda72ebd7385f99698d73faf126471da6b37f9bc0e4214`。

两根fixed/adaptive的native7终距分别286.14501/225.35502、638.91334/306.66831；jointXY分别140.30623/137.57747、211.20175/246.17704。即native7均下降，但第二根jointXY更差，velocity项会改变排序。不能把reward改善自动称任务改善或C3胜出，四臂均未native成功，仍无独立泛化/等FLOPs证据。

P2全部四对的composition−joint平均变化：official jointXY −32.05957、native7 +17.14238、objectXY +4.79865。保留全四根，不能只引用有利指标或将0/4当理论不可能；原动作库已封存，这一轮只检验它们的真实执行。

P5 PushT JEPA新增两包action-word候选（原生visual及proprio递推）MAIN完整审读543行和文档、35项回归通过；真实v1 prepare又捕获跨schema哈希方言错误，未进入CUDA：原v12继承v11紧凑JSON摘要 `9283f25481795ecee6c0f8bdfbf5ee9f83bee96b945bd4ca80e071332dd83aa1` 与封存report精确匹配；新代码误用自身默认空格JSON摘要 `d0bae714328b22b4123c00db2b2f39db767e7c27b78604d231f7c4318fc3d0a2`。MAIN逐项复核其他完整性predicate全true。保留v1与空prepare目录，已要求独占v2仅修原manifest owner摘要函数并补真实saved回归，不改变自有摘要/权威SHA，不把工程拒绝当跨模型科学负结果。

## 后续已实际提交的主机作业

P5原生action-word v2只修上游owner摘要，MAIN逐项diff审读和新增真实saved回归通过，44passed。真实v2 prepare成功，manifest文件SHA `1515cb228c6ab0a4fe826074324fd566781b06812a1e5163c41652492518a905`；已显式主机CUDA运行，三known根×四个两包程序、无physics、不训练，不放行action witness。终态仍待核，不能用准备成功作P5通过。

P1 zero5同时间控制主线程也完成505行审读、38项回归通过；实际prepare完整8,467,100 bytes成功，SHA `1fbd2607d64ba75aa17ae42833e20ab353c017fb4fec92aef6d6663c24c2c076`。已主机物理提交20分支warm10→zero5→原probe5；不重复旧before/after，不按结果挑probe，仍只是排除elapsed time混杂，不能隔离state/velocity/contact中介。

主线程P2独立原生审计额外完成全部80控制返回reward/stop与最终三种距离复算，回执 `results/audit_p2_native_MAIN_20261028_v1_MAIN01.json` SHA `2f2060f00512aa8710456202cdea289a6c7379a076a65eda9dff136442323c16`。P2/P4/beam MAIN独立审计、新P3 saved consumer、beam严格consumer联合回归 **77passed in6.35s**，无神经/物理重跑。

## P5本轮新主机CUDA及strict终态

原生JEPA action_words v2实际成功，6.368秒，errors=[]、finalization_errors=[]、三个冻结module before/after一致。三known根×4固定两包程序共12个完整预测；24 predictor API/sample、3 proprio API/sample，0视觉encoder forward/AD/physics。123原生完整数组保存并严格重建逐步visual/proprio/action历史和全部对比数值，无未来teacher forcing、不选winner。

报告 `results/agent_p5_native_action_words_20261028_v2_cuda_MAIN01/report.json` SHA `cdbccbdf894d8d6e7c078e02f7078bd8cd26e802c9a00d59421ad0e4b0883bbd`。MAIN实际strict消费成功并排他持久回执 `results/consume_p5_words_MAIN_20261028_v1_MAIN01.json` SHA `5237000615227a5cb6741e19c3feb4414c710077983399c46804b96c223b01db`。

| seed | XY−YX terminal patch MSE | X−X vs zero terminal patch MSE |
| --- | ---: | ---: |
| 2026102702 | 0.00103427 | 0.00065507 |
| 2026102703 | 0.00245490 | 0.00187251 |
| 2026102704 | 0.00204855 | 0.00264654 |

这里只证明当前已冻结JEPA原生递推对固定程序顺序/取消的特征预测不同，不是物体控制能力、真实非交换性、task witness或P5复制成立。所用原三根已打开；不是新根、新checkpoint或第三架构。

后续独占物理adapter任务已派Pauli：全12程序精确重建原source recipe/prefix10，不按对比值挑程序；必须先认证原任务goal/接触前置，缺则明确阻塞，不补造任务成功定义。MAIN仍亲自审读，worker不运行神经/物理。

后续实际prepare/run/consume以终态产物为准；本页不预填未来SHA，不将已启动当完成。P1/P2/P3/P4/P5科学闭环及原研究goal保持未完成。
