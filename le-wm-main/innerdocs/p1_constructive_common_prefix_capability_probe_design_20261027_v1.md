# Constructive两根：共同真实能力probe的最小后续实验设计

日期：2026-10-27。状态：**仅新文档设计，未实现新runner、未prepare、未运行物理／模型／编码。**
不修改shared doc、旧producer或sealed结果。本设计不是P1准入证书，不新增科学阈值、不训练、不拟合Q。
使用experiment-design技能的“先最小可核验诊断、再机制与确认”分层；不采用其通用训练／调参模板。

## 1. 当前剩余问题：先测真实响应，而不是再比较各臂own residual

当前两个known roots的D2P winner均为自身同库stage-Q greedy（index2、index7）。因此这两根目前
没有严格非贪心选择证据。物理有非零物体响应，官方success均0；这既不是不可控定理，也不是模型完成任务。
8+4张新实际RGB编码已经由MAIN完成。本设计只引用其canonical状态，不在此重认证CUDA数组或新预测桥接。

必须区分两种已发生的比较：

- 原GN与D2P跨臂：原selectedRGB receipt下，两根step5/10的同Q排序未反转。
- 相同D2P prefix下的后缀反事实重放：observed-context模型预测原后缀→所选新后缀的Q下降，
  root0为0.586243→0.539916，实际为0.856466→0.875436；root1为0.496337→0.487855，
  实际为0.385148→0.428327。是**后缀选择排序反转**，不是原跨臂结果被推翻。

四臂prefix精确重放审计支持原/新0..5 state、snapshot、命令和PNG匹配。这排除了这些保存对象的错根／
prefix漂移，不排除后缀模型、history接线、导数或渲染合同问题。v2 RGB producer本身predicted字段为null；
上面的模型值来自canonical对observed suffix compiler的记录，不能假称v2已独立复算预测误差。
checkpoint训练overlay合同仍unknown，不能将其当已证根因或据此改旧render合同。

现在最小可回答的问题是：**同根执行不同事前固定prefix后，施加同一个合法有限动作探针集，
对同一冻结任务方向的真实响应是否发生变化？** 它不要求模型猜对，也不读取各臂own J/own V作为答案。

依据为只读canonical：
[/home/gbk/code/LeWorldModel/docs/D2P_PARALLEL_CUDA_PROGRESS_20261027.md](/home/gbk/code/LeWorldModel/docs/D2P_PARALLEL_CUDA_PROGRESS_20261027.md)，
其中“真实RGB、反馈后缀和尺度一致GN”及其续报；以及既有mechanism prerequisites、physical consumer、
feedback prefix replay audit和selected RGB v2独立review。旧文档的“未来未执行”状态保持历史，不回写。

关键封存输入（后续runner必须显式path+完整外部SHA，不能glob最新目录）：

| 对象 | 项目内路径 | SHA256 |
| --- | --- | --- |
| 原planner | results/p1_constructive_blind_planners_cuda_20261027_v1_cuda_main01/report.json | 4dd5823e9a273b820d6b7b3f6a36a4a5a19b193383eaa9cbf94442082e79d98d |
| 原physical | results/p1_constructive_plan_physical_20261027_v1_main01/report.json | d205850e95f639b649fba9b5c6ee8059f2507431753a538b1ce30eccbd234300 |
| feedback compiler | results/p1_constructive_observed_suffix_cuda_20261027_v1_cuda_main01/report.json | ac8a65900b94639f94b3797e90f80e73fa0500e13a055629750dc4dc418aaf79 |
| feedback physical | results/p1_constructive_observed_suffix_physical_20261027_v1_main01/report.json | f864cc80a88eb289b578d271679d0bc895043a3b7bc798ae141097797a555b5e |
| feedback actual RGB v2 | results/p1_constructive_observed_suffix_rgb_20261027_v2_cuda_main01/report.json | 2cc16ea94dc76640a09ddb6d3d3cf8c70f171d9423b3896702715d48132306b1 |
| matched actual RGB v2 | results/p1_constructive_matched_ridge_gn_rgb_20261027_v2_cuda_main01/report.json | f6dbf29177769e8ec0be9e5ea38b928d201812a5ea44c07fbcedd0ebd3ebabbe |

这些SHA是本设计引用的既有记录，不是本轮重新读取每份report得到的新独立审计结果。

## 2. 三个角色、唯一位置与真正分母

本设计的三个“root positions”均为**相同原根经过5步prefix后的真实测量位置**：

1. Z：原根执行5个raw zero，得到zero-prefix位置。不是直接用原根当作零时长对照。
2. G：原D2P十候选库stage-Q最小的prefix，stable-index tie-break，按封存模型分数确定。
3. D：原D2P实际winner的prefix，按封存stage+own-V及stable tie-break确定。

不是GN的prefix，不是旧16候选稀疏任务库，也不按这次feedback最终物理好坏换prefix。
feedback不改变前5动作；它的chosen suffix、原后缀或最好native终点都不参与这三个角色的选择。
两根与三个role mapping必须在新probe任何物理之前封存，失败不补角色、不补根、不找下一个winner。

预检只读保存数组，独立核对十候选固定分母、模型stage/winner、prefix logits、float32 tanh raw5，
并对照原实际物理前五条命令。不得从certificate读取witness_commands或future帧填候选。

去重键：`source root+goal身份 / reset与placement recipe / warmup / renderer+env/source合同 / prefix raw float32 bits / action-history合同`。
相同键只分配一个`position_id`，保存全部`role_aliases`、候选index及各自原logit/hash；
真正执行的对象是raw动作，所以即使不同logit映到完全同raw，也可共用物理分支，但原模型身份仍分别保留。
不跨根合并；不因RGB、object pose或state7“接近”合并；不同action history即使终点相同也不合并。
浮点容差不得用作新的角色合并阈值，signed-zero处理若需要规范化须在新实现中预先明确而非事后改。

当前G=D，预检确认完整raw/history等价后应合并，通常每根只剩Z与G/D两个位置；若Z也相同再合并。
令每根唯一位置数为m_r，总数M=m_0+m_1：一般三角色上限M≤6；当前G=D成立时应为2≤M≤4。
若预检却得到每根三种不同prefix，应先报告与当前记录不一致，不能静默按六位置启动。

报告必须同时列：2个原始根、6个逻辑角色、M个唯一位置、21M个计划物理分支及所有alias；
当前D−G能力差因同干预结构上为0／同位置，不产生第二次效果观察。
**21条probe不是21个独立样本；G/D alias不是两个样本；两次重放也不增加root数。**
不要为了获得“非贪心对比”事后改选次优prefix。当前能比较的是Z与G/D的响应变化，不是非贪心优于greedy。

## 3. 最小真实共同probe：21条，单一固定尺度，无动作搜索

每个唯一位置使用完全相同、事前封存的suffix列表，固定5步、time-major十维logit坐标：

`P = [0, +rho*e0, -rho*e0, ..., +rho*e9, -rho*e9]`，`rho=0.25`。

0.25沿用既有same-prefix诊断设计的小尺度之一，是本次明确的工程干预幅度，不是效果阈值、成功margin，
未根据本轮新probe响应选择。仅一个尺度是最小完整坐标覆盖；不宣称跨尺度线性有效。
第二尺度0.5不是失败后自动补试选项；若需跨尺度诊断，应另在看本轮probe前完整预注册41条版本及全预算。

每条raw5×2严格用同一既有float32 `tanh(logits)`转换，验证`||logits||2≤0.75*sqrt(5)`；
不clip、不round、不改变action units、不把normalized token当raw primitive。
同一P用于所有根／位置；每条后缀从独立原recipe重建，而非依次接着上一个probe走。
所有动作及其SHA在第一次真实probe之前封存，**不调用模型、AD、优化器，不根据native outcome调动作。**

“共同”有两层，不能偷换：

- 同一position的21条试验：同一真实prefix、相同状态／history起点；只改变后5动作。
- 跨position：起点有意不同，但动作坐标、P、时域与任务测量方向完全相同。

zero suffix也是新真实5步动力学，不是零成本或原step5静态state。每个位置先执行其zero后缀，再按上述固定顺序
执行20条signed后缀；此顺序不取决于zero outcome。终止／guard拒绝只影响能否合法执行，不影响角色保留。

## 4. 从自身recipe重建同prefix，不能用hidden-state捷径

每条完整物理分支沿用原source规则：原reset seed、原placement recipe，两次reset，10个真实zero warmup，
原goal绑定，5个本position prefix动作，再执行5个probe动作。恢复动力学必须走原recipe；
不能直接把旧post-prefix state7调用setter当作同一真实起点，因为速度、接触和solver状态可能不同。

runner需要NEW接口和新封存，不改旧physical consumer的固定arms/预算，也不伪造compiler response：
probe是明确标注的固定干预，不是模型提案。原source证书仅允许evaluator读取重建所需白名单字段，
不读取witness动作或construction未来轨迹。probe executor和evaluator不向planner返回hidden state。

每个完整分支保留原fresh RGB guard强度与逐步持久化次序：PNG→fresh observation→guard→下一次dispatch。
每次原生terminal/truncation立即停，不因“还差两步”续走。记录原生success/reward和独立wrapped诊断，
不修旧native多圈角差bug、不把object-only指标改成success。

同position在所有probe的step5检查：state7 float64 bits、RGB完整bytes、raw动作float32 bits、
两body的position/velocity/unwrapped angle/angular velocity/force/torque/sleep等可导出snapshot均精确一致，
并保存action history。与该position第一条zero分支比较，不给不同根或不同position互相套相等条件。
对G/D位置，还对原实际prefix检查相关源、命令、state/PNG身份。
exact不符就标`prefix_not_reproduced`，不算可配对能力数据，不挑某次成功重建替换失败记录。
导出的snapshot不等于完整contact/solver cache序列化证明；仍需原recipe/版本、固定执行顺序与这一局限说明。

提前终止、guard失败、prefix不再现、I/O或预算失败均保留21M完整计划分母、attempt/completion和未开始项；
缺少该position zero完整终点时其增量不可计算；缺一对±时该坐标中心差分不可计算，不能丢列后算“完整能力”。
不以“实际response非零／native success=0”决定排除；完整性失败与科学负结果分开。

## 5. 共同任务方向：只做无拟合算术，不拿own residual换答案

主要能力诊断先使用evaluator原生几何，不需新神经。对根r，令p(s)为agentXY+objectXY四维位置（均px），
在原pre-prefix根s_r和原goal g_r固定

`d_r = p(g_r)-p(s_r)`，`u_r=d_r/||d_r||2`。

**同根所有position共享这一u_r**；从原root/goal定义，不从各位置新残差、最新结果或最好probe方向定义。
这个方向只在evaluator，永远不送入部署selector。若d_r恰为零，方向诊断为undefined而非换目标或丢根；
仍报告原始四维响应与固定分母。不拟合权重，不把角度rad和位置px拼成有利的混合norm。

设完整probe终点为s_{r,p,k}，该position自身zero后缀完整终点为s_{r,p,0}，定义

`Delta_{r,p,k}=p(s_{r,p,k})-p(s_{r,p,0})`，
`T_{r,p,k}=u_r^T Delta_{r,p,k}`。

报告全部21条的四维Delta与T，正号只表示相对该位置zero漂移朝**共同冻结方向**的有限响应；
不是已到goal或满足native success。zero漂移项`p(s_{r,p,0})-p(s_{r,p,step5})`单独报告。
相同k逐项比较位置间T差；不按native outcome挑probe/位置，不把最大值对应动作再下发。

每个十维坐标j还报告无需训练／拟合的中心差分

`B_{p,j}=[p(s_{p,+j})-p(s_{p,-j})]/(2rho)`，
`a_{p,j}=u_r^T B_{p,j}`，
`E_{p,j}=p(s_{p,+j})+p(s_{p,-j})-2p(s_{p,0})`。

B是单尺度真实secant，不声称无穷小真J；E显示非对称／非线性。全部a向量及其L2 norm可作为
共同方向的局部响应幅度描述（px/logit），不是新阈值、全局可控性证书或可执行能力值。
较大幅度可能需要反向控制、受接触/饱和限制；必须看signed有限T与E，不能只凭norm宣布有用能力扩大。
不从B拟合控制器、不算ridge解再执行、不SVD择mode，不将probe后验最优动作归给原D2P。

次级分解同时保留agent-only与object-only的原始Delta，不独立挑有利权重。角度另用
`q(theta)=[cos(theta),sin(theta)]`的终点减zero响应，并可投影至由原root/goal角定义的固定单位chord方向；
chord为零就undefined。这样无branch cut、无rad/px混权；原unwrapped角和wrapped误差并列保留。
原生success、native state7 distance（含速度）、位置／角条件逐项另报，绝不由T或chord改写。

各position的own goal residual、own terminal Q或native改善可作为**另一张进度表**报告，不能进上面的
共同方向定义或替代共同response表。若以后要测latent共同方向，须另封存原task identity-Q方向、
新增实际CUDA编码receipt及预算；本最小物理probe不拿现有8+4图冒充21M个新probe终点latent。

## 6. 明确工程预算：都是真实新成本，不向旧source退款

最小版本每position21个完整分支；每分支10warmup+5prefix+5probe，共20public。
沿用原native两reset及其内部substep：每分支2reset、204physics、22explicit render、2goalrender、
4setter、1goalbind、22张PNG。全部physics须hook计attempt/completion，不能只凭public步数推定实际账本。

| 计数 | 一般上限M=6（126分支） | 当前G=D且Z不同，M=4（84分支） |
| --- | ---: | ---: |
| reset | 252 | 168 |
| public step | 2520 | 1680 |
| internal physics step | 25704 | 17136 |
| explicit render / PNG | 2772 | 1848 |
| goal render | 252 | 168 |
| setter | 504 | 336 |
| goal bind | 126 | 84 |

实际prepare依据保存数组去重后冻结M、21M行、全部cap，不能利用物理结果决定M。
若当前Z也alias，按相同公式减少预算；logical roles仍6，独立roots仍2。
没有额外重试／重复数，失败不自动开新批。原始G/D模型编译和旧phys/encoding成本原样另列，不伪称本诊断免费。

本设计为新runner提出的工程cap：模型加载0、encoder/projector0、predict/action_encoder0、AD/grad/engine0、
J/SVD/solve0、GPU allocation0；MAIN-only物理。建议整个run wall上限600秒（非完成保证），
单个PNG≤1MiB，总NEW输出≤256MiB（包含partial，预留final manifest/report 2MiB），prepare≤256KiB，
全程200MiB disk floor；启动至少需cap+floor，逐次dispatch/写入继续检查。
若实际容量或wall不足，partial拒绝，不删帧、降guard、丢probe、改顺序或声称全量完成。
后续可在首次物理前另冻结更高合理工程cap，但不得依据已看probe结果扩预算补负例。

上述是可实现的预算规格，不冒称旧runner已经支持126分支。MAIN审查NEW runner后才可prepare/run；
本文件没有虚构CLI入口或声称准备包已存在。

## 7. 启动前置、最小实现交付与停止条件

执行前必须有一个NEW小producer/tests/doc，至少提供以下接口，不重写旧sealed：

1. `collect_saved`：输入原source/prepare的获批白名单绑定、两原task、planner及物理receipt的显式path+SHA；
   独立复算角色/动作、原task/goal、alias表、共同方向和probe数组。缺合法source材料就报missing，不补假capability。
2. `prepare`：只saved计算，生成NEW protocol，冻结source/env/render/model/task/normalizer、role mapping、
   rawprobe和所有cap、fixed分母、错误规则；MAIN可在首次执行前审核完整文件SHA。
3. `run`：仅MAIN显式authorize-physics，零模型加载，原recipe重建、逐步fresh guard、durable动作封存后dispatch；
   全事件hashchain、所有PNG/metadata manifest、first/secondary error、final report，超cap不重试。
4. 独立saved auditor：不调用producer.verify，不导入环境/模型，核对hashchain、每条raw动作与phase、
   prefix exact、PNG、official native/wrapped、全部固定分母、alias不重复统计，纯数组复算Delta/T/B/E。

mock测试须至少覆盖G=D与三角色全同去重、跨根不合并、终点相同但history不同不合并、slot交换、wrong task/goal、
prefix再现最低位差、partial/terminal/guard、丢zero或±、tanh/time-major/动作球、账本/时间/磁盘拒绝、
错误journal不覆盖first error、probe结果不得回流selector。mock通过不等于真实probe通过。

模型/后缀接线审计、AD对FD核查、渲染合同来源和同任务强采样baseline可并行做，但不得用其探索结果
事后换这批probe/角色。只要原source/动作合法、goal与物理receipt可核，几何probe本身不必等待
训练overlay来源完全闭合；但不能在闭合前把物理响应差归因于冻结模型的机制预测正确。

## 8. 诊断输出如何读，P1还缺什么

- Z与G/D共同响应不同：支持“这两个已知根上，这些固定prefix改变了所测有限任务方向响应”；
  不自动说明D2P使用非贪心重配置，普通接近／接触位置变化仍是可能解释。
- own误差下降而共同T/B不支持：只能说进度变化，未测得共同方向能力改善。
- 全probe无响应：仅限这一尺度、这些位置和5步时域；不称全动作球不可达、不回推合法witness不存在。
- 后缀模型预测方向与真实probe不同：只有在NEW模型参考也按相同actual history、P、时域、goal封存后，
  才可称有限响应失配；不能把本次无模型物理probe和旧own J直接比较后称导数bug。
- G=D：没有D对greedy的独立干预差；无论Z对G/D多大，都不能填严格非贪心门。

这是**机制诊断，不是P1独立确认**。两根已被看过大量结果，不能报告根级显著性或以probe数扩样本量。
严格P1仍需原门：模型动力学支持、合法两阶段、足够强direct困难/plateau证据、真实非贪心选择、
共同真实能力改变与真实任务收益、全资源强基线，以及新根独立确认；构造性witness成功不归给模型。

后续确认必须在未知root outcomes前冻结source family、完整有限池/生成尝试分母、task lineage去重、
全部选择器/候选/预算/metric/失败处理与已有科学判据；按原生success标签评价，不改success，不用native
outcome筛根/选role/挑动作，不让evaluator hidden state参与部署selector。
模型条件准入若属原P1要求，须用独立calibration规则并报告条件队列，不按D2P已赢或probe已好再次筛选。
新的native-success探索无需先强制筛出direct困难，但应称探索，不能替代原严格P1确认。
本设计既不补造epsilon/tau或统计门，也不撤销它们；无论这两根probe结果如何，均不能追认本批P1通过。
