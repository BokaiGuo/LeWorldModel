# P1下一候选合同：固定两两谱混合prefix，不改β

日期：2026-10-28。状态：**供MAIN审查的预注册实现合同，未实现新runner、未prepare、未调用模型或物理。P1=false。** 只新增本文，不更改任何封存源码、结果、阈值或共享总账。本文在已知两根结果之后提出，因此下一轮仍是校准研究，绝非事前未知根的独立确认。个人交付完成仅指文档完成。

## 1. 具体问题与不重复的边界

已读取以下现有合同与证据，路径相对项目根：

- `le-wm-main/innerdocs/p1_conditional_procedural_source_20261027_v1.md`及同名source：calibration2为有界8次proposal/attempt，随机reset与placement reset分开；重复RGB/zero历史只是bootstrap，不是真实三帧moving history。不能将该source直接冒充本轮moving任务。
- `le-wm-main/innerdocs/p1_conditional_task_admission_20261027_v1.md`及同名core：完整H2直接优化平台、合法模型见证、独立authority和有限来源队列的接口已经存在；没有这些材料不能以候选数填P1。
- `le-wm-main/innerdocs/p1_constructive_reachable_source_20261027_v1.md`、`p1_constructive_mechanism_prerequisites_20261027_v1.md`：真实witness定义可达任务，但不得进入blind部署输入；真实success不自动证明模型可达或direct停滞。GN短预算不等于平台，旧B3不能覆盖后来B8未停滞的事实。
- `docs/D2P_GEOMETRY_REAL_CAPABILITY_AUDIT_20261027.md`：旧full-geometry/stopK已对未来K做有限差分优化，新增方向未获选；真实反馈能改变接触/物体运动而未稳定改善任务。不再把“对未来K求导”当成尚未做过的新信息。
- `docs/D2P_FIXED_CAPABILITY_DIRECTION_RESULT_MAIN_20261027.md`：teacher上下文改善八锚点预测，但仍有误差/反转，只有两known roots。object-only响应不是含agent+object的完整PushT任务成功。
- `docs/D2P_ALLMODE_CAPACITY_REPAIR_MAIN_20261028_v2.md`及`le-wm-main/innerdocs/agent_allmode_prefix_20261028_v2.md`：实际v2 report SHA `c3e38899a12cd163d75918da79ef029adb14b2479f74f15bdfac0b6ffc57242d`；1032 predict/172J完成、strict consume通过。四项42候选选择仍与stage-greedy相同；预测非贪心候选虽有20/27/18/18个，却均不在(stage,own-reach) Pareto前沿。v1容量失败及已支付调用继续保留，不是理论负结果。

**待检验的最小假设**：全方向不等于全组合。42库覆盖10个基向量的两档轴向位移，加zero/DLS；没有系统覆盖两个谱系数同时非零的prefix。有限幅度下，rolling-context、接触相关预测和条件J变化可能具有交互效应，轴上被支配不推出二维混合也被支配。这里提出的是有限候选覆盖假设，不声称模型已经学会真实交互、一定存在Pareto解或已找到实现bug。

## 2. 单一算法变更：42+180=222固定prefix

新版本须独立命名并在任何新模型结果之前封存本文SHA、source/tests/config SHA；不能修改allmode v1/v2。以下是全部生成规则，没有数据依赖的方向筛选、步长搜索、权重搜索或早停。

1. 直接复用已封存allmode的root nominal/J、原active Q下`spectral_bank`给出的完整10×10右谱基、符号canonicalization及原42项。不得重新挑rank、删除小奇异值/null mode、改SVD阈值或根据旧物理方向旋转基。
2. 记完整基向量为`v_i`，`i=0..9`，原外半径`R=.75*sqrt(5)`；按以下字典顺序附加：`i=0..8`，`j=i+1..9`，`s_i=-1,+1`，`s_j=-1,+1`。
3. 每项logit为`u = R*(s_i*v_i+s_j*v_j)/sqrt(2)`。共`C(10,2)*4=180`项；每项两个谱系数幅度相同。仅使用原外半径，无第三半径、连续角扫描、物理几何输入或候选适应。
4. 沿用原kernel逐包logit球的screen projection处理浮点边界；保存投影前/后logit、norm、原helper收据。不得新增逐坐标raw clip。用原float32路径生成一次literal raw(5,2)，模型、独立重放、export、未来执行均消费同一权威buffer，禁止第二次CUDA tanh或再次除100。
5. 原42项索引0..41逐项保持；附加项42..221，保存`(i,j,s_i,s_j,R)`和原基矩阵SHA。保留全部重复raw、null响应和失败槽位，不去重、不退款。

归一的`sqrt(2)`防止将“混合方向”偷换成更大动作半径。这个设计只测固定等权两两混合，不覆盖一般稠密组合、全部方向、连续半径或更长horizon。SVD近简并基的数值不唯一仍是限制：固定软件/后端和实际basis数组，独立验证候选算术，不宣称跨平台基逐位相同。

## 3. 固定任务、评价、选择与新信息

校准分母仍为原两root×off/on四项，按`(root_slot,condition,task_id,blind_sha256)`join，不能按结果只做某一根或on条件。输入沿用allmode v2已认证父task闭包：原真实moving三帧、两包实际warmup raw history、同condition真实goal、原native192/Q、normalizer、模型与transform。只传获批blind task，不传certificate/witness/真实未来状态、能力probe输出或其排名。

每项均执行：fresh root nominal/J → 222个prefix各自预测 → **各自**fresh suffix nominal/J → 原`.002` bounded ridge suffix → 原非线性H2结果和独立H2 replay。条件context的latent shift、实际raw归一化action token和全历史metadata必须逐项保存。不能只在root解一次suffix后复制。

选择完全沿用`objective=S_i+V_i`、β=1与`(objective,index)`稳定argmin。`V`仍是原ridge残差加原惩罚整体除192，不改成非线性terminal T，不增加接触奖励或object-only Q。全体评分/选择先封存，再进行winner独立whole与segmented replay。重放不允许改变选择；不一致为工程拒收，不自动退回次优候选。

saved报告必须对四项全222候选给出：S、V、T、原index、raw/参数、winner、same-bank stage-greedy、zero、初始C0；额外标注旧42支持内的winner及新增支持是否进入Pareto前沿。Pareto按原(S,V)计算；严格描述用精确不等式，数值不确定性与既有authority裕量另列，不能新设“几乎非贪心”门。

真正新增信息是：固定半径下**混合坐标的非线性prefix→条件suffix响应**是否产生轴库遗漏的tradeoff，并且β1是否实际选择它。不是再次统计旧42 winner，也不是以model T下降声称物理收益。以下结果均事前接受：

- 无新增Pareto：只否定本固定两两库的覆盖假设，不证明全动作球无解。
- 有新增Pareto但β1仍选greedy：候选覆盖有所增加，但冻结选择器的非贪心机制未建立；不调β、不物理oracle代选。
- 新winner非贪心：只取得模型候选，仍需第6节三重检验；不得单凭模型宣布P1。
- 原42数值不复现、raw/history不一致或不完整：工程失败，全分母保留，不作机制负结果。

## 4. 计算预算：与旧42不等计算，不以padding匹配

令N=222，沿用allmode编排。每项root nominal+root ADforward共2 predict；每候选prefix1+suffix nominal1+suffix ADforward1+非线性suffix1+独立H2 replay2，共6；最终winner wholeH2与两个H1 replay共4。因此`predict=6N+6=1338`，不是把Jacobian前向漏掉后的计数。

| 资源轴 | 每root×condition | 四项总额 |
| --- | ---: | ---: |
| predict API / predictor samples（batch1） | 1338 / 1338 | 5352 / 5352 |
| action_encoder / pred_proj | 各1338 | 各5352 |
| AD forward（已含predict，不再相加） | 223 | 892 |
| H1 Jacobian(192×10) | 223 | 892 |
| 数学VJP方向 | 42816 | 171264 |
| 成功helper内部SVD | 224 | 896 |
| bounded-ridge suffix solves | 222 | 888 |
| root spectral_bank调用（含DLS等原处理） | 1 | 4 |
| 新encoder/projector、训练、物理 | 0 | 0 |

SVD为root bank原2次+每suffix1次；异常中helper内部完成到哪一步若无hook必须unknown，不伪记全部完成。root DLS及bounded-ridge可能有内部标量迭代，调用数与内部迭代/线性求解次数分栏，后者由runner如实测量或标unknown。数学VJP方向不是autograd engine API次数；FLOPs、kernel计数、同步GPU时间没有实测不能填0。共享历史16编码、父规划、旧42与失败v1成本独立列历史账，不能宣称新算法全管线免费。

无失败重试、无结果早停、无candidate去重减免；每操作attempt/return与原始数组及时落盘。前置/source校验失败在dispatch前终止；运行失败后四项均保留completed/failed/not_started，不补根。旧结果不能在新预算里抹去，也不能将5352predict与GN/CEM的相同某轴称等总算力。

工程预案而非当前运行许可：prepare≤2MiB，artifact总cap2GiB、final reserve1GiB、disk floor200MiB、wall前派发检查7200s，MAIN外部watchdog；MATH/TF32false、CUDA-only、模型before/after tensor hash。容量较旧dense row按候选比例放大，仍须新runner以完整密集合成数组实测canonical/压缩与final重复尺寸并由MAIN确认，禁止删完整module trace或降低精度过cap。未通过这项不能prepare/run。壁钟不是算力匹配指标，尤其并发时。

## 5. 新根生成与模型筛查：不从旧成功根挑下一批

本轮四项仅校准，不生成新root，不更换goal。若之后确认，必须先单独冻结source-family、有限slot顺序、所有seed/reset spans、尝试上限、split、历史排除metadata与覆盖authority。当前seed未知就保持未授权，不在本文虚构fresh seed或认证书；未知历史绑定不因新recipe命名而消失。

可复用原procedural source的有界proposal/attempt及reset源码审计，但必须明确其bootstrap历史差异；若选择moving/constructive family，另用对应真实warmup/source合同，不拼接旧source。constructive可达证书只能供builder/evaluator，不能作为222库的额外候选。全部构造尝试、失败和拒绝保留，无物理结果筛选、无凑成功quota追加seed。根身份须绑定actual placement后的state/render与goal；不能reset(saved poststate)近似重建。

现有`p1_conditional_task_admission_20261027_v1`可消费完整fullH2 direct轨迹及模型见证；其authority须独立校准并在确认前冻结B1<B2、全部初始化/restarts、epsilon_plateau、epsilon_escape、tau。本文不产生新数值。direct预算按该合同`R*[6*(B2+1)+2*B2]` predict/AE、`R*B2` J（fullH2 192×20），GN另`R*B2` SVD；witness verification另6 predict/AE。R/B1/B2/authority缺失则筛查blocked，不能以本轮223个H1 J替代这些fullH2筛查。

## 6. P1三重检验：必须同根完整链，不拼不同根的优点

### I. 条件任务前提与模型见证

模型/坐标/目标/history及fullH2导数链同一；direct全预算连续incumbent、合法两段whole/segmented replay与外部authority齐全。沿用原四条件：

`0 <= min L_direct(B1)-min L_direct(B2) <= epsilon_plateau`；
`L_direct(B2)-L_witness > epsilon_escape`；`L_direct(B2)>tau`；`L_witness<=tau`。

真实builder成功不替代模型tau，有限预算平台不叫数学驻点。确认队列只依原冻结条件规则；不能额外筛“222已经选非贪心”再把选中率当独立验证。

### II. 所选非贪心prefix确实改变真实任务相关能力

首先模型selected与same-bank greedy/zero/initial三参照分别报告；相对greedy stage更差不自动等于相对初始短期无进步。其次真实step5仍要用原native任务轴核短期代价，agent+object联合位置、wrapped角、state7距离/reward、native success分列；success不含速度，raw-angle原生异常保留并列wrapped诊断，object-only结果不得覆盖任务标签。

只在另封存的共同有限probe合同上测每个角色自身zero endpoint-relative响应，同时比较root固定公共残差和own残差；所有角色同probe/尺度/时域，含负号，zero惯性漂移不算控制收益。signed任务响应与unsigned谱norm分开；own-V下降可能只是残差变小，不叫可控性增加。必须使用实际prefix轨迹和同conditionRGB/history，不用imagined prefix代真实teacher，也不把末态近接触推断成整轨迹接触。当前全方向生成没有提供这项真实证据。

### III. 真实可执行收益与独立确认

选择先封存、执行原10raw commands；若新增step5反馈，另版本且所有强基线同feedback预算，不将开放环与反馈混比。比较完整H2 direct GN/GD、已明确实现身份的CEM/nominal-MPPI、同库greedy/zero及原提案仍欠的learned-subgoal等；缺者明确缺失，不能用privileged witness冒充方法。

同root/source/horizon/Q/目标/动作域/观察日程匹配才配对。资源向量不同只报对应预算前沿，不宣称同FLOPs；不靠无意义padding追平。原官方成功、任务距离和完整guard/失败分母先列，latent-Q仅作辅助。两个known roots上的正结果只算校准；独立root确认、原统计/阈值authority及强对照不足时，P1仍false。

校准物理若MAIN另行批准，可固定四项×selected/greedy/zero=12个角色槽位，角色碰撞仍保留标记、不事后换次优；不为碰撞伪造独立样本。按每分支2reset、warmup10+future10，上限24reset、240public（120warmup+120selected）、2448内部physics（含每reset2），freshRGB可按21个公共状态×off/on/off保守上限756显式render，另24次reset内部render。目标绑定/原producer额外render与constructor/close须在真正executor闭包里另核完整cap，未核齐则不启动；本段不是物理运行授权。能力探针不包含在这12分支，也不能用这12条终点替代II。

## 7. MAIN下一步、验收与停止

下一可执行实施项是**新的222-prefix runner**：仅替换bank生成，复用allmode v2的已封存数学/动作/模型桥和失败持久化逻辑，不monkeypatch旧globals。先做RAM候选算术/预算/序列化测试，再由MAIN审读prepare/verify，最后MAIN决定主机CUDA。尚无该runner，本文不提供不存在的CUDA命令或虚构prepare SHA。

最低工程验收：原42前缀逐项精确保持；附加180项索引/配对/半径/球投影可独立复算；literalfloat32 raw只生成一次；tokens为float32 `(raw-mean)/std`且全history chronology精确；全222 own-context J/ridge与独立replay；四项census完整；选择严格β1且replay不重选；5352完整module attempt/return与模型before/after hash；first error不被source/close/final写失败覆盖。测试必须含密集真实维度array的存储估计，不能再仅靠稀疏fixture承诺体积。

停止规则：一次固定完整计算，工程失败即保留失败并另版本修工程，不修改科学合同；模型无非贪心Pareto/选择不成立则原样发表有限负结果，不调β、不自增候选/半径/seed、不中途加物理筛选。独立确认材料不足，输出`calibration_only_not_admitted`；三重检验任一不满足不开放P1。

本交付实际验证仅为文档静态检查及标准库预算算术；无模型/物理/prepare/新结果。未完成项：222-runner与RAM测试、密集尺寸核验、MAIN代码审读和CUDA、独立authority/direct筛查、完整physical/probe合同、强基线全资源匹配与fresh确认。它们不是本文测试能够代替的证据。
