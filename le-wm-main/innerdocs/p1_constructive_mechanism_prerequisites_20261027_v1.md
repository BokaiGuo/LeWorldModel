# Constructive source之后：P1最小机制实验前置审计

## 结论与适用范围

本文件只新增中文前置审计，不是新实验协议、阈值authority、运行授权或P1准入。没有运行模型、
物理、训练或新编码，没有修改任何封存source。采用experiment-design的探索／校准／确认分离原则；
不采用其通用训练模板，也不以新增默认epsilon、统计样本量或预算替代现有科学门。

当前constructive source解除的是“这一新source recipe有合法真实动作可达目标”的前提。
它不解除“冻结模型能表达这条动力学”“直接梯度在足够强的固定预算下停滞”“选择了非贪心动作”
“动作改变真实后续能力”“超过强基线”“独立确认”这六项前提。
**构造witness的成功只能归给数据构造器，不能归给GN、D2P或模型。**

当前最小可继续对象是：两根已知探索任务上，真正不读witness的模型计划及其物理pilot；
随后在另行冻结的机制诊断中比较同根、同支持、同预算的prefix后续能力。
两者均可产生有用正／负证据，但不得借“最小”二字把它们重命名为已完成严格P1。

## 一、依据与当前证据归属

原始提案附件
`/home/gbk/.codex/attachments/861f6764-1b25-4227-bf15-f1a237ba7d8f/pasted-text-1.txt`
§4.2（343–374行）要求短期不降低任务误差、改善下一步任务相关可达能力并带来真实闭环收益；
§8（681–689行）要求在模型已有对应动力学前提下，选择直接梯度易停滞而两阶段可达的状态，
检验超越单层搜索及普通latent subgoal。这是本文所称严格P1，不混同更早SRA文档中同名P1阶段。

本次读取的现存合同：

- [原研究计划](/home/gbk/code/LeWorldModel/docs/SPECTRAL_REACHABILITY_ATLAS_RESEARCH_PLAN_20261023.md)：
  Reconfiguration program的真实能力／闭环验证、冻结模型与CUDA限制。
- [闭环审计](/home/gbk/code/LeWorldModel/docs/D2P_RESEARCH_CLOSURE_AUDIT_20261025.md)：
  C1程序生成与C2能力改变分开；强基线、旧CEM后缀限制及完整资源向量边界。
- [条件根前置诊断](/home/gbk/code/LeWorldModel/docs/p1_conditional_root_preconditions_20261027_v1.md)：
  greedy-relative、zero-relative、initial-relative三种参照不能互换，固定beta选择不能事后改。
- [新source独立saved审计](/home/gbk/code/LeWorldModel/le-wm-main/innerdocs/p1_constructive_reachable_saved_audit_20261027_v1.md)：
  两根真实构造及重放；commit时序、goal_bind瞬间状态的独立证据边界仍保留。
- [blind encode接口](/home/gbk/code/LeWorldModel/le-wm-main/innerdocs/p1_constructive_blind_encode_cuda_20261027_v1.md)、
  [编码saved审计](/home/gbk/code/LeWorldModel/le-wm-main/innerdocs/p1_constructive_blind_encode_saved_audit_20261027_v1.md)、
  [GN/D2P规划接口](/home/gbk/code/LeWorldModel/le-wm-main/innerdocs/p1_constructive_blind_planners_cuda_20261027_v1.md)、
  [物理consumer](/home/gbk/code/LeWorldModel/le-wm-main/innerdocs/p1_constructive_plan_physical_20261027_v1.md)。

本文件核对接口源码，不重新读取privileged sourcepack/certificate供部署，不加载模型；
上述文档的实际编码记录作为其原有证据引用，不在这里重复认证新CUDA结果或未来planner/physics结果。

| 已有对象 | 可支持的结论 | 不能转授的结论 |
| --- | --- | --- |
| 固定witness产生端点goal，独立两次真实重放成功 | 新任务有合法执行见证、不是凭空指定不可达goal | 模型发现了witness；原随机goal可达；D2P有效 |
| 8image native192编码及saved一致性 | 两根blind任务可交接给模型规划器，states3/actions2与goal身份绑定 | 模型预测准确、直接优化停滞、物理控制成功 |
| GN/D2P真实数组接口及mock | 能调用冻结内核、可核算动作与任务合同 | mock线性backend就是冻结模型；代码通过就是机制通过 |
| 未来GN/D2P计划的真实执行 | 对各自预选程序的校准物理评价 | 相同根上的多个臂是多个独立样本；有收益必为能力改变 |

source prepare SHA：`98ffb806018a270f5ef27ffa76be929393a74355c213bc274bf158c72d80b086`；
sourcepack SHA：`9ad3a414fd2148258524d4a7e9d4f070f59532ec2e5943cfa61f9145b3a41bd9`。
blind encode源码审阅版本SHA：`c3b071de1ad0b322e1fff77a716550f42014f0303a6b5fa09271fd4e55609b4a`。
这些是身份，不是机制通过证书；其余封存文件继续由各自原manifest绑定，不复制大protocol。

## 二、先固定因果问题，不能把终点成功倒推为机制

待检验的顺序为：模型在同一root/goal提出prefix并封存选择 → 自身真实执行prefix →
从自身新RGB/history重新估计suffix响应并编译 → 执行自己的suffix → 独立官方评价。
机制测量支路从可重建的同root与同prefix出发，施加预声明共同探针；该支路不是给selector试错的搜索器。

单次D2P赢过GN的终点差可能来自额外搜索覆盖、不同初始化、不同目标、普通progress、模型误差或偶然。
要归因于reconfiguration，必须在同根同时建立：

1. 原P1条件任务前提：模型支持、直接优化困难与合法两阶段可达；
2. 冻结选择器确实选择具有短期非进步性质的prefix，而不是研究者从候选库代选；
3. prefix后**真实**目标相关响应改善，不只是own残差缩小／model J谱变化；
4. 在匹配支持、观察和资源的强对照下有真实任务收益；
5. 预先冻结规则在独立确认根复现。

不能拿根0的非贪心、根1的能力改变与另一批的成功拼成一条机制链。
当前两根可以做机制探索；看过探索结果后的任何新规则都只能进入新协议，不追认原批通过。

## 三、不泄漏的数据与决定边界

部署planner输入只允许获批task export：observed0/5/10 native192、10条真实raw zero对应的
两条normalized between-state token、goal native192、固定Q、模型/normalizer及不透明任务身份。
若后续闭环，新增输入只允许该臂当时新RGB编码与其实际已执行action history。
source_slot只用于外部evaluator join，不能用slot查witness或另一路根状态。

构造器／evaluator可持有state、goal state、reset recipe、certificate和官方评价器；
部署侧禁止取得witness_commands/logits、construction未来RGB/state、replay未来帧、物理probe输出或
由它们计算的候选排名。目标RGB是任务定义，允许；产生该目标的动作序列不是规划器输入。
API白名单只是代码边界，MAIN还须隔离可读路径及调用方参数，不能声称SHA本身提供OS访问控制。

固定source family并不自动盲：研究者已知radial/tangent构造配方。若人为把同配方动作塞进候选、
根据privileged root几何构造初始化，或按source_slot硬编码答案，就失去“从冻结模型发现程序”的归属。
如要比较可部署RGB几何heuristic，必须作为独立标注基线，所有几何估计只用允许观测；
privileged builder witness只能作可达性上界／数据证书，不能与RGB-only算法混排名。

确认期对所有方法同时封存候选库、动作、分数、task/model/Q SHA及选择索引，再交物理executor。
物理失败、guard中断、未开始、模型无合格候选分别保留；不能在查看真实未来后更换prefix、goal、Q、
suffix算法、尺度或种子，也不能让能力探针结果回流成为该根planner proposal。

## 四、非贪心：三个参照并列，不能偷换

在同一root、同一goal和固定identity192 Q下，记初始代价C0、候选prefix的stage代价Si，
own-reach surrogate Vi。当前conditional compiler实际选择
`w = argmin_i (Si + Vi, index_i)`，beta固定1；
同库immediate greedy为`g = argmin_i (Si, index_i)`。候选库及tie-break都在物理前固定。

- **greedy-relative**：比较Sw−Sg及Vg−Vw。只说明相对同支持贪心的取舍；若w=g，不叫非贪心选择。
- **initial-relative**：比较Sw−C0。原提案“短期不降低任务误差”的严格机制解释需在这个参照下核验，
  不能仅因比greedy差就声称暂时牺牲目标进度。
- **zero-relative**：同样5步zero-prefix为coasting控制，有真实时间和动力学；不是C0，也不是greedy。

模型Q的上述数值用于解释选择；真实step5的官方task distance/success与预声明object诊断用于检验其
是否兑现。模型非贪心与物理非贪心分开标记；不把Q恶化但官方任务改善强称真实“牺牲进度”。
native success仍用原生定义，wrapped角诊断并列；不把object-only cost换成新的成功门。

既有合同中的epsilon_stage/epsilon_reach/epsilon_score如需用于判定，只能引用对应任务下已授权、
预先冻结的authority；本文不赋值、不从两根差值反求margin。原始严格不等式的描述性结果也不能
自动升级成有统计裕量的科学通过。若固定beta1没有选到合格prefix，就报告选择机制未建立；
库中有一个事后好候选、调大beta后能选中、或物理oracle选中，都不填这个槽位。

## 五、真实能力改变：共同干预，而不是只看更近的终点

最小同根角色为D2P所选prefix、同库greedy-prefix、zero-prefix。角色在模型输出后、物理前固定；
角色相同时不伪造不同干预，也不为了得到差异而换另一个次优prefix。
每个角色都先从自身原recipe重建root/warmup，再执行自身5步prefix，获得实际边界RGB与完整动作上下文。
诊断可重复重建这些同根分支，但不能将这种重复当新独立样本。

### 共同suffix探针的最低测量合同

在三种真实post-prefix状态上施加**同一冻结**完整5×2 logit坐标探针集合，包含zero和正负坐标方向；
覆盖全部10坐标，尺度／重复数／顺序须在新测量协议中显式封存，不能根据响应强弱换尺度。
所有probe均满足原每包logit球与float32 tanh，没有clip后冒充原动作。
若采用双尺度全坐标设计，每个角色是1个zero+2个尺度×2个符号×10坐标=41个后缀分支；
两根×三角色共246分支。这只是可审阅设计的计数，不是本文批准的运行预算，重复验证另计。
每个完整分支若需重新构造：2reset、10warmup+5prefix+5probe=20public、
2次reset内部各2physics加public每步10physics，即204physics；提前终止／拒绝按实际计账。
故完整246分支上限为492reset、4920public、50184physics，不能把它称几条便宜suffix；
render、encoder、AD、wall、磁盘cap仍须新runner单独完整封存，未给全cap不启动。

每一探针必须有自身实际轨迹／fresh guard／first error；必须服从原生terminal，不把提前成功分支
补成5步。零响应只说明已测有限方向和尺度，旧稀疏null或新非零都不构成全动作球可控性定理。
探针输出只进入独立evaluator；不能用它挑获胜prefix、调beta或执行拟合出的oracle suffix。

### 去除“已经更近”的混淆

首先把每个probe响应减去该角色**自身5步zero suffix的真实终点**，而不是减初始root：
这样分开prefix改变的自然漂移／速度与新增suffix动作响应。有限差分必须使用相同真实后缀时域、
同坐标／动作尺度，保存中心差分、正负非对称及跨尺度一致性，不能把任意rank上升叫任务能力。

能力至少并列报告两种任务方向：

1. **common-residual响应**：所有角色使用root时封存的同一目标方向、同一坐标基准、同一尺度。
   对同一个方向比较真实响应的投影／局部有界拟合残差，隔离“换了J”与“换了残差”的效果。
2. **own-residual可完成性**：各角色实际状态对应自己的目标残差，报告共同probe的真实任务后果。
   它同时包含进度和响应变化，不能单独归因为能力扩大。

physical evaluator可用agent/object位置与局部wrapped角构造task相关响应诊断，但该坐标、权重、
角分支和不连续处理须预先冻结，不在当前两根上拟合；它不是部署readout，不改变原生success。
保留agent项以检查“只把推杆挪近goal”的混淆，并分别列object位移／转角，不用object-only指标
替换官方标签。若用新RGB CUDA编码测同Q latent响应，必须追加真实编码收据并单列成本；
不能凭saved PNG宣称已得到真实latent，也不能在CPU运行neural。

局部线性拟合得到的common残差/V是**测量**，不是可执行reach证书；小奇异值、较高rank或
更低own-V均不够。至少要由预声明的有限真实probe与共同方向响应支持，并由真实own-suffix执行
与完整任务收益交叉验证。若物理响应与模型J排序相反，报告模型能力预测失配，不用调Q掩盖。

## 六、当前接口的确切边界：尚非真实反馈机制runner

blind task的feedback字段是`open_loop_model_only`。当前D2P `compile_arm`在预测prefix上下文
调用fresh `nominal_jacobian`并生成suffix，整条10步动作在真实执行前选好。
当前physical consumer是执行封存own10命令，不会把实际step5 RGB送回该编译器。
因此完成这两个接口的HOST run也只能先叫“模型两段程序的盲计划物理pilot”。

要检验原P1真实边界机制，需要独立NEW driver而非放宽旧task.feedback或改sealed source：

- 在该臂自己的step5重观察；真实history为root时的0/5/10再推进一包后的5/10/15边界，
  suffix只使用更新后的三帧与两包between actions，不能拼接旧construction未来帧。
- actual action tokens必须由该臂10步warmup及自己的5步prefix移位得到；zero raw归一化非zero，
  不能padding或复用另一臂context。goal/Q/model/normalizer/transform保持同一冻结合同。
- 后缀在新RGB编码后重新计算J与有界动作；不能只替换latent、复用root初始J或把imagined suffix
  换名observed。新selection在后5步前durable封存，attempt/completion与失败全分母计账。
- 所有比较臂获得相同feedback时点与剩余动作预算；若要隔离feedback效应，用各自独立预声明
  imagined-versus-actual配对对照，不能只让D2P拥有新观测而GN/CEM执行陈旧tail。

source本身10步warmup不占未来H2，但它是真实成本：每个新物理分支均须计入，不以已有source
或已编码history抵扣reset、warmup、guard和重建。上述新runner未在本文实现或授权运行。

## 七、强基线与公平：GN四步不是整个P1的充分反证

最小pilot可以先比较现有line-search GN与full10 conditional D2P，不能据此称kill experiment完成。
当前接口每根GN为78 predict/action_encoder、4个full-H2 J、768数学VJP方向；D2P为66
predict/action_encoder、11个H1 J、2112数学VJP方向。AD forward已含在predict计数内。
它们同任务但非等计算，也不是相同目标；GN固定4更新不证明plateau或数学驻点。

严格比较仍需遵守原研究计划／闭环审计中的强对照要求：

- 单层完整H2 action-GD与阻尼／line-search GN，多起点及restart、incumbent保留；
- 真正完整执行／按共同feedback重搜的CEM或MPPI、随机search参照；不截断其已搜索tail再改成单次DLS；
- 明确实现身份的gradient-and-restart／GRASP对照，GRASP-style不得冒称外部GRASP复现；
- 无重配置／无组合谱规划，分辨普通谱优化与程序机制；
- 普通latent-subgoal及其相同底层控制器／可执行性合同，不能只摆一个latent waypoint形状检查；
- 原完整闭环审计要求的kinodynamic／learned-subgoal等额外对照在尚缺合法实现或来源包时继续标缺失，
  不用本地mock、privileged witness或任意假decoder填补，也不把精简pilot当成豁免。

所有方法同冻结model/Q/goal、同RGB/action/history、同未来10步与每个5×2 logit球、同feedback日程。
同库greedy是**机制对照**，不能替代全域强优化基线。共享bank／模型查询如复用需列共同成本和
增量成本；不能让greedy承担全部预计算后宣称D2P免费，或把缓存查询忽略为0。

资源比较应封存并报告向量：predict与action_encoder样本、encoder/projector、AD forward、
数学VJP方向、grad/engine API、J/SVD/solve、restart、候选数、重规划次数、peak显存、wall与
真实reset/public/internal physics/render/guard。相同predict不是相同算力，API数也不是FLOP。
若无法同时匹配全部资源，预注册资源上限与多预算性能曲线／预算前沿，结论明确限定所匹配维度，
不声称无条件equal compute。不能从已有两根结果反推恰好有利的B1/B2或某臂停止时刻。

直接优化困难的既有条件合同仍是：冻结B1<B2、多起点最优incumbent、epsilon_plateau/
epsilon_escape/tau与来源authority；平台、逃逸、direct高于tau、合法模型两段低于tau四项分开核验。
真实builder witness不自动满足模型Lwitness≤tau，新的native success不替代model-cost tau。
相反，新探索任务可以不经这项筛查直接做盲模型pilot；其定位只能是探索，不能追认严格P1样本。

## 八、独立确认：新seed不足以自动获得不泄漏结论

当前两根已经用于构造、编码、接口审查和可能的探索计划，只能作为已知calibration任务。
两次witness replay、GN/D2P两臂和多尺度probe都是同根重复，不增加独立root数。
新confirmation应在查看其模型／物理结果前冻结有限source池、全部种子／source lineage、角色分配、
quota、失败处理、排除清单及模型／阈值／选择／预算／统计合同；不得靠扩池补到出现成功。

constructive目标由固定witness端点生成，这种**预先定义的任务分布**允许用于可达任务构造；
但只适用于该条件分布，不能称任意随机goal或广泛场景泛化。所有构造尝试／拒绝与最终可达池都保留，
报告source通过率及条件模型成功率。不能只公布最终成功planner根。
同generator新root可作为该family内根级确认，但必须审查episode／root-goal／任务祖先重复；
跨family／分布泛化需要额外设计，不能仅因seed不同就宣称已完成。

严格P1若用模型预测的plateau+两段达到条件筛选confirmation根，筛查规则必须来自独立calibration
并在确认前封存；结果仅对该**model-conditioned队列**适用，不能报告为无条件成功率。
不能额外要求“D2P已预测胜出／已选非贪心”再把其选中率当独立验证；若研究此类队列须另显式
命名并承担选择偏差，不能填默认P1归属。未来物理outcome始终不得参加准入或调margin。
threshold与effect size／样本量／检验方向／多重比较／缺失数据规则需预注册，本文不发明数字；
两根探索无论差异多大，都不替代所需根级配对统计与独立确认。

## 九、MAIN下一最小可执行接口与停止条件

这是一份前置清单，不自动启动任何作业：

1. **盲计划pilot**：只读已批准task exports，完成GN与D2P各自新真实程序、选择与预算封存，
   不导入witness／privileged source；如已由MAIN完成则审其原receipt，不在此重跑。
2. **盲物理pilot**：按封存own程序同根配对执行，官方success/cost、wrapped诊断、guard及全失败分母；
   只回答模型是否能独立找到可执行程序，不授予机制功劳。
3. **机制校准协议**：先冻结greedy/zero/D2P角色、共同真实probe、共用／own残差、阈值来源和全资源cap，
   再由NEW observed-prefix suffix runner检验短期非进步→能力改变→真实收益，不复用旧imagined J。
4. **严格P1确认协议**：在独立有限池上执行原条件任务门、完整强对照及真实闭环，使用已冻结统计规则。

遇到接口／hash／history／guard错误：工程失败，保留现场，不作方法反例。
模型不支持已知可达任务：模型任务支持未建立，不抹掉物理见证，也不称程序发现成功。
direct仍改善：plateau未建立；两阶段成功只能作探索规划结果。
w=g或无initial-relative非进步：非贪心机制未建立，不从库中替选一个物理更好候选。
仅own-V改善／common真实响应不支持：能力改变未建立；模型J与真实响应反转另报失配。
探针缺失／早停／非线性明显：测量范围不足或估计不稳定，不按最有利尺度给通过。
强基线缺失／资源不匹配：限定pilot比较，不作完整P1或equal-compute结论。
无独立确认：已知任务描述性证据，不声明新根复现或推广。

最终状态：`P1_admission=false`。本文件没有降低、替换或宣告满足任何原P1科学要求；
它把当前可继续的工程／探索动作与仍需独立验证的机制条件分开，不把构造性见证成功转授给模型。
