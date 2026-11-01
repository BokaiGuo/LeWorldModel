# C1到P1：当前目标代理与任务本体的最小判别审阅

2026-10-27；独立保存证据审阅。先完成本文，随后按MAIN新增授权补四分支sampling保存审计函数、8个轻量测试和16.8KB报告（交付路径见末节）。没有模型、CUDA、autograd、模拟或prepare，没有pytest临时目录；测试禁用cache/pyc。只读取源码、JSON/gzip与既有NPZ/PNG作NumPy算术，未改Q、阈值、动作、根或任何sealed字节。本文远小于200KB；报告≤200KB、写前保留200MiB，不授权任何新实验。

## 结论：不应再把三个不同命题合成一个“C1失败”

当前两阶段编译器**符合C1的基本对象定义**：冻结视觉动力学，动作生成中间状态与终态，保存局部response、horizon、成本和动作见证；没有把任意latent waypoint或privileged simulator state当动作程序。它尚不证明P1要求的“直接优化停滞—非贪心重配置改变未来能力—真实收益”。四臂/sampling仅提供这两个已用校准根上的搜索与执行诊断。把C1动作生成、P1 kill experiment和C2能力改变分开，不改变任何准入门。

最短的科学断点不是“还缺一张总体结果表”，而是下面三个可被单独否证的命题：

1. **原RGB距离下降是否代表这个native任务的进展？** 已有实际反例否证普遍排序等价；不是goal文件错绑的证据。
2. **direct真的stall了吗？** B1/B3轨迹不支持该描述；MAIN新B8实际CUDA进一步显示GD持续改进，root1已胜two-stage。GN重复拒绝不等于驻点。
3. **two-stage前缀改变了能力，而不是普通progress/不同搜索覆盖？** 两根所选前缀在模型Q及实际Q上都比zero-prefix更接近目标，尚未建立非贪心性；同根物理能力探针仍缺。

## 1. 本体到底是什么：source、动作、success的精确接点

原提案的仓库书面定义在外层 `docs/SPECTRAL_REACHABILITY_ATLAS_RESEARCH_PLAN_20261023.md:329` 附近：program含动作chunk、预测后继、局部response、horizon/成本与生成见证；reconfiguration必须改善后续真实可控性和闭环结果。其 `:396` 附近明确区分planner latent objective和真实环境success/cost。外层 `docs/D2P_RESEARCH_CLOSURE_AUDIT_20261025.md:417` 起区分C1、C2及P1。这里依据这两份现存提案合同，不假称重新获得了仓库外原始聊天全文。

### Source并非已认证controllability-gap任务族

当前固定source为 `results/p1_conditional_calibration2_main_20261027_v1/sourcepack.json`。generator的 `sample_state` 从native reset的goal复制状态，物体位置加±16px jitter、角加[-π,π]，agent放到物体polygon一条随机边附近；`initial_domain` 仅检查canvas、几何gap[-0.5,3]、物体位置>20px或wrapped角>π/9的非终止条件。有限扫描取first legal，不读取未来；但**没有direct plateau、合法H2可达性或能力改变筛选**。source自身记录threshold authority缺失、global freshness false、原history为causal padding。

重新读两个selected NPZ得到：

|固定root|agent到goal agent距离(px)|object到goal object距离(px)|object wrapped角误差(rad)|
|---|---:|---:|---:|
|0，SHA前缀02a13916|242.044137|9.167074|3.017657|
|1，SHA前缀05844ca2|318.982782|13.691348|1.782513|

这是“近接触起点、较大转角、远处goal agent”的任务，不是“只需把近处T平移几像素”。初态近接触既不证明可完成，也不证明必须先重配置；当前没有物理不可达证书，不能因转角大/十步短就宣布不可达。两个goal从native随机state得到，不是有动作到达它们的构造性见证。

### Primitive：当前没有旧repeat-5限制，但有明确局部支持限制

`p1_conditional_direct_screen_20261027_v1.py:evaluate/projected_logits` 与 `p1_two_stage_witness_20261027_v1.py:PROTOCOL/prefix_slots`：每个5步packet有10个独立time-major坐标，H2共20维。每packet **logit** L2≤`.75*sqrt(5)=1.6770509831248424`，再float32 tanh为raw动作；两包各自球，不是一个full20共同球，也不是全部raw box。root0/1 direct、witness与sampling沿用同一支持，不能借用不同ball的旧P2基线。

实际installed native `env.py:313` 起relative动作是 `target=agent.position+100*raw`，PD积分；10Hz、每public command10个dt=.01内部step。H2十命令是一秒open-loop，不是十次位置瞬移。raw既不能额外除100，也不能解释为绝对像素目标；window clip在该路径被注释。模型history另做 `(raw-mean)/std`，没有把该标准化token直接dispatch。任务中mean=[-.008700000122189522,.006800000090152025]，std=[.20190000534057617,.20020000636577606]。训练normalizer出处仍未认证；本次不由数值合理性代替出处证明。

这些限制是原提案允许声明的一个局部任务实例，不是“原提案全动作域已穷尽”。旧repeat-5 bug不能当当前解释；也不能把局部球约束下的失败外推到一般动作程序。真实RGB可观察性guard进一步限制安全执行；本批8分支均十步完成，所以本批负结果不能归因为guard截短。

### Success：RGB目标正确，不意味着RGB距离就是task success

当前task原Q逐数组等于I192：`mean((z_terminal-z_goal)^2)`。goal是原source真实render的RGB，包含agent与T；不是只有T的几何目标。

native `eval_state` 在 `env.py:347` 使用 `norm(goal[:4]-state[:4])<20` **同时包括agent与object位置**，再要求native angle<π/9；reward为负的完整state7 Euclidean distance。独立object pose `object_xy+20*wrapped_angle`是第三种诊断，既不是native reward，也不是native success。故“去掉goal agent一定更符合官方任务”不成立：agent位置本就在native success中；若改为object-only，是新任务合同，不能静默套回本批。

native角公式未先mod的bug真实存在，state distance也不具角度周期不变性；但本批80步negative-angle与native/wrapped终止分歧均0。不能用这个已知风险解释本批0/8，也不能把+2π表示差叫goal几何错位。

## 2. 三个最小可证伪命题，而非再扩一份缺项清单

### H_goal：固定目标下Q排序忠实于native任务进展——已被保存实际数据反例限制

最小检验不需新Q/新根/新物理：同一root与同一goal比较两个已执行终态。

root0 zero→GN：actual RGB-Q `1.3611217372→1.2453098420`，但native state distance `242.2396808055→304.4694370443`；物体pose完全相同。root0 zero→two-stage：actual Q `1.3611217372→1.2580053533`，native distance `242.2396808055→257.2033997666`。因此不仅object-only评分与Q不等价，**原native连续reward排序也能反转**。无须假设模型预测错误，因为这里用实际RGB编码；它否证“Q单调进展”的普遍假设，不证明agent像素是唯一原因，更不证明某个新的Q可修复。

root1 two-stage actual Q `1.7257656042` 优于zero `2.0772934423`，但物体未改变，native state distance仍从319.3081963126降至288.6989521855：在native任务上agent移动可以算连续距离进展，却仍没有完成转动或success。应保留这个区别，不能将它一概称为目标代理完全错误，也不能称机制成功。

**最小后续判别，待CUDA而非本轮执行：**若要隔离“goal agent敏感性”而非仅记录排序不等价，应先固定同一物体goal、同一候选/真实端点，声明有限goal-agent-only渲染对；重编码后只比较原I192成本/argmin变化，同时保持原native evaluator作为主标签。不得拟合Q、按结果挑渲染点或修改tau。该干预改变了native goal的agent部分，故解释为目标敏感性诊断，不是保持全部官方任务不变的消融。旧其他根的goal干预只证明旧路径，不替本批做该CUDA验证。

### H_stall：最后拒绝一次GN就代表direct无法继续——当前证据不支持

direct实际B1=1/B2=3，zero restart1，step_scale1，damping.002，无line-search。`run_direct` 仅尝试一个固定proposal；拒绝则保留incumbent，未做投影KKT驻点认证。

|root/method|L(B1)-L(B3)|末轮是否接受|末轮导数处gradient L2|
|---|---:|---|---:|
|0 GN|0.134661884|否|0.114468221|
|1 GN|0.100254187|否|0.261431236|
|0 GD|0.079716233|是|0.202862149|
|1 GD|0.090327793|是|0.203840438|

非零gradient本身不排除球约束驻点；这里的关键是**没有**约束驻点证书，且B1到B3仍有实际改善、GD持续接受。未经冻结epsilon不能把这些数字叫plateau。`p1_conditional_task_admission:check_task` 的四合取本来还要求plateau/escape、direct>tau及witness≤tau；当前authority缺失，本轮不发明数值。

root0 two-stage预测1.195592148还弱于GN incumbent1.155789711；该根不能叫逃出了GN局部盆地。root1 two-stage1.720242946胜GN1.968030272/GD1.830458096，只证明有限预算搜索差异。sampling实际两根CEM/IID各60候选：root0 CEM1.17114157、IID1.18210362均胜two-stage；root1两者1.90143550/1.86180073落后。126forward匹配不匹配witness的AD/SVD/solve成本，不是普遍算法优势。

**最新MAIN actual B8，已独立读压缩报告复核：** `results/p1_observed_direct_cuda_20261027_v2_cuda_main01/report.json` 为 `observed_direct_calibration_completed`，errors=[]、model_unchanged/exact_budget_verified=true，SHA `954af696d88eecebdcdc7345de702809517491967263f395a8f2819753d889d5`。逐component report.gz重算文件SHA后读取全部iteration：root0 GD从B4=1.2419287552降至B8=1.2104092115，root1从1.7990429813降至1.7164053655，每轮均接受；root1 B8低于two-stage1.7202429459约0.0038375803。故两根**直接GD未停滞**，不能继续称B3已stall或root1已逃出direct不可改善的盆地。新observed task身份不与旧task_id混为一谈，数值可比依据是同祖先根/原goal/I192/支持及已核对的history数组一致，而非任务hash相同。

两根GN B4-B8改善均0，但还找到具体算法解释：GN第3至8轮的**proposal与完整derivative收据逐字段完全相同**。固定步长、无line-search，拒绝后incumbent不变，再编译同一J/同一proposal；重复拒绝是该更新规则的固定行为，不是约束驻点/真实不可达证书。没有按本批结果修改step/damping；如未来研究其他更新规则，须新source/预算与前瞻协议，不能追认旧seal。

现在待CUDA的不是“补跑B8”（已完成），而是任何新冻结任务族上的独立direct资格与误差验证。缺authority时依然不标eligible；不能按已见B8/two-stage差值选择epsilon/tau。sampling预算与B8新增导数成本仍分账。

### H_reconfig：两阶段赢家的首段是非贪心能力改变——本批未建立，已有反向提示

编译器的选择代码是64候选 **原Q非线性终点最小值**，不是最大rank/最小Vreach直接准入，也没有强制非贪心。root0赢家prefix slot3是root DLS scale1；root1 slot11是谱mode/radius/sign之一。名字或谱方向都不能赋予“重配置”机制。

以paid zero-prefix为参照，两个赢家在step5已经更接近原Q目标：

|root|predicted Q5 zero→two-stage|actual Q5 zero→two-stage|
|---|---|---|
|0|1.3669167201→1.2703217333|1.3611217372→1.3484050007|
|1|2.1410437193→1.8767186442|2.0772934423→1.8310610965|

所以本批所选首段不是“相对zero先牺牲Q-progress再获益”的见证。它们相对真正同支持greedy-prefix的非贪心性仍待按冻结定义核验；本表不将zero冒称greedy，也不把“不劣于zero”当成对所有非贪心机制的否定。

**待物理的最小判别：**若将来检验能力改变，应先固定这些原prefix和same-bank greedy/zero角色，再在各自真实step5状态施加同一预声明完整10坐标suffix探针（固定正负、双尺度、失败全分母；不按响应挑角色/尺度）。先对原native evaluator作响应诊断；如需latent Q能力另行MAIN CUDA编码，费用单列。分别报告common残差（同一目标方向）与own残差，避免把“前缀已接近目标、残差变小”误算为“response能力扩大”。有限差分响应/局部拟合只作测量，不能将拟合tail直接执行或按outcome反选方法。旧geometry能力探针对应另一source族，不填本批槽位；当前8条选中轨迹不是这一干预。

## 3. 已排除、待CUDA、待物理的边界

**本批已有实际证据支持的排除范围：**8attempt reset/render审计中goal wrapped几何差和goal时刻hidden block velocity均0，selected root/state/RGB精确重现；observed-history真实执行zero后仍与旧padding数组相同；本批零点FD方向及history/normalization、whole/segmented保存合同已有独立核验；8物理branch完整10步，24帧实际编码与原Q复算一致。它们分别排除这几条已测路径的特定假负面解释，不是“源码错误已全部排除”。本轮仅重读这些actual证据及有限数组，不谎称重跑其完整审计。

**待CUDA的实质问题：**本批sampling的12张真实checkpoint还没有编码，actual Q必须留空；本批目标图像成分敏感性尚未独立干预。新任务族若要direct停滞/逃逸资格，仍需独立冻结而不是把本批持续改进叫停滞。上述问题不能由sampling模型评分或native几何重现代替；现有真实Q/native反例已经足以阻止无条件的“目标保真”声称，不必等新GPU才承认。

**待物理的关键问题是构造性能力证据：**固定合法动作域和H2下，是否存在完成任务的动作见证，以及改变prefix后相同有限suffix探针是否改善目标方向响应。前者现在未知，不先假定不可达；后者不能从Jacobian秩、prefix几何接近或末态接触推断。sampling物理已实际完成且在下节核查，提供新已冻结赢家的真实后果，不自动变成能力探针或独立root确认。

最小科学推进顺序因此是：保留当前C1工程证据；不宣告P1 stall；不把I192当success certificate；在新预声明协议下分别验证direct停滞与构造性任务能力。当前材料既不支持P1通过，也不支持提案本体的理论负结论。

## 可核查输入与源码定位

全部inner路径相对 `/home/gbk/code/LeWorldModel/le-wm-main`，外层proposal路径已在前文注明。此处记录本次读到的SHA，不新增seal或追认旧结果。

|材料|SHA256|
|---|---|
|`results/p1_conditional_calibration2_main_20261027_v1/sourcepack.json`|`728f14cb83c4fcd0ea4f5348bad2bff5b5dd372ab0185b6e6b3a26dbfda58057`|
|`experiments/spectral_reachability/p1_conditional_procedural_source_20261027_v1.py`|`faddd18ae8d7e0a97c50348929cb4c860321c88a1aedfb755eedc8a0f717deb3`|
|`experiments/spectral_reachability/p1_conditional_direct_screen_20261027_v1.py`|`a05a2bea10ab365c8f2ee2cfcbec6ce4ccc53dc525e8b92b6e96a76529f1c104`|
|`.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py`|`c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2`|
|`results/p1_matched_sampling_cuda_20261027_v1_cuda_main01/report.json`|`cfd13ca478603c90c56b81c70f0ed161ef56d6a8de87da4c0e317388bd138e4e`|
|`results/p1_witness_physical_20261027_v1_main01/report.json`|`48df27e54eca33159c44a834ce0be1a13723553dc9d64dde996b7b4ca9eaf747`|
|`results/p1_witness_saved24_audit_20261027_v1_audit01.json`|`23536377645defbb81bdac3cf16deb33040c677fe3163ab9d70223bda5a94914`|

数值复算路径：`results/p1_direct_cuda_calibration_main_20261027_v2_cuda_main01/root_{0,1}_{full_H2_GN,full_H2_gradient}/report.json` 的 `direct_runs[0]`，同目录 `task_{0,1}.json.task`；witness目录 `p1_two_stage_witness_20261027_v1_cuda_main01/root_{0,1}.json` 的 `winner/prefix_compilations`；物理报告按root/arm join，saved24按root/arm/step join。不以array序作为根身份。

## MAIN追加sampling actual：独立保存审计与一个更窄的代码级代理缺口

实际input `results/p1_sampling_physical_20261027_v1_main01/report.json` SHA `c3eef328bee57ce5d42b25fcf5a9935af2d2282c9de3f8caf46f2b35b53a2ef6`，prepare `results/p1_sampling_physical_20261027_v1_prepared01/prepare.json` SHA `450e620dbe3bf1ec1957c8d2d06fda9dfc05c7cc6bd50453939dc07a2133033c`。

NEW `experiments/spectral_reachability/p1_sampling_physical_saved_audit_20261027_v1.py:audit()` 使用冻结consumer纯verify重建全部60候选/赢家/source绑定，独立检查physical hashchain、原recipe双reset、实际root/goal、selection、40条raw commands/返回state、fresh guard先于dispatch、44份fresh RGB收据、12张PNG文件/像素/guard计数与initial原图。独立复算40步native/wrapped/pose、reward=-native distance；8reset/40step/44render全部attempted=completed。内部reset physics是16个implied，不伪称已instrument；逐publicstep内部physics仍未知。没有加载模型或创建环境。

四分支全十步完成，0/4 success；负native角差0/40、native/wrapped成功分歧0/40。两根CEM/IID的物体末态都与原zero物体pose相同，不从末态推断整轨迹无接触。

|root/arm|object pose cost|native state distance|wrapped state distance|相对原two-stage native差（本臂减two-stage）|
|---|---:|---:|---:|---:|
|0 CEM|69.520206|248.811838|248.808708|-8.391562|
|0 IID|69.520206|202.466196|202.462350|-54.737204|
|1 CEM|49.341614|380.016748|379.994277|+91.317796|
|1 IID|49.341614|525.087763|525.071500|+236.388810|

root0 sampling物体pose比two-stage低5.279507（two-stage改善角但损害位置），root1相同；native排序又不同。这仍不是算法显著优势或能力机制证据。报告还保留相对原zero/GN/GD/two-stage逐根差分、旧搜索资源、12帧路径/hash；actual_Q=null，因为12RGB编码未执行，不能拿prediction或state投影冒充actual latent。

### 新定位：单帧终点RGB代理不能一般等价于含速度的native reward

installed `_get_obs` 将agent velocity写入state[5:7]；`eval_state` 对完整state7作Euclidean距离。其精确分解是

`native_distance² = all_xy_error² + raw_angle_error² + ||agent_velocity-goal_velocity||²`。

本批goal velocity均零。root0 CEM：all_xy误差157.887409，终态速度误差192.271249；三项平方为24928.434028、10.663678、36968.233027，总距离248.811838。相对zero，位置误差242.217669→157.887409改善，却被速度项抵消，使native distance242.239681→248.811838恶化。root1 IID速度误差285.951690，速度平方贡献81768.368744，不能把其525.087763距离全部解释成几何远离。

`_render_frame` 绘制body几何/颜色，未显式编码速度；当前原生单帧encoder给的是图像latent，planner成本只比较终点单帧latent与goal latent，不比较速度。对相同几何、不同速度的状态，native连续reward可不同而单帧目标像素可相同。因此“静态goal图像距离与完整state7 reward排序必等价”在接口上缺少一般充分性；时间history虽供predictor使用，当前评分仍不是一个显式terminal velocity指标。

这不是所有Q失配的唯一因果解释：渲染器细节/模型预测/任务姿态仍有其他因素，未经actual sampling编码也不能说本批CEM的真实Q怎样。尤其native **success不含速度条件**，不能由reward的速度缺口推出native success不可由图像判别，更不能新增速度成功阈值。最小可证伪检查已经是上述saved分解，若解释忽略速度，应被否证；若研究success，继续使用原agent/object位置与wrapped角分列，不擅自替换official。

### 小输出与复核命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p1_sampling_physical_saved_audit_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p1_sampling_physical_saved_audit_20261027_v1 --output results/p1_sampling_physical_saved_audit_20261027_v1_audit02.json
```

第一条实际8 passed（0.13s），纯NumPy边界/身份测试，无tmp fixture；不是神经或物理实验。第二条已实际执行到独占 `audit01.json`，16,807 bytes，SHA `58bc758dcb676f74c86cbe7c199f8be90fffcb28eabf395403a04785c48b37dd`；复跑必须新filename。模块SHA `cfa8b0482fa255d248f602fdc7eed598bd6a2de3fc2dd6a9a01cf798a7872e07`，tests SHA `e773925bdc6478c23b1a446d45694e887d1a3e708ba12be15cb186eb99f4c4db`。上限200,000 bytes，写前保留200MiB，未创建prepare、未删除资产。本交付未创建pytest tmp目录；无可报请删除的本轮mock目录。
