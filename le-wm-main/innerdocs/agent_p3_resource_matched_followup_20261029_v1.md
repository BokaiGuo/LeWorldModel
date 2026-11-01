# P3 有效8臂之后：资源匹配的事前实施设计 v1

范围：仅设计交付，2026-10-29。只读冻结源码，不读取实验outcome、不选择较优根或bank、不运行prepare/consume/CUDA/物理。本文件不改变任何现有合同，也不是运行授权。使用experiment-design技能组织可分离干预；所有数值预算由现有调用结构推导，不由结果调参。

## 1. 锚点与当前8臂能说明什么

源码均在`experiments/spectral_reachability/`：

|文件|本次读取SHA256|
|---|---|
|agent_p3_effective_schedule_20261029_v1.py|7cef2600e9844d981f97fdd6c385bff0c3ae9bdf55713ac8c92cebd28b003c23|
|agent_p3_fullcommit_live_20261028_v1.py|b3e1c321883f47e9030777312989d2a318c699dfa207f53c11006294059e6188|
|agent_p3_local_signal_cuda_20261028_v1.py|6ce7722eca40cdd5f8958ba9590d8dc3d76d38bd10895d5ca494c1db3b6d100b|

可直接定位：effective_schedule的`contract/costs/caps/plan_node/Session/episode`；fullcommit_live的`predict_axes/costs`；local_signal的`Backend.compile_finite_horizon_J`。原8臂是known roots0/1 × commit15/feedback5 × endpoints/midpoints。实际总执行20、原始公共prefix5、后续3个五步包；不是跨根配额路由。两个bank分别为[0,1/3,2/3,1]及[1/8,3/8,5/8,7/8]，均4候选。

commit15在b5生成H3并承诺15步；feedback5在b5/10/15各生成H1执行5步。因此跨schedule比较同时改变计划horizon、重编译频率、候选完整动作空间；跨bank又改变候选支持。不能将组合变化归因于单一“反馈”或原C3全部机制。

## 2. predict51的逐轴解释

下表是**每episode完整到20的合同计数**，不是读取实际回执后的统计。提前终止、失败、attempt未return都必须另列，不把cap记为actual。

|资源轴|commit15|feedback5|
|---|---:|---:|
|同节点H1/H3信号predict|24|24|
|候选whole+segmented+winner predict|27|27|
|inclusive predict API / samples（batch1）|51 / 51|51 / 51|
|AE / predictor / pred_proj调用|各51|各51|
|predict上下文token / pred_proj行|153 / 153|153 / 153|
|encoder / projector（b5/10/15/20）|4 / 4|4 / 4|
|J / signal compile|6 / 6|6 / 6|
|VJP / grad / raw_backward|各1152|各1152|
|program compile / ridge solve|1 / 1|3 / 3|
|array SVD（含6个信号诊断）|7|9|
|完整trajectory replay API|9个H3|27个H1|
|候选metric / schedule decision|4 / 1|12 / 3|
|native public目标（含warmup10）|30|30|

推导：每节点信号nominal H1+H3=4predict，独立AD primal再4predict；三节点24。每个4候选程序需2×4+1=9次完整轨迹重放，H3一次27predict，H1三次也是27。J分别192×10及192×30，每个192方向；每节点J元素7680、384VJP，三节点23040元素、1152VJP。三节点nominal12predict与AD primal12predict分别收费，不能把AD forward算免费。

必须纠正“51相同但encoder/J不同”的笼统表述：这份冻结8臂**完整路径的encoder/J计数本来相同**，因为commit也在b10/15编码并支付不用来改动作的H1/H3诊断。实际早停可使到达节点及encoder/J不同；自然省成本commit若省掉这些工作也会不同，但它不是当前共享诊断实现。二者不能混报。J计数相同也不能推出各H反传成本相同；长H图含更多原生模块及中间buffer。

全8完整目标为408predict、32encoder、32projector、48J、9216VJP。source/model哈希、precision守卫、GPU allocated/reserved、PNG/preprocess/native buffers、每个attempt/return、AD图长度、SVD/solve、候选支持和动作float32位模式仍需独立保留。嵌套模块墙时不能相加冒充总墙时。没有算子级测量时FLOPs=unknown，equal_FLOPs=false；equal_wall=false。不能用51或VJP方向数估算真实等FLOPs。

## 3. 下一项主对照：同反馈、双horizon完整面板

建议新runner先实现下述**受控shared-pipeline消融**，而非宣称最省成本fixed基线：每个真实b5/10/15，两个策略都取得相同RGB观测、真实last10动作、paid H1/H3 J，两个horizon各生成同一固定4候选bank，全部whole/segmented重放并独立winner replay。两个策略都完整支付两个面板；每次仅执行所选面板赢家前5动作。b20编码终态，不再规划。H3始终仅lookahead，不执行其余10，不作为三个child信号。

四臂固定为fixedH1_endpoints、adaptive_endpoints、fixedH1_midpoints、adaptive_midpoints；全2known roots共8episodes，不因bank结果选择只运行其中之一。fixedH1始终用短面板；adaptive复用原冻结local_signal `runtime_signal`及`core.choose`原决策（chi=.5/gain=.05及既有支持范围），不新设科学阈值。原`decide_root`仅根节点，不能改标成child；新runner需按真实当前节点调用原policy/choose并绑定新paid信号。非progress或未支持决策整节点failclosed，不伪造reconfiguration/composition。每面板稳定(cost,index)；所有面板成本/拒绝/碰撞都保存，即使未选。

这个对照固定反馈频率、实际执行长度、每个horizon候选支持及计算队列；改变的是固定/原信号决策是否使用同一已付面板。各arm实际未来状态当然可分叉，不能声称它们后续J数值相同。每节点44predict=8信号+9×1+9×3；每episode132predict，其中nominal12、AD primal12、程序108。每episode6J/1152VJP、6program compile/solve、12array SVD、54完整replay API、24候选metric、3policy decisions、4encoder/4projector。全8上限1056predict、48J/9216VJP、32encoder/32projector；batch1下samples=predict API，predict上下文token及pred_proj行每episode396。所有这些是完整路径精确预算，不是填充。

两个面板都参与真实可选程序库，fixed策略未选长面板则是明确的共享诊断开销；不能把这种付费方式叫自然fixedH1最强成本基线。后续可另报省略未用面板的fixedH1（51predict），但不称与132匹配。若要评价“省计算”就不能用这份shared-panel设计代替自然实现。

H3在b15仍预测至task时刻30，实际只执行至20：这必须在新合同明确允许为terminal-goal lookahead，不借旧fullcommit的remaining-fit规则偷偷截断/拒绝。两个策略均付相同面板，但短长面板的预测终点不同，故不宣称同预测终点的价值比较；物理endpoint评价仍固定20。若研究问题要求所有预测也终止于20，应另冻结H3/H2/H1倒计时协议及真实H2 J适配，不能将现有H1/H3面板冒充它。本建议优先已有可执行H1/H3 ABI，不默认H2实现。

## 4. 可分离的补充对照（不能与主对照混并）

|问题|固定项|改变项|完整每episode预算与限制|
|候选支持|feedback5、H1、原规则4候选、所有资源|endpoints vs midpoints|51predict/6J/4encode；现有8臂中feedback5两bank已提供工程入口，不重复生产|
|计划horizon|feedback5、同bank4候选、每次只执行5|固定H1 vs 固定H3|H1=51，H3=105predict；各6J/4encode，非等predict，更非等FLOPs|
|动作更新频率|H3、同bank4候选、warmup/prefix/20步评价|b5承诺15 vs 每5步重编译并执行5|原commit=51，receding H3=105predict；不可称等资源|

105=24信号+3×27程序。若需动作更新频率的**资源匹配受控版本**，commit臂也在b10/15按真实新观测生成完整H3 shadow面板，作为明确诊断而不替换承诺；receding臂使用它。二者均105predict/6J/9SVD/3solve/27完整replay/12metric/4encode。shadow未参与动作控制，必须单列，不叫有用搜索或自然成本；不能用重复零动作/replay空包padding。晚节点H3同样须新合同明确lookahead允许越过执行终点。该对照固定H而改变使用新反馈更新动作的频率，不能再同时混换bank。

预算轴无法全匹配时直接报告差异；不以减少候选、不同bank、早停后补调用、降低精度、缓存旧J、改变ridge阈值或缩短horizon凑51。fullraw候选可能因方向零/量化重合；保留全部固定索引和碰撞，不删赢家或重采样。

## 5. 最小实施接口与事前资源门

现有frozen runner不接受双面板/固定H3反馈新schema，不能给它换arm名字后声称已运行。本文件没有提供不存在的生产CLI。下一独占runner需复用：effective_schedule `generate`及原生ledger/backend replay；Session真实观测/packet ACK；episode原RGB与raw10接线；原strict对PNG预处理、native journal、precision/source/model前后哈希、全量事件census。新增仅为双面板节点schema与“选择赢家首包”session；两面板的请求、4候选、完整path、selected replay以及丢弃的lookahead动作都持久化，严格消费者独立重建两面板与原policy。不可monkeypatch旧module globals，不把旧receipt重标新schema。

新prepare必须绑定原task/goal/Q/normalizer/模型与所有安装类源码，封存4arm×2root×3node×2panel×4candidate完整分母，未知字段拒绝而非0；实际末帧缺失保留null。request白名单只含RGB产生context/真实动作history/原目标与模型信号，不把reward/state/evaluator距离作为选择输入。native state仅strict与安全检查可见。

建议主对照整job事前上限：wall7200秒、输出2GiB（含256MiB终态reserve）、磁盘floor200MiB、CUDA allocated3GiB/reserved4GiB；这是新设计建议cap，不是已测内存或时间承诺。物理安全cap沿原8episode：constructor8/reset24/set_state48/goal_render24/goal_setter32/goal_bind8/public240/physics2448/explicit_render256/close8；encode/projector32各。实际warmup/physics/构造渲染逐event记录，cap不是实际账。1056predict及上述逐轴资源在dispatch前检查；超cap、异常或原生提前停均保留partial与未运行槽，不重试、不退费、不补时长。新增runner测试需验证attempt失败不refund、unknown事件拒绝、同根身份、两面板候选/rawbit/独立winner、late-H3语义、早停null、固定政策与原信号政策各分母。

## 6. 研究边界与交付状态

仅known2工程对照；不打开reserved roots、不自签freshness。原P1准入没有因这里的接口或测试被满足；独立fresh root证书、完整原任务成功定义（含agent+object）与MAIN当前科学准入要求仍独立。原C3还含program type/depth/reconfiguration/composition，此progress-only双horizon消融不覆盖它们。没有科学通过、物理收益或等FLOPs声明。

本次仅新文档；未运行测试（无新代码）、未执行任何生产或读取结果选参数。后续先由MAIN决定主对照/是否允许late-H3 lookahead，再实施新runner和RAM负向测试，MAIN审读冻结后才prepare/hostCUDA/native/strict。所有门限与候选支持需在任何新结果前固定；不改现有8臂源码及结果。
