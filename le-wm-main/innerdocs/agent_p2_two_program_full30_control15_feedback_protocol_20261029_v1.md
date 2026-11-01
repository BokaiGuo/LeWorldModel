# full30 control15 反馈后缀重编译：事前协议草案

## 1. 状态、输入与固定分母

本文件是待MAIN批准的新探索性协议，不授权实现或生产。旧full30 planner/native/RGB源测文保持冻结。
known根固定0、1，各保留two_program_transfer与joint_two_round，四个base slots一律保留，
不依据本轮Q、物理误差、成功或RGB挑根/臂/候选。fresh=0；P1/P2mechanism/scientific/physics release均false。
这是full30/H6诊断，不与旧full15合并，不签能力、成功率或公平因果通过。

实际已存在的模型权威（以下path相对 `/home/gbk/code/LeWorldModel/le-wm-main`）：

- prepare `results/agent_p2_two_program_postcontext_20261029_v1_prepare_MAIN01.json`，
  SHA `da1803815f14a1e3ee254e7915dc781d66367c7f96b011547c10f0736ba3f9c1`。
- report `results/agent_p2_two_program_postcontext_20261029_v1_cuda_MAIN01/report.json`，
  SHA `99f2b5edceae839bc3b0387ac93ef1d866a3fb74ec869ec7a30ceeaf4211ae00`。
- receipt `results/agent_p2_two_program_postcontext_20261029_v1_consume_MAIN01.json`，
  SHA `af2890ae4d330f69e8b01c7a144786ee454b08091e6d79ef63fa1b42e0bf6c5e`。
- native report `results/agent_p2_two_program_full30_physical_20261029_v1_MAIN01/report.json`，
  SHA `39dae04ac59d9fd1b1e3e65dfe675a4627c13ada00d8c065e655053018b35c90`。
- native receipt `results/agent_p2_two_program_full30_physical_20261029_v1_consume_MAIN01.json`，
  SHA `aabc5cce3855021eba0fe5751c0ae87809cfefb4c2ed54b0d2899048fd6b3cd8`。

RGB生产正在MAIN执行，本协议不猜其prepare/report/完整receipt SHA。
未来输入必须显式给三者真实path+SHA，校验本轮28槽RGB自有schema、四plan顺序、全部bindings及实际权重源。
继承MAIN完整receipt时标记full_upstream_rebuilt_here=False，不假称重新执行上游fullstrict或免费认证任意artifact。
原有main同时间标量分析与当前28槽encode不重跑、不修改。

## 2. context、冻结动作与三策略

每个base slot建立三个预定策略，共12策略槽，按root→base arm→policy固定排列：

|policy|control1..15|control16..30|编译输入|
|---|---|---|---|
|O，原openloop|原literal prefix15|原literal suffix15|不新增编译，原结果明确继承|
|I，imagined重编译|同一prefix15|固定算法选出的suffix15|原planner contexts[2]|
|F，observed重编译|同一prefix15|同算法选出的suffix15|RGB receipt对应链step15.context|

O是原A15→B15或原joint两轮后的完整raw30，不能把joint改称独立A/B技能；
I/F均不改prefix、不换root/task/goal/identityQ/checkpoint/normalizer/action scales。
I与F只改变编译输入context；两者候选公式相同，但梯度与产生的候选当然可能不同。
O对F包含额外优化与信息两个因素，不能独归因反馈；I对F才是固定优化过程下的信息输入对照，
仍仅known-root有界诊断。transfer与joint之间支持和原付费不同，绝不称纯模块化因果或等FLOPs。

F context直接取actual_context_chains中四分支各自cp15完整对象：
latent_history为实际cp5、cp10、cp15的192维向量；action_history为对应cp5→10、cp10→15
两段raw5经原normalizer恰好一次生成的2×10 token。绝对native时间为15/20/25，
control时间为5/10/15；不得误用warm绝对0/5/10或control0/5/10。
严格验证root/task、plan_index/arm、全部prefix15 rawbits、三帧来源与normalizer；只last latent不合格。
I使用完整imagined context而非拼入真实latent的混合体。原cp20..30实际图/状态/标签禁止进入生成与选择。

缺cp5/10/15、源损坏、提前停止或历史不符：原槽blocked/null并保留四base/12策略分母；
不得补帧、padding、借另一臂相同向量、换root或使用cp20反推。cp15前原生成功视为已停止，不能继续动作。

## 3. 预定单轮H3后缀优化、域与guard

这是新算法合同，待MAIN批准后才实施，不冒充原H6两轮joint算法或旧R bank。
每个I/F槽固定一轮，不按结果加迭代。u0取该base最终full30 logit60的后30个float64原值，
其raw必须逐位等于原commands[15:30]；不逆atanh、不重建近似logit、不修复iterate0。
transfer保留原B独立球R=.75sqrt(15)。joint冻结其原前30维logit p，后缀有效域为
`||v||²+||p||² <= (.75sqrt(30))²`，即剩余球R；不放宽成完整joint球，也不缩为transfer球。
先认证原float64合法witness与原tanh64→cast32 rawbits，不拿postcast逆变换范数拒绝历史见证。
剩余平方值负数、非有限或原witness无法认证均blocked，不取abs/epsilon强行修正。

从相同goal g、所选输入context C与literal u0新付费计算三packet nominal z及raw-leaf
`J:192×30 float32`；保存3个完整AD primal context并与nominal逐位匹配。
连续chart约定（不声称float32 cast可微）：

`grad = (2/192) [J diag(1-tanh(u0)^2)]^T (z-g)`，float64核算。

若||grad||=0，target=u0；否则target=−0.5 R grad/||grad||。
d=target−u0；若grad^T d>=0则d=0。固定alpha=(1,.5,.25,.125)，
候选为u0和u0+alpha*d，共5个，不去重、零方向仍完整付费。全部float64 logit合法且rawbox合法才执行，
原chart `numpy.tanh(float64).astype(float32)`、原一次normalize，无clip或后验半径调参。
保留R=0退化情况下的全部重复候选及费用。

以各候选实际非线性末节点mean((z−g)^2)稳定选择(Q,index)，tie保index0，guard只保证该context
下模型Q不比其新付费incumbent差。比较基准不是旧imagined Q；不承诺真实Q、XY、angle或success改善。
选择后独立selected replay，再分别whole15与按5+10分段replay，所有3context逐位相等。
模型参数冻结，不训练，不使用未来native outcome/成功标签调学习率、阈值或候选。

## 4. 新增费用、继承费用与资源闸

每个I/F槽固定：incumbent3 + AD primal3 + 四候选12 + selected3 + whole3 + segmented(1+2)=3，
共27 predict API、27 AE/predictor/pred_proj、1个192×30 J。
I/F各四槽：108predict/4J；合216predict/8J，1536 requested VJP、46080 J元素、24 AD primal packet
（已含于216predict，不能再加一次）。新增SVD/solve/encoder/projector/physics/train均0。
engine backward与FLOPs未测为unknown，requested VJP不是engine backward。
I/F对各自新context的incumbent三predict同时是O suffix在该context的全额新评分，不重复收费亦不称旧预测复用。
O本身无新planner调用，但继承原228predict/4J、native与28encode费用单列，不从新增账抵扣。

未来runner建议沿full30模型阶段上限：whole1800s/slot450s、输出128MiB/reserve2MiB、floor200MiB、
CUDA allocated4GiB/reserved6GiB、HOSTCUDA/TF32 off/autocast off/MATH；这不是本协议执行授权。
必须先以完整216IO/8J/事件与元数据做RAM shape预算证明；超过须报告MAIN，不得删候选、降精度或扩cap。
verify前开始计wall，贯穿模型加载/原生IO/优化/落盘/fsync及完成hash；attempt/return独立，失败不退款不重试。

## 5. 真实执行与同时间指标（后续另批）

本轮只模型协议；未来物理必须自有H30反馈schema，不借旧H15 outer validator。
每个I/F执行独立reset/place/warm10→冻结prefix15，首suffix intent前持久化全部history/action/native guards。
采用已存F context是saved-feedback replay：新执行prefix到cp15的native7及相应原始RGB必须与原权威
逐位一致才允许承接保存context。不能把保存状态直接注入env冒充连续反馈，不能在control15 reset。
若不同则blocked，不能偷偷新编码或容差放行；在线新编码是另一个需事先批准与计费的实验。
O可引用原same-action完整轨迹，明确历史重放而非新随机复现；所有12策略槽保留，包括重复动作，不按outcome复用。
未来物理及编码资源另按实际native代码推导/审批，不把当前0新增physics写成完整实验免费。

预定报告每base O/I/F同control时间0/5/10/15/20/25/30的jointXY、objectXY、wrapped angle、
native7（含速度）、原native任务success及first-success/control-stop；cp20/25/30的真实编码Q需未来新增encoder预算，
不得用模型预测Q冒充。任务判定沿原native成功标准，不新设threshold。
所有活跃分支目标长度30；早停保event-time终态，缺cp为null，不把早停14/29终态当cp15/30或padding。
两分支比较只用真实共同同时间集合，并列完整固定分母、缺失原因和各自终态event-time，
不把不同长度末状态差当同时间增益；success仍按全部固定槽报告，不只统计有末帧的子集。
每root/arm分别列，两个known根不是四或十二个独立根。跨arm不称配对同前缀；仅同base内O/I/F同prefix。

## 6. 可执行接点与封存次序

未来prepare输入：模型三权威、native三权威、RGB三权威的path+SHA，及本协议与新源测文pins。
先构造四base的task/raw/logit/domain/context身份表，再生成固定8个I/F jobs；无候选先看label接口。
标签与cp20..30真实结果不进入模型job可见数据；全部候选/selection/replay封存后再分析。
输出每job包含base/root/arm/policy、parent full30 SHA、prefix bits、C/input SHA、u0/domain witness、
所有candidate commands/contexts/Q、J/全IO/计账、selected suffix与完整raw30的新witness。
只有原prefix15与新后缀相接，不伪造原planner receipt；新plan与旧预测字段分名并保存parent出处。
MAIN批准算法及预算→新kernel/runner/tests/doc与RAM预算冻结→MAIN planner HOSTCUDA/strict→
再审物理handoff与新预算→MAIN物理/strict→另批终态编码及纯saved比较。当前止于协议，不执行任何阶段。
