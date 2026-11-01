# P2 two-program full30：原生物理与RGB验证接线协议

日期：2026-10-29。仅事前工程协议；没有新增执行器、prepare、NN、physics或fullstrict运行。
本文件不修改已冻结full30四文件。以下资源是新物理/编码合同的设计，MAIN审读实现后另行授权。
P1/P2/scientific/physics release均false；known2不是独立新根，不签科学成立。

## 1. 已完成的模型输入权威

路径均相对 `/home/gbk/code/LeWorldModel/le-wm-main`：

|资产|路径|实际SHA256|
|---|---|---|
|prepare|`results/agent_p2_two_program_postcontext_20261029_v1_prepare_MAIN01.json`|`da1803815f14a1e3ee254e7915dc781d66367c7f96b011547c10f0736ba3f9c1`|
|report|`results/agent_p2_two_program_postcontext_20261029_v1_cuda_MAIN01/report.json`|`99f2b5edceae839bc3b0387ac93ef1d866a3fb74ec869ec7a30ceeaf4211ae00`|
|完整consume|`results/agent_p2_two_program_postcontext_20261029_v1_consume_MAIN01.json`|`af2890ae4d330f69e8b01c7a144786ee454b08091e6d79ef63fa1b42e0bf6c5e`|

prepare文件名是拟用MAIN01接点，后续实现必须由CLI实际path+SHA读取验证，不仅相信本文路径；
report和receipt路径/SHA来自MAIN实际完成交付。MAIN报告4sealed、228predict/4J返回、errors=[]。
本轮协议没有重新执行该consume，也不把MAIN的完成声明冒充本文件独立数值认证。
后续adapter显式输入三组path+SHA；验证receipt的report/prepare关联、schema、全部4个plan及源闭包。
可继承MAIN完整验收，但须标记 `full_upstream_rebuilt_here=False`，不能伪造旧full15 receipt。
若选择重新运行模型saved consumer，须另记实际审计成本与边界；worker本轮不做。

固定槽序：root0 transfer、root0 joint、root1 transfer、root1 joint。
arm字面值为 `two_program_transfer` / `joint_two_round`。
不按本轮Q选择根/臂，不因同Q、同动作或同root去重。全部4槽始终保留。
原model_plan的task_id/task_sha256、root_context、root_warm_raw10、goal、normalization、
model_hash、source_programs、logits60、commands、raw_float32_bits、contexts及witness_sha256完整继承。
transfer为两个完整15raw程序，start_controls=[0,15]；joint是一个30raw联合程序，start_controls=[0]。
三packet不是三个program；joint不得改名为两独立程序。

## 2. 新H6/full30原生接线

不能直接向旧raw15 validator、旧episode或旧checkpoints塞raw30；不得patch旧globals。
未来新stem自有prepare/report/consume schema、raw30校验、episode、checkpoint表和全轨迹consumer。
可复用未修改的place、bind_goal、invariant、native_hooks、原生dispatch/metric底层；
调用接口若写死15则只提取policy-free原语，不能伪装旧策略receipt成功。

先从原source_material及其真实bindings恢复原task/reset配方，精确join task/root/goal与blind身份。
不得从latent猜物理state，也不能把模型postcontext写入环境当“实际接缝”。
每槽独立新env，`swm/PushT-v1`、rgb_array、224、relative=True、action_scale=100、
with_target=False、render_action=False；完整封存env/wrapper/package源与版本。
原place两次reset：先root.reset_seed取原goal，再seed=0带原sample_state与goal_state放置。
原warmup_commands完整10raw；逐步核实际action float32 bits、native7、reward、t/tr及goal不变。
warm0/5/10 off RGB与已认证source root历史逐位核对；原task raw10与normalizer恰好一次核对。
原source未保存的中间on参考不能编造；仅作当前off/on/off不改动力学的render审计。
绑定原task goal native7及原goal RGB，必须证明不改snapshot、goal pose、renderer、物理状态。

根通过后原样执行commands[30,2]。只接受有限、原float32精确可表示且rawbox合法的原位模式；
以原logits64→tanh64→cast32与witness认证命令，不再tanh、clip、投影、乘scale或逆atanh修复。
action_scale=100只由原env.step内部应用一次。绝不重新选择程序、重编译或反馈替换B动作。
transfer在control15记录A→B接缝native状态/RGB/action历史；不reset、不重新放置、不重新warm。
joint执行同长度30raw，没有强行人为的program边界，但也保存control15同时间观测。

时间身份必须同时保存：warm0绝对步0/control−10，warm5绝对5/control−5，
warm10绝对10/control0；随后control c对应绝对10+c。
checkpoint固定control[0,5,10,15,20,25,30]，对应绝对[10,15,20,25,30,35,40]。
每warm0..10及plan0..实际终步均新采off/on/off RGB，保存全部三份，控制始终off。
warm10与plan0虽同物理状态仍各记一次，不能为节省成本合并。

## 3. 预算从原代码推导，不按旧full15比例猜测

源接点（均为 `experiments/spectral_reachability/`）：

- `p1_moving_target_overlay_source_20261027_v1.py:place/bind_goal`：每槽两次reset、一次goal_bind。
- `p1_moving_overlay_recompile_physical_20261027_v1.py:native_hooks/capture`：reset渲染与显式render分别计账；每observe为off/on/off三次。
- `agent_allmode_physical_20261028_v1.py:dispatch/observe`：原raw dispatch、不增加step内部render。
- `agent_p2_wallcapped_physical_20261029_v1.py`：原四槽模板，H15/旧schema不能继承为H30。
- 实际installed `.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py`：
  reset调用两次_set_state、一次_set_goal_state、一次render；_set_state各space.step一次；
  step循环int(1/(dt*control_hz))，dt=.01/control_hz=10，故10次physics/public；
  step返回obs/info不调用render。构造器无reset/space.step调用；实现须将原生类源hash纳入闭包。

|资源|每槽满额|四槽满额|推导|
|---|---:|---:|---|
|constructor / close|1 / 1|4 / 4|独立环境，finally close|
|reset|2|8|seed+placement|
|warm / plan public|10 / 30|40 / 120|固定原warm、新H6|
|public合计|40|160|不是仅120plan|
|physics step|404|1616|2×reset +10×public|
|set_state|4|16|2×reset|
|goal_setter|3|12|reset+goal_bind|
|goal_bind|1|4|原task绑定|
|reset goal_render|2|8|1×reset|
|显式render / PNG|126|504|3×[(warm10+1)+(plan30+1)]|
|全部native render|128|512|显式+reset；两项不能漏记或重复收费|

public与physics是嵌套轴，不能相加声称独立计算量。构造/失败intent与returned分别记录，不退款。
全部根已warm并绑goal时，若实际计划返回S≤120步：public=40+S、physics=416+10S、
explicit_render=144+3S。warm或初始化失败时不能套这个简式，必须按逐事件census。
每个观察点三张PNG，但只一个控制时刻，不是三个独立样本/根。

新物理阶段模型encode/project/predict/J/VJP/backward/solve/SVD/train全部0；
输入与SHA审计、CPU序列化/PNG/IO另计，不能声称整个工作零成本。
建议新物理合同wall900s（verify前到最终保存后）、floor200MiB、final reserve8MiB、
output cap256MiB；这是待MAIN批准的工程上界，不是已测成本或旧64MiB的自动继承。
仅RGB裸像素已504×224×224×3=75,866,112bytes，故旧64MiB不能作全图无条件保证。
建议逐PNG≤256KiB（总126MiB）、journal≤64MiB、其他metadata≤16MiB，reserve8MiB，
总214MiB≤256MiB；超限保留partial，不缩图/删帧/扩预算继续。此为强制拒绝边界，不保证PNG必可装入。
压缩前后字节、disk floor、实际wall/CPU/IO分别保存；增量计账避免每event遍历全部目录。

## 4. 原生指标、早停和严格输出

每步保存原native7（agentXY、objectXY、angle、agent velocityXY）、原goal7、reward、t/tr、
requested/actual rawbits、state/snapshot和goal不变证据。独立重算jointXY、agentXY、objectXY、
wrappedangle、velocity、native7距离与reward=-native7距离。
原native成功规则保持：jointXY<20且min(abs(angle差),2π−abs(angle差))<π/9；
wrappedangle仅诊断，不能替换原native角项或静默修正其极端角度行为。

原生t/tr立即停止；不为了凑30继续、不padding、不以成功终态补后续cp。
4rows全部保留，缺export、失败、missing checkpoint均独立标null+原因，不能替根/改臂。
每根两臂比较只取common control0..min(actual_steps)，事件终态单列eventtime。
不能拿早停14的终态与另一臂30终态声称same-time提升；失败不算科学反例也不能删掉。

新receipt至少输出physical_report_sha256、prepare_sha256、plans[4]（完整原commands）、rows[4]、
逐步trajectory、CP引用/缺项、全部计量intent/return/pending、工程complete与科学false分开。
后续encoder只读取该新schema的真实RGB引用，不套旧full15 physical receipt。
全journal事件已知ABI逐条校验、原生digest/hashchain、全部NPY/PNG文件和pixelSHA、source前后hash、
闭包inventory/unknown event拒绝、原序列化LF格式实测、真实Store.emit/put→磁盘reader回归均须实现。
只RAM codec测试不足；失败final路径可写partial但不得绕预算签complete。

## 5. 后续RGB：独立新授权与费用，不预填未来SHA

物理prepare/report/完整receipt实际path+SHA目前全部null；须MAIN完成后提供。
新encoder同时绑定模型三权威和物理三权威，逐槽逐动作逐时间join，禁止仅凭terminal值相同接缝。
建议固定4槽×7cp=28encoder+28projector：4个root anchor与24个H6预测端点，0新predict/J/physics。
这是完整无早停时上限，缺cp保留null不编码、不以邻帧替换；保留28槽分母。
实际control0历史为warm0/5/10；control5为warm5/warm10/plan5；control10为root0/plan5/plan10；
其后滚动三真实cp。封存三帧身份、preprocessing、CLS、projector、归一化native192，normalizer只一次。
对各control5..30原predicted context末latent与实际编码192独立算MSE、Q_actual/Q_pred及差，
identityQ/goal/model源/weights保持。anchor核原root encoding，不能拿本次anchor重拟合模型或Q。
逐臂编码不默认为免费共享；如将来采用已认证同图复用，须另声明精确bits/证据与新增计数。
RGB bridge有独立HostCUDA/显存/wall/输出caps及模型before/after守卫，本文不授权其运行。

## 6. 对照解释与下一实现验收

full30与旧full15时距不同不可混表成同任务效果。joint更大外包球、更多模型计算，
不是纯“是否组合”的公平因果、等FLOPs或同支持对照；两根四槽不是四独立根。
原228predict/4J是已付继承模型费用；新physical/render与未来RGB分列，FLOPs未知不填0。
物理结果不反馈选择本轮root、程序、参数、cp、预算或成功门。

下一步仅在MAIN批准后实现独占新physical source/tests/doc及其真实prepare/verify/run/consume CLI。
当前没有可执行H30物理CLI；旧full15命令不能声称可直接运行本协议。
验收至少含raw30位级/任务join、4槽完整分母、control15不reset、早停无padding、
满额1616/504与部分预算census、真实磁盘journal、源/数组/PNG/计账篡改拒绝及timeout保全。
本协议不自动授权physics、NN、fullstrict，也不修改任何已封存模型结果或科学gate。
