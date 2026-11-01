# full30 control15 O/I/F：自然重放物理部署协议（待MAIN批准）

## 1. 范围与实际/未来权威

只新增本文，不实现、不运行prepare、NN、physics或fullstrict，不改已冻结v1/v2模型源测文。
此协议部署冻结模型编译输出，不在物理阶段重新选择候选、策略、根、alpha或阈值。
known根固定0/1，base顺序root0 transfer/joint、root1 transfer/joint。
每base固定O/I/F，总12策略槽；O的4条轨迹继承原完整full30物理，I/F共8条新自然重放。
两个known根不是8或12个独立根；P1/P2mechanism/C3/scientific/physical release均false。

原O权威（路径相对 `/home/gbk/code/LeWorldModel/le-wm-main`）：

|资产|实际path|SHA256|
|---|---|---|
|native prepare|results/agent_p2_two_program_full30_physical_20261029_v1_prepare_MAIN01.json|04d4a887e09381cc9869d76b259af41159d6678269c48e5a3cf0caa42df66100|
|native report|results/agent_p2_two_program_full30_physical_20261029_v1_MAIN01/report.json|39dae04ac59d9fd1b1e3e65dfe675a4627c13ada00d8c065e655053018b35c90|
|native完整receipt|results/agent_p2_two_program_full30_physical_20261029_v1_consume_MAIN01.json|aabc5cce3855021eba0fe5751c0ae87809cfefb4c2ed54b0d2899048fd6b3cd8|
|RGB prepare|results/agent_p2_two_program_full30_rgb_encode_20261029_v1_prepare_MAIN01.json|20a248dc940188ff3858fe71769cdc2d726ec23d2e77e8e50783ff777db63527|
|RGB report|results/agent_p2_two_program_full30_rgb_encode_20261029_v1_cuda_MAIN01/report.json|915b6760db8bf30abe16e4bfba347604e6359c17d08c1ff3f2a7a076b7714df4|
|RGB完整receipt|results/agent_p2_two_program_full30_rgb_encode_20261029_v1_consume_MAIN01.json|42164a225310c39ff83e398f55501e36e8db9876d5b52dbc697d031e31468166|

新模型四权威现已由MAIN实际完成；以下均为实际文件，前缀同上：

|资产|实际path|SHA256|
|---|---|---|
|feedback v2 prepare|results/agent_p2_full30_control15_feedback_20261029_v2_prepare_MAIN01.json|498b561855d7859681a18ed47e0157908a146a796145b929b5436a1e73b79630|
|feedback v2 report|results/agent_p2_full30_control15_feedback_20261029_v2_cuda_MAIN01/report.json|6e2c82633140e071c880aed92c3cde025635cf7dd92c6687d9e0334bc30db3a8|
|feedback v2 completion|results/agent_p2_full30_control15_feedback_20261029_v2_cuda_MAIN01/completion.json|8622879f9c674d6acffcbc6646fbaccf25023486b0534626c3b63a5bab1d7329|
|feedback v2完整receipt|results/agent_p2_full30_control15_feedback_20261029_v2_consume_MAIN01.json|4e7d5e1f6b4b03b28b68eb8a31016211f7d1e0ff3115671923b6df2e97f3d893|

MAIN认证全216predict/8J返回与完整consume；本次worker仅hash四文件并读report元信息，不重做fullstrict。
不能拿v1失败prepare或拟用名填充权威；即使6个job selected0，也不省掉任何固定I/F轨迹或改变分母。
新source已冻结SHA `2c01821b10a21dd9f9fad43234b71fb76a4ef3169d7993786ab0f006a4983362`；
v1准备失败是set序列化代码错误，不是模型/理论结果，不作O/I/F效果证据。
未来prepare仅继承MAIN完成的完整receipt，明确full_upstream_rebuilt_here=False，重新hash全部来源闭包，
不回调上游fullstrict、不伪造旧H15/H30 receipt。新物理需要自有schema和MAIN显式工程授权。

## 2. 固定动作、reset与两道durable guard

从模型v2完整receipt逐base/policy精确join其8个new model_plan及12 policy_slots；
task/root/parent_witness/raw30/goal/model/normalizer与原O逐项一致。
核新commands前15raw逐位等于原O，后15raw逐位等于各自已封存selected suffix。
不能根据new_suffix_Q选I或F，不把同动作/同向量合并，不根据原成功或原误差跳base。
所有缺export、缺source、失败及未执行保留对应槽null+原因；不以O替补I/F。

每条新轨迹独立constructor→原两次reset/place→warm10→绑定原任务→prefix15→新suffix15→close。
相同原 `swm/PushT-v1`、224 RGB、relative=True、action_scale=100、with_target=False、render_action=False。
warm命令和parent任务placement均来自原权威；不能从latent还原state或重新找接近状态。
raw只按literal float32下发，env内部原scale恰好一次；不再tanh/normalize/clip或修复动作。

第一道guard复用原严格root资格：逐warm native7/snapshot/raw/goal/t/tr/reward、原warm RGB历史、
原root hash及goalRGB/pose/renderer恒定，durable native_history_guard必须早于plan control1 intent。
warm0/5/10绝对0/5/10，对应control−10/−5/0；所有warm中间渲染同样保存核对。

第二道新增feedback_prefix_guard必须晚于control15完整return/三图落盘，早于control16任何action intent：

- 对照同base原O，逐control0..15及warm0..10，完整native7、snapshot、rawbits、reward、goal、t/tr与
  off/on/off RGB pixel bits逐位相同；所有返回与计量顺序可消费。PNG本体hash与pixelSHA均保存，
  图像编码字节差异不能冒充像素相同，解码像素不一致绝不容差放行。
- 绑定原O/RGB/模型三权威、base/root/arm/policy/task/parent与new witness SHA、prefix raw15 bits。
- F的编译输入context必须精确等于已认证真实cp5/10/15向量及control6..15原normalizer两token；
  原图的实际复现证明保存context可用于当前起点，不新编码、不重推理。
- I编译输入仍是imagined cp15，不得把它改写为actual；同时记录相同真实起点与I输入SHA，清楚区分
  model-context输入与物理state。两策略都须通过相同真实prefix guard。

guard不得hydrate、set_state、reset、重绑goal或额外warm。control15无任何状态注入和观测替换。
不得把原O后15真实图/state/成功标签给本阶段动作选择器；模型新plan封存后只按字节执行。
这是保存反馈context的自然重放部署，不是部署时在线新编码/新编译；不称已验证实时闭环延迟。
原native自然t/tr立即停止。若止于或早于15则不下发suffix，guard不齐为missing/null；
若prefix不一致则engineering blocked，不能悄悄编码当前图、选另动作、放宽exact或继续后缀。

## 3. 满额shape与费用：单128MiB不能承载8条，提请固定拆分

计量来自冻结H30 native实现的 `contract/episode/observe/execute` 及其原reset/dispatch/render原语：
每轨迹两个reset，各两次set_state/一次goal_setter/一次reset render，reset共4 physics step；
每public step原10次physics；每observe为off/on/off三图，warm0..10与plan0..30均保留，
warm10与plan0虽同状态仍单独保存。新增guard引用这些已采图，不额外render，不复制O目录。

|原生轴|每新轨迹|建议每job（同base I/F）|全部8新轨迹|
|---|---:|---:|---:|
|constructor/close|1/1|2/2|8/8|
|reset|2|4|16|
|set_state|4|8|32|
|goal_setter|3|6|24|
|goal_bind|1|2|8|
|reset goal_render|2|4|16|
|warm/prefix/suffix public|10/15/15|20/30/30|80/120/120|
|public总计|40|80|320|
|physics_step|404|808|3232|
|显式render/PNG|126|252|1008|
|全部render（含reset）|128|256|1024|

满额PNG=3×[(10+1)+(30+1)]/轨迹；只有cp0/5/10/15/20/25/30算7个时刻，三图不是三个根。
满额physics=4+10×40/轨迹。public/physics是嵌套费用，不能相加当独立FLOPs。
全部8新轨迹裸RGB像素151,732,224 bytes，已超过128MiB；逐PNG256KiB硬上界合252MiB。
所以不能把8条塞单128MiB job，也不能静默缩图、删on/off、删IO、降精度、扩cap或按结果取消轨迹。

**提请MAIN批准固定4-job拆分，而非本协议自动授权：**
job0=root0 transfer I/F；job1=root0 joint I/F；job2=root1 transfer I/F；job3=root1 joint I/F。
每job独立输出目录，固定2轨迹；全局12槽索引与8新轨迹保持，不按运行结果重排分组。

建议每job硬caps：总128MiB、reserve8MiB、PNG≤256KiB/张且最多252张（63MiB），
journal≤32MiB、所有其他metadata≤16MiB，floor200MiB；63+32+16+8=119MiB≤128MiB。
8MiB final reserve是新物理提案，不继承模型2MiB reserve；32MiB journal需要新实现写前计账强制拒绝。
四job总输出最大512MiB，分组分项预留合476MiB；不是原128MiB全实验保证。
这里只给shape硬上界，不声称actual journal/重复receipt必装得下；实现前必须以2轨迹完整实际字段
做RAM/磁盘fixture预算，含新增guard、12槽索引、完整PNG/事件/行/receipt重复与失败final。
若超过任何组cap，报告MAIN，不能先运行再扩。四job不复制上游物理目录/原28encode数组。

建议whole900s/job（verify前至report fsync/hash），4job合可用上界3600s且实际总wall另计；
不是沿用单次900s宣称全实验。floor要求每次开job前至少200MiB+128MiB可用，产物持续累计。
最终completion外部SHA/边界计时沿新模型v2的严格模式；不能用report出版前elapsed冒充inclusive。
无NN/encoder/projector/predict/J/VJP/backward/solve/SVD/train，均为新增0；IO/CPU/physics显式计账。
模型216predict/8J、旧228predict/4J、原O 160public/1616physics/504PNG及28encode各自单列继承，
不抵扣本轮320public/3232physics/1008PNG，不称等算力。

## 4. 事件、失败与saved strict

新schema自有12槽总索引、四job固定任务表、每job两新rows与O继承引用；不把继承O叫新reset或新样本。
每operation intent/return/pending、requested/actual raw、所有native7/snapshot/RGB/goal均保存。
root_guard与feedback_prefix_guard全字段/时序入journal；control16必须在后者已fsync之后。
消费者精确重建全事件链、source/全部产物inventory、每图文件及像素、两个guard与raw30时序。
不得过滤未知event、只比末状态、宽容LF/hash ABI或借旧H15/H30 outer validator伪造成功。
完整资源账以实际durable intent/return为准，不按“已满30”倒填满额计数。

异常保首因、全部partial与cleanup独立错误；每env finally close，hook恢复不改旧globals。
guard失败/超wall/超输出/cleanup失败不退款、不重试、不替补；四job运行/未运行状态均保全局分母。
原生earlystop正常保留t/tr与首次成功event-time，不继续凑30；缺cp明确null+原因，不padding。
计划完整但原生早停与工程失败是不同字段，不把unknown/missing填False或Q=0。
缺模型export则该固定槽不执行，仍保留对应I/F slot；不能把父O作为替代动作。
全工程完整并非科学放行；P1/P2/C3与physical_release门不随任务success变化。

## 5. 对比指标与后续编码范围

每base O/I/F只在真实共同control时刻对比jointXY/objectXY/wrapped angle/完整native7（含速度）、
原native reward/success/first-success/t/tr。原native成功函数原样，不用新wrappedangle诊断替换原判定。
control15之前应exact相同；后15是同长度目标，earlystop后只列event-time终态，不当control30终态。
cp0/5/10/15/20/25/30固定，全部12策略槽；每根/每base各报，不能只保有末帧或成功子集。
O是历史已认证轨迹，I/F是本轮新重放，重复运行/时间差限制明确披露。

物理adapter不计算新的实际encoded Q。若未来要Q_actual与模型Q对照，须另批新HOSTCUDA桥与预算，
不能拿旧O后15编码当I/F后缀响应；即便动作相同也不在本协议免费去重。
prefix编码继承资格只由本轮exact guard认证，不能解释为新神经parity或在线观测闭环。
O→F不是纯反馈信息因果（额外优化费用不同）；I→F同优化次数/域，比较模型输入信息差异，
仍受known2、保存反馈/历史O与实际支持限制。跨transfer/joint不称纯模块化因果或等FLOPs，不宣称C3。

## 6. 待批接点与封存次序

模型v2实际prepare/report/completion/完整receipt已完成，按第1节四pins固定接入，禁止混用旧版本或仅凭stdout状态。
MAIN批准固定4-job分组、每job119MiB分项/128MiB总、8MiBreserve及900s边界后，才可实现新source/tests/doc。
未来接口：prepare(all_model_and_parent_pins)→固定12槽/4job manifest；
run(job_index, explicit_MAIN_AUTH)→两条新连续物理轨迹；consume(report_SHA, completion_SHA)→own strict；
join四份strict仅连接固定身份、缺失与实际计账，不择优或新签上游fullstrict。
此处没有可执行物理CLI，不借旧runner命令假称已可运行；下一阶段才交完整prepare/verify/run/consume。
当前只协议、无RAM执行器或生产，全部旧冻结文件不变。
