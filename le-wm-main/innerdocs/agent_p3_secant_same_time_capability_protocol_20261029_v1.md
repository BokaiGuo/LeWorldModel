# Secant：同时间零/原progress对照与实际cp10能力probe（提请批准）

仅NEW协议，无实现/运行授权，无NN/physics/fullstrict；不修改冻结件。使用experiment-design
组织固定分母/判据/费用，不采用训练、调参或择优扩展建议。协议形成时已知secant模型资格
4/0/0/0、唯一root0H1 delivery。初稿固定动作/分母/预算后，MAIN补发candidate物理终态；
本版仅补绑定及接口要求，不据native_success=false改对照/动作/判据或扩大预算。
本设计是known2探索性闭环检验，不是fresh确认；不得据一个可派发candidate宣称P1/P3通过。

## 1. 最小问题与明确的动作来源

要检验的具体链：从同actual-cp5完整历史出发，candidate付出更差即时原生进展后，
是否在**同时间实际cp10**形成更有用的局部目标响应，并由同一个冻结H1编译器得到
比两个固定对照更好的**实际cp15**结果。必须测实际cp10而不是复用cp5 J/imagined postcontext。

固定原四逻辑槽 `(root0,H1),(root0,H3),(root1,H1),(root1,H3)`，每槽三个角色：

|角色|control6..10的raw5|选择规则|
|---|---|---|
|C：secant|原delivery.commit_raw5|仅原唯一selected且delivery/物理guard均认证；另三槽null|
|Z：zero|little-endian float32全+0，shape(5,2)|事前定义，不由模型生成、不补tanh|
|P：original progress continuation|原shared-panel prepare `/layers/{root}/commands[5:10]`|原固定line_search_GN raw10后半包；不是在cp5重新优化|

P的prefix必须与同一layer.commands[:5]和cp5 provenance.prefix5逐位相等；不得用
H3新winner、最优旧物理分支、native结果或newmodel输出替代。若P原动作域外/不精确f32，
明确blocked，不clip、不从其它bank找替身。Z/P均使用原goal、raw域、renderer/off、mean/std。

完整12个逻辑角色槽保留。Z/P与h无关，固定按(root,role)物理去重，H1/H3引用同一个
真实轨迹SHA，不把重复引用当独立样本：Z/P共4条，C共1条，共**5个cp10端点**。
C的另外3槽no_witness仍null；新增Z/P不是其fallback、不是扩出4个可交付candidate。
8个C-vs-baseline比较槽（4逻辑槽×Z/P）全保留；当前只有root0H1的两对可有值。
原有资格/排序/科学门不变，不在这两个对手里事后选一个有利对手。

## 2. 可复用来源及精确前置

以下相对路径根均为 `/home/gbk/code/LeWorldModel/le-wm-main/`。

|来源|路径与SHA256|
|---|---|
|固定P/root/goal/history|results/agent_p3_shared_panel_20261029_v1_prepare_MAIN01.json；7ca85050eeab4e47ed35cf84c284ed7e08fc64bce8064d7db9450074045b643a|
|该shared-panel完整receipt|results/agent_p3_shared_panel_20261029_v1_consume_MAIN01.json；d23ffa48bc954306bc27a15f924da8daafffc9b4bc76a91e59aa949ab1abbafd|
|actual cp5 task/provenance|results/agent_p3_actual_cp5_task_builder_20261029_v1_prepare_MAIN01.json；ce6670d84f6f0f712bdfe5e1b7471be65b0441338c18010f79b3967ef28397c7|
|原history/物理来源桥|results/agent_p3_physical_commit_authority_bridge_20261029_v1_MAIN01.json；9c47743431166ddb2564bf5dea5d5335e794d98ec0873a8b90417e632eefc97d|
|固定C模型完整receipt|results/agent_p3_secant_delivery_20261029_v1_consume_MAIN01.json；6f583507a9e5ce6d0f859032344fc303527c250b5633fa8d3482ded3615e4fb6|
|C冻结物理adapter，仅复用guard语义/来源|experiments/spectral_reachability/agent_p3_secant_commit_physical_20261029_v1.py；a065c8f652919d8315f7848c70e9b944ad6aec0717bc6bbe310b54b224c72aea|
|C实际物理report|results/agent_p3_secant_commit_physical_20261029_v1_physics_MAIN01/report.json；f64d2843ecf0c217a5581d25a10fa77ccf78c372bf1ddc2aca8995ab4af0a616|
|C实际物理completion|上述物理output/completion.json；f68fb8ef6638a2fa3c74a8139d1b7a517a7c2d36df02122104b759998b470d75|
|C实际物理prepare|results/agent_p3_secant_commit_physical_20261029_v1_prepare_MAIN01.json；a085c3c6a609664e61305b156ce68d1cbdf44296634a42a8bc5416e7091fe295|
|C完整物理receipt|results/agent_p3_secant_commit_physical_20261029_v1_consume_MAIN01.json；d0c5a34b14dc84ffece573d904e3950e54628df307987ee77fcb6265ca91b638|
|真实RGB→native192接口|experiments/spectral_reachability/agent_p3_h3_witness_encoding_20261029_v1.py；c3c7f4421ffe843acc845ba9ef1bc19424ae66e4d08fa96b7a3fc63f22bb4748|
|native signal/replay接口|experiments/spectral_reachability/agent_p3_parent_native_cuda_20261029_v1.py；eeea5710d4171b7c43f99d1f2b5ad2412e03a25ba14e8fc5589ea5ef73f57b3b|
|固定suffix chart/组件|experiments/spectral_reachability/agent_p3_parent_r_native_components_20261029_v1.py；ce48a8c665ad5434fe65f5238b303570ffc18183956d24a0c830864fb5f12c7a|
|原bounded solver|experiments/spectral_reachability/pusht_full_packet_compiler_20261027.py；f359e7b21d1970f95f065dbee206d2dc9359c2eaff7ddbcd7611e07686c6cf2a|

MAIN已完成C完整消费；下一prepare显式接收并核上表四权威，依完整消费字段认证
C的root0H1 cp10已到达、66观测/动作/原51guard及原goal相等，才能把该端点列为可用。
worker只核实际文件SHA及绑定字段，未重跑原执行或其生产fullstrict。
失败保原结果、C槽null/blocked，不重放求过门。
现有shared-panel layers的arm确为line_search_GN，root0/1 selection_sha256分别
`97bcf5e4d3fbf5ed20db407cc7bbeadedf5d8a663bd4aad640c0056e45b71ed9` /
`9120059452c4a1760e4b355f61051639b9445f29d6a628f7f419499a96f40d60`。

本最小预算不假定旧Z/P cp10轨迹已经获同合同认证，固定新采4条独立原生对照。
若希望复用其它旧物理结果省费，必须另在运行前逐字段认证完整source/raw/history/guard/
native/render身份并修订封存预算；不能看此次结果后决定哪些旧轨迹“足够相同”。

## 3. 阶段A：同时间真实cp10采集与输入ABI

新4条Z/P各自constructor→原reset/place→warm10→同prefix5→原51 exact guard→固定raw5。
原native termination不抑制；未到cp10保missing，不将early terminal贴作cp10，不padding。
C直接引用当前冻结部署的真实cp10，不能用想象latent、cp5标签或旧R物理对象替代。

固定5个端点顺序 `(r0,C),(r0,Z),(r0,P),(r1,Z),(r1,P)`，每个只新编码其off cp10 RGB一次。
原RGB224预处理→encoder full(1,257,192)→CLS→projector native192，batch1、f32、无额外
L2/whitening/goal重编码。继承原abs10/control0、abs15/control5的已认证实际latent；
新context历史为abs10/15/20即control0/5/10。incoming raw10必须为(prefix5,该角色raw5)，
原mean/std一次f32 normalize，逐位接成action_history(2,10)。
重建local binding保留原base task/root/goal/Q/normalizer/model，新增endpoint/history SHA，
不得继续使用cp5 task_id冒称新cp10输入。实际state/RGB/encoding/动作/完整三槽context全绑定。
重复H角色只引用，不重复编码；5个角色即使PNG相同也各付1次，不据实际相同结果删调用。

## 4. 阶段B：能力测量与唯一固定probe

同一冻结HOSTCUDA模型、原TF32off/noautocast/MATH guard，禁止训练/适配/重新拟合Q。
为排除跨job cp5 baseline疑虑，本协议事前固定**新测两根cp5 H1参考J**，不是读取结果后补。
再对全部5个实际cp10输入各新测H1：nominal(rawzero)与AD primal、完整192×10 J、192 VJP。
两cp5参考和五cp10测量同job、同loader/source/loaded tensors、同zero raw坐标、相同全3 ABI。
保存全部输入/output/J；不得将AD primal再假称另一组独立对照。

对这7个输入均只运行原bounded_ridge(J,goal−nominal)，保持identity192投影、
原DAMPING=.002、CAP=sqrt(5)*.75及原实现，不调半径/截断秩。保留own_reach_value、
linear_terminal_cost、ridge_penalty、完整10 singular values/J；复用solver返回谱，不另做SVD。
cp5两个solve仅做同job参考，**不执行其动作**。

每个cp10只编译一个U：`rawU5=tanh(original_solve.params.float64).astype(float32).reshape(5,2)`，
沿原components.suffix_raw chart，恰一次mapping、一次原normalizer。不是挑一组spectral
候选，也不是把线性R值当实际性能。U须独立whole与segmented各实际重放一次H1（均[5]），
从同完整实际cp10 context开始，完整native IO逐位相等且raw合法，否则该角色probe blocked，
不改参数/不选第二项。未通过不能执行U；Z2 continuation也不能冒充其成功部署。

全7 J/solve、全5 U/10 replay先durable seal，再计算全部22项原Q：5 actual-cp10、7 nominal、
10 replay末端；原goal是条件允许进入编译，未来cp15标签不得进入任何编译/选择。
所有5角色都保留，不按cp10进展/R/singular values筛probe，不按C物理效果扩大搜索。

## 5. 阶段C：实际能力probe必须有zero延续对照

对5个cp10角色固定各2条后续物理轨迹：U5与Z2=rawzero5，共10条；不运行模型后的重选。
新env必须原reset/warm10/prefix5/角色raw5自然重放到cp10，不reset/hydrate poststate。
保原51 cp5 gate，并在control11/absolute21 intent之前核**完整66观测和实际cp10**与阶段A/C
reference exact，以及原history/raw/goal/RNG/public状态；durable accepted guard后才执行probe。
若无法exact复现，保工程失败且整比较不认证，不容差、不重试挑运行、不补画或重建contacts。

每条只执行control11..15=absolute21..25的固定5动作；原成功/terminated/truncated早停，
没有cp15则该槽missing，不用earlystop帧充终点。固定10槽全保留，继承各cp10来源hash。
U/Z2各自独立重放同路径到cp10，不能共享一条已受另一个probe影响的live环境。
native success原agent+object jointXY及角度条件、native7 reward（含velocity）不改。

## 6. 最小可证伪判据：不把相关性包装为能力因果定理

定义D(s)=原native7 Euclidean distance=−原native reward；同时全量报告原jointXY、
wrapped angle、velocity、native_success，不能以object-only或RGB Q替换成功。
令D10(a)为实际cp10，D15(a,U/Z2)为同时间probe终点，
`G(a)=D15(a,Z2)−D15(a,U)`，正值表示该固定controller比passive zero延续更有用。

对每个固定B∈{Z,P}分别报告，不取max/有利对手；预先固定完整证据链为以下严格无容差合取：

1. `D10(C)>D10(B)`：C即时原生进展更差，非只看旧模型I。
2. `R10(C)<R10(B)` 且 `G(C)>G(B)` 且 `G(C)>0`：原目标相关模型可达量和
   实际probe主动收益同向；cp5同job参考的R/J差异全报，不把预估改善当真实能力改善。
3. `D15(C,U)<D15(B,U)`：相同后续H1编译器下实际最终距离反超。

只有对Z和P两者都成立，才可描述为该known-root上的“本固定两包程序的延迟收益及
目标定向probe可用性证据”。任何一项反向/相等/missing都如实保留，不改门、不筛角。
原native_success若仍全false，不能称任务成功/科学准入；连续距离反转与成功必须分栏。
全8逻辑比较槽、5实际端点、10probe槽一并输出；候选缺失的3组不能变成0收益或失败样本。

该设计能排除：cp5充cp10、时距不齐、仅预测poststate、passive drift充主动probe收益、
只比一个有利baseline、单次candidate-only轨迹充闭环对照。
仍不能排除/证明：J的变化是唯一中介（state/context/残差都随动作改变）、全方向控制容量、
其它目标/动作bank/独立根上的能力、全策略树或全C3。U按同controller但各context重新编译，
故这是共同算法下的局部目标定向能力，不是相同物理动作的全局系统辨识。
若仅R变好而G/最终native不改善，机制链不成立；不能事后改成谱图好看即通过。
独立fresh确认仍需原科学合同的认证cohort，known2及重复重放不增加独立样本数。

## 7. 事前新付账（完整可用时；不padding缺失）

|新计算轴|固定上限|分解|
|---|---:|---|
|encoder / projector API、samples、frames（各）|5|五cp10 RGB；goal/前两帧继承|
|predict API/samples；AE/predictor/pred_proj calls（各）|24|7信号×2 + 5 U×2布局|
|上述context tokens/proj rows（各）|72|24×3|
|J API / AD primal（各）|7|2 cp5参考 +5 cp10|
|VJP / grad API / raw-backward callbacks（各）|1344|7×192；AD primal已在24内|
|Jacobian元素|13440|7×192×10|
|signal_compile / solve / arraySVD（各）|7|原solver一次；不另做spectral cutoff|
|program_compile|5|每实际cp10一个U|
|独立replay calls|10|5×whole/seg|
|原Q metric调用|22|5 actual+7 nominal+10 replay|
|其它NN/训练|0|无encoder重复、无extra best replay|

实际module traces与load/source/model前后hash及真实attempt/return全付费；FLOPs/backward
engine底层调用次数未知不得填0，不称equal总算力。每次实际native/capsule必须持久IO后返回。

|新物理轴|阶段A四对照|阶段C十probe轨迹|合计|
|---|---:|---:|---:|
|constructor/close（各）|4|10|14|
|reset / set_state|8 /16|20 /40|28 /56|
|goal_render / goal_bind / goal_setter|8 /4 /12|20 /10 /30|28 /14 /42|
|public_step|80|250|330|
|physics_step|816|2540|3356|
|explicit_render / PNG（各）|264|810|1074|

probe每行warm10+prefix5+角色5+probe5=25public，2reset共4内部physics，再250=254；
RGB triplets=3×(warm0..10十一帧+plan0..15十六帧)=81。非免费reset/goal bodies；
C当前已付20public/204physics/66PNG作为继承，不能从上表扣掉probe的真实重放。

不能把1074图塞进现64MiB目录。工程建议固定阶段A一个64MiB新job（264图），
阶段C十个各64MiB新job（每job81图），各含8MiB reserve/floor200MiB/执行wall900s；
每job完整metadata/PNG census和completion外部SHA，最终汇总保所有11job与失败分母。
CUDA工程建议独立2GiB输出含256MiB收尾reserve、wall1800s、GPU allocated3/reserved4GiB，
实现前仍需7全J/1344 VJP原生IO重复存储shape census，不足先报MAIN，不缩分母/精度。
这些仅待批准的工程上限，不是现有64MiB执行器的原地扩容或科学配额优势。

## 8. 最小接线交付与当前停止线

若MAIN批准：先做固定Z/P FIRST5新adapter及完整saved consumer（复用原native低层与51guard，
不改已冻结secant adapter）；再做5端点RGB→实际cp10 task桥、7signal/5U的HOSTCUDA runner；
最后新control11 guard的10槽probe native adapter及独立native标量分析器。每件单独审读冻结。
所有NEW执行CLI stdout只返回compact output路径/reportSHA/completionSHA/status/计数摘要，
完整report仍durable落盘；不修改已有冻结CLI，不把日志体积当科学错误。
输入ABI为原source/selection/raw5/hash + 外部实际C物理authority；端点桥输出实际history3/
raw10/normalizer/model/goal/PNG/latent SHA；probe计划输出actualcp10 task+J/solve/rawU5/
dual replay witness；物理consumer输出同时间native状态/收益，不用耗时receipt冒科学结论。

当前C实际authority已齐；仍缺MAIN批准本设计/新预算、四条新Z/P实际cp10；
五真实cp10新编码/七J/五U的认证部署witness；十条probe真实结果及独立分析。
本协议未实施任何上述运行，也未授权worker运行。P1/P3/scientific/freshness/physical_release/
independent_confirmation/full_C3所有门保持false；下一步是MAIN审定此固定设计和新增费用。
