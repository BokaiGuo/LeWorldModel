# P2 full30：28位置同动作RGB桥协议与接点

2026-10-29。仅协议，尚无本协议的production adapter或执行结果。
不修改已冻结H30 physical三件套、planner四文件、原协议或旧RGB桥。
采用documentation-and-adrs技能记录API与复用边界；没有NN/physics/fullstrict/生产prepare/子代理。
MAIN负责审读后另行授权HOST CUDA。P1/P2/scientific/physical release全部false。

## 1. 固定权威及尚不存在的输入

根目录 `/home/gbk/code/LeWorldModel/le-wm-main`。模型三个实际权威沿physical.PINS不变：

|资产（相对根目录）|SHA256|
|---|---|
|`results/agent_p2_two_program_postcontext_20261029_v1_prepare_MAIN01.json`|`da1803815f14a1e3ee254e7915dc781d66367c7f96b011547c10f0736ba3f9c1`|
|`results/agent_p2_two_program_postcontext_20261029_v1_cuda_MAIN01/report.json`|`99f2b5edceae839bc3b0387ac93ef1d866a3fb74ec869ec7a30ceeaf4211ae00`|
|`results/agent_p2_two_program_postcontext_20261029_v1_consume_MAIN01.json`|`af2890ae4d330f69e8b01c7a144786ee454b08091e6d79ef63fa1b42e0bf6c5e`|

新physical三权威现已实际交付，均相对上述根目录：

|资产|实际SHA256|
|---|---|
|`results/agent_p2_two_program_full30_physical_20261029_v1_prepare_MAIN01.json`|`04d4a887e09381cc9869d76b259af41159d6678269c48e5a3cf0caa42df66100`|
|`results/agent_p2_two_program_full30_physical_20261029_v1_MAIN01/report.json`|`39dae04ac59d9fd1b1e3e65dfe675a4627c13ada00d8c065e655053018b35c90`|
|`results/agent_p2_two_program_full30_physical_20261029_v1_consume_MAIN01.json`|`aabc5cce3855021eba0fe5751c0ae87809cfefb4c2ed54b0d2899048fd6b3cd8`|

MAIN已报告完整native strict成功；本轮仅核文件SHA及report metadata/schema，没有调用consume。
不得把producer源码SHA当作物理执行receipt。未来桥仍要求CLI显式实际path+SHA及全部bindings认证。
physical源码实际SHA：`df65ccb93a800dea8b8e810b488b4818a70f058ee721b685dd49ef10d680a4a9`；
其tests `4a88745f50151dbd4c1926d796b531d6e0b60bd5b479cdb1bceaf6fafeeff3a9`，
doc `33385238923507676f8f0cd1ede0cb4ff2965b212f60353f0df99282978959ec`。
上述文件继续冻结，不能为本桥改变字段、补帧或重写旧receipt。

未来桥使用独占stem `agent_p2_two_program_full30_rgb_encode_20261029_v1`，自有prepare/report/consume。
不冒充旧full10、full15或H3 16位置schema。模型三权威是实际产物，非拟用文件名。

## 2. 已存在的physical接口与字段

源码 `experiments/spectral_reachability/agent_p2_two_program_full30_physical_20261029_v1.py`：

- `validate_execution(p)`：自有H30 prepare/ordered plans/raw30/task/model/context/action连接验证。
- `rehash(bindings)`：绝对regular非symlink文件完整hash封印。
- `consume(output,expected)`：自有完整physical saved消费；本协议不调用。MAIN实际消费一次后交receipt即可继承。
- receipt字段：`physical_report_sha256, prepare_sha256, plans[4], rows[4], bindings, counts, completion, byte_census`。
- receipt边界：`full_upstream_rebuilt_here=False, own_native_saved_reconstructed_here=True`；完整物理报告状态为
  `full30_four_native_recorded_NOT_admitted`，errors必须为空。合法native earlystop仍可为完整工程记录。
- physical report的 `execution_binding.prepared/prepare_sha256/planner_authority` 连接原模型三权威；
  `journal_sha256` 连接全部native_history_guard与真实动作事件；`completion`在receipt绑定最终wall与report SHA。
- 每个plan：`index,root_slot,condition,arm,task_id,selection_digest,commands[30,2],planner_plan`，
  加原native root/goal/goal_pixels哈希。planner_plan保留完整root_context、6contexts、goal192、原rawbits、witness和源program身份。
- 每个row：`plan,initial_exact,root_state,goal,trajectory,checkpoints,missing_checkpoints,responses,own_returned,status`。
- `checkpoints[]`只有 `step,png,file_sha256,pixel_sha256,state,snapshot`，没有path/condition/task字段；
  桥必须与同row.trajectory（phase=plan、overlay_index=0、condition=off）的唯一对应frame完整join，再构造绝对path。
  不能照抄旧桥“checkpoint已经包含这些字段”的假设。

所有4槽都保留，固定root0 transfer/joint、root1 transfer/joint，arm为
`two_program_transfer` / `joint_two_round`。不能因同raw、同root、同Q或PNG相同而dedup。
实际report四槽均initial_exact=True、own_returned=30、7cp齐、missing=[]，每槽126PNG；
160public/1616physics/504显式render均返回且errors=[]。本协议仅只读核该元信息，神经输出尚未生成。
root1两臂即使相同仍各保留7位置，不去重，也不因已见结果更改方案。

## 3. 28位置确定映射与同动作验证

slot索引 `7*plan_index+j`，j=0..6；control cp=[0,5,10,15,20,25,30]。

|control cp|absolute步|角色|冻结模型参考|
|---:|---:|---|---|
|0|10|root anchor re-encode|planner_plan.root_context.latent_history[-1]|
|5|15|H6 endpoint1|planner_plan.contexts[0].latent_history[-1]|
|10|20|H6 endpoint2|contexts[1].latent_history[-1]|
|15|25|A terminal/B input接缝诊断|contexts[2].latent_history[-1]|
|20|30|H6 endpoint4|contexts[3].latent_history[-1]|
|25|35|H6 endpoint5|contexts[4].latent_history[-1]|
|30|40|H6 terminal|contexts[5].latent_history[-1]|

transfer在control15没有reset，B已在本次模型A完整postcontext重新付费预测；此桥不重新调用B。
joint同30raw、two-round既定动作，不从物理或RGB结果重选。程序数transfer2、joint1，不改成同定义。
对于cp c，要求所有已返回plan action1..c实际bits等于原commands[:c]（包括signed zero）、
native dtype float32、time连续、无post-terminal动作；warm10返回bits必须原位一致。
不是只核terminal同值，也不把MPC/别的root轨迹当同动作标签。

cp0没有plan response，这是合法anchor：核row.root_state/root_hash、cp0 state与snapshot及原goal，不能要求response step0。
cp>0要求唯一同step response，native7/state/snapshot与frame/checkpoint相同；关联原goal/native指标，不改成功门。
原native_history_guard必须已认证：实际warm0/5/10 RGB及raw10、root_context、任务在首个plan intent前封存。
引用MAIN完整physical验收范围，并检查当前文件/journal seal；不伪称本桥再次完整运行physical.consume。

cp c因果raw10=`concat(original_warm_raw10, commands[:c])[-10:]`，原raw按float32bits记录；
normalizer原float32 subtract/divide恰好一次，不能乘100或再tanh。
warm0/5/10的control标签是−10/−5/0；不能误标为plan0/5/10。

若未到cp，固定位置source=null、actual192/Q/MSE=null且附原因，不编码、不补帧、不取earlystop终态替代。
根未认证（initial_exact=False）不能把warm图当cp0；物理工程失败不签本桥完整成功。
实际返回到cp的native earlystop允许该cp编码；其后cp保持null。全28分母不随可用数变化。

## 4. 可复用源明细：只取policy-free接口

以下均位于 `experiments/spectral_reachability/`；实施时完整闭包hash封存，不只pin一层别名。

|源文件及当前SHA|允许复用/参考|不能复用的外层假设|
|---|---|---|
|`agent_p2_contextual_physical_rgb_encode_20261028_v1.py` `fa1d44fb2b753b2deb2c1743cfa0d0496b4fdea42f92d4190cce231260d5375e`|`load_pixels, latent, same, mse, verify_bindings, Trace, counts`；Trace接受自定义expected calls，scope.phase=encode|fixed_rows/build/process/run/validate/consume写死8plan/24encode+8predict，不可直接用|
|`p1_moving_overlay_recompile_rgb_cuda_20261027_v1.py` `70c03508d775b8b488c12133da929bae607d08d6163d5e1c6578edd9a2422bcc`|`load_image, config_shapes, load_array, flat, syncdir`；array transport primitives|其Budget/Store外层默认32MiB/2MiB，须显式新预算，不能继承旧physical schema|
|`agent_p1_teacher_library40_encode_20261028_v2.py` `f7e3abb842cc185c77fe217922cd83ce709f4a6bf8fce0d9337c87bf4f66f75b`|`preprocess_pixels`及原224输入数值核对|旧teacher候选/身份/门不迁移|
|`p1_constructive_blind_encode_cuda_20261027_v1.py` `c3b071de1ad0b322e1fff77a716550f42014f0303a6b5fa09271fd4e55609b4a`|冻结模型hash/frozen检查、原生API禁止guard与Ledger（按实际接口设置28image-cap）|不继承旧image-count、任务或receipt|
|`agent_p3_h3_witness_encoding_20261029_v1.py` `c3c7f4421ffe843acc845ba9ef1bc19424ae66e4d08fa96b7a3fc63f22bb4748`|参考`precision_state/model_inventory`、encode回调、visual IO链、sealed-output先于分析的实现|16位置fixed_rows/process/validate/context_chains及build嵌套fullstrict不可调用冒充H30|

原生encode回调接点：`p0._image_tensor(pixels, transform=p0.image_transform(), device=CUDA)`，
保存preprocessed后 `model.encode({'pixels':x})['emb']`；原CLS→projector，不调用model.predict。
`physical.saved.overlay.traced_scope`/`visual_bridge`只是旧模块中policy-free追踪/保存数组校验原语，
实施前须封存其真实owner源，不能因别名路径含旧physical就借用其任务validator。

## 5. 新接口草案（尚未实现，不是已可运行CLI）

`resolve28(authorities) -> {plans[4], rows[28], bindings, inherited_resources, expected_calls}`：
authorities显式输入physical prepared/report/receipt及各SHA，加上述model三权威/源SHA。
独立重构本协议映射、identity、原raw/time/RGB/task/source join及全部hash；不嵌套重复旧fullstrict。
`full_upstream_rebuilt_here=False`，注明依赖MAIN实际完整physical/model receipts，不能只看status就接受。
每row固定保存plan_index/root/arm/task/checkpoint/absolute step/witness、goal192、predicted192或anchor192、
fullrawbits、causal_raw10bits/normalized action history、source frame、physical status、native response/缺项原因。

`process28(p, store, encode_callback)`仅对source存在的位置调用一次encode；所有位置先封存roster。
`validate28(p, report, events, load_array)`严格重建全部IO、scope/模块/资源账和null语义；未知event不filter。
未来自有prepare/verify/run_cuda/consume；此协议不提供冒充已实现的命令，也不授权worker实现后自行跑生产。

28encoder/28projector上限，batch=1、no dedup、no retries；实际expected calls等于source存在位置数n≤28。
0 action_encoder/predict_API/predictor/pred_proj/J/VJP/backward/solve/SVD/physics/train/新goal encode。
固定原policy/checkpoint/model tensor hash、identity192 Q、goal192、normalizer，eval/frozen。
HOST CUDA only、TF32 off/autocast off/MATH、allocator3GiB/reserved4GiB，加载峰值也记；
未知loaded module source拒绝，原model before/after、全部sources/weights前后hash不变。

每位置保存pixels uint8[224,224,3]、preprocessed f32[1,1,3,224,224]、encoder实际input f32[1,3,224,224]、
完整hidden[1,257,192]、projector input[1,192] CLS、output[1,192]、native latent[192]。
严格保存完整192vectors，不只Q；不增加latent归一化（本模型encode输出已是原native表征）。
28位置全部数组原payload上界43,524,096bytes（含两份preprocessed实际IO），不得隐含多份hidden复制。
建议新output64MiB、arrays48MiB、events+metadata8MiB、final reserve8MiB、floor200MiB、wall600s。
NPY headers/压缩开销须在实施RAM完整bundle证明≤48MiB；超限先报MAIN，不扩cap或丢IO。
输入physical目录单独沿256MiB总/64MiB journal/16MiB metadata守卫，不把它全部复制到新64MiB输出。
inclusive wall从verify前到report fsync/hash、completion最终检查；partial/cleanup沿新own失败路径，不签超时成功。
以上为待MAIN批准工程caps，不冒称旧RGB runner已直接支持28位置。

## 6. 数值消费与可解释边界

先seal native outputs/IO/资源/错误，后saved分析。cp0实际编码须与原root192逐位一致，保存anchor MSE，
不能用新anchor覆盖原task或重新拟合normalizer/Q。cp>0独立float64重算
MSE(pred192, actual192)、Q_pred=mean((pred-goal)^2)、Q_actual=mean((actual-goal)^2)、signed/absolute Q误差。
cp30 Q_pred须等于原predicted_Q；中间cp Q由冻结对应向量独立重算，不把terminal Q复制到所有cp。
原predict native3输出的last endpoint才是对应预测，不把native3误认为三个未来独立时刻或shift history。
cp0只anchor，不混入24个forecast分母；最终输出固定28rows、4anchors、24forecasts、所有missing计数。

可输出实际history链（不调用预测）：cp0沿原root_history，之后按实际新编码逐5步roll并插入原归一化动作token。
这只是保存数据派生context，不是反馈控制实验；原predictions仍来自原imagined history，不偷换成actual-history预测。
缺实际latent后链标missing，不用pred填真实history。尚未授权新的history tree或suffix重编译。

同root跨臂比较只用共同到达cp及其各自实际raw，不叫同动作因果（两臂raw可能不同）。
不能跨时距与full15合并；joint外包球更大、更多计算，notmatched；known2并非4独立根。
 inherited228predict/4J、新physical计数、新28encode/project及IO/wall分栏；未知FLOPs标null。
此桥仅补同动作模型→实际RGB误差证据，不开P1/P2、不宣称组合效能或物理任务改善。

实施验收fixture：全28映射/bitexact anchor、cp0无response、cp15不停机来源、早停null、不同raw/time/root join拒绝、
全部输入/权重/source封印篡改、真实Store/array/event落盘reader、未知event、precision/资源/timeout与完整字节预算。
物理三权威已齐；尚缺新28槽桥的source/tests/doc实现与MAIN审读/授权，不是模型输入缺失。
MAIN另写`main_p2_full30_same_time_summary...`独立28cp/14pair物理标量分析，本桥不重复该工具或产物。
本桥仅提供严格向量/RGB/model-Q误差及其原生来源join，不按物理scalar分析结果调配置。
