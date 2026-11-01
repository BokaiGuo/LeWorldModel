# Secant delivery：known2 actual-cp5 FIRST5 原生部署协议

状态：仅供MAIN审读的前瞻工程协议，尚未授权实现或执行。拟用新adapter stem
`agent_p3_secant_commit_physical_20261029_v1`，source/tests/innerdoc均另建。
不修改旧physics、delivery或任何封存结果；不调用NN/physics/fullstrict，不生成未来receipt。
使用experiment-design组织固定分母与资源，不采用训练/调参/择优扩展建议。
已知原secant资格4/0/0/0；不是盲预注册、fresh确认或控制收益实验。

## 1. 来源与执行前必须补齐的authority

所有相对路径根均为 `/home/gbk/code/LeWorldModel/le-wm-main/`。

|已存在来源|路径|SHA256|
|---|---|---|
|delivery source|experiments/spectral_reachability/agent_p3_secant_delivery_20261029_v1.py|7b3449b27ec68d7f923ac0af93f186c651ccb97231894152cbfa9a01649f57d2|
|delivery prepare|results/agent_p3_secant_delivery_20261029_v1_prepare_MAIN01.json|ad289da249cb21190ee4b59748c2944c0866cea8c24be0be0a9af37e1ec7a15b|
|delivery协议|innerdocs/agent_p3_secant_delivery_protocol_20261029_v1.md|500bfc08e0b542f365c6912dfa1ec6333276035a33ac596ec3f21bf3f1b3d065|
|原secant完整receipt|results/agent_p3_reach_secant_actual_cp5_cuda_20261029_v1_consume_MAIN01.json|d96f97dd076894901edea43f6223cf8cb32da4dc630968fea8fae7834b143151|
|actual-cp5 task prepare|results/agent_p3_actual_cp5_task_builder_20261029_v1_prepare_MAIN01.json|ce6670d84f6f0f712bdfe5e1b7471be65b0441338c18010f79b3967ef28397c7|
|旧R物理实现，仅低层参考|experiments/spectral_reachability/agent_p3_actual_cp5_commit_physical_20261029_v2.py|a17ee3512ebb3289fdc862c9651641fa0e61a1670689518786701f3e64021776|
|原physical authority bridge|experiments/spectral_reachability/agent_p3_physical_commit_authority_bridge_20261029_v1.py|fd136726fc67eac13e91b3b21514eb8a2049d398b2e86fee0dadfdca588b6647|

delivery仍由MAIN运行，本协议不声明其终态。新prepare必须显式接收实际
`delivery_output/report_sha/completion_sha/receipt/receipt_sha/prepared/prepare_sha/source_sha`，
无默认未来SHA。report、completion、完整strict receipt缺任一项即blocked，禁止只信status。
MAIN的prepare须调用冻结delivery.consume（含completion_sha）完整重建并与持久receipt整份
canonical相等；它递归重建原secant saved consume。worker不执行这项生产fullstrict。
必须核四组、16候选、全部flags、source闭包、model/context/task/raw/资源与completion文件census。

原bridge仅认证原root/history/物理来源，不将旧R producer认证移植给secant。
由其绑定读取原source/reset recipe/warm_trace/prefix51/frames和SHA，不按文件名猜身份，
不从当前源码反推历史loaded bytes。新prepare保存每项实际路径、JSON pointer、SHA与完整闭包。
模型hash沿原delivery：`4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546`；
部署本身不加载模型，不新编码，不用该hash宣称训练lineage已认证。

## 2. 固定四组和唯一动作来源

固定顺序 `(root0,H1),(root0,H3),(root1,H1),(root1,H3)`；known roots=2，fresh roots=0。
始终保留四行和原16候选来源，不把一个root的不同h或候选算独立根。
原eligible顺序按固定 `(immediate,index)` 升序，唯一selected=首项；物理阶段不再选择。
仅当完整delivery重建确认该组 `model_delivery_verified`、whole/seg均通过且selected/raw一致，
才允许该行物理dispatch。selected验证失败不得试第二项；不能看原生结果换候选。

原资格为空的组保留 `status=no_witness`，selected/model_delivery/commands/native_success/
final_state/final_metrics=null；两路model verification引用按上游保留，不捏造物理零值。
这类行constructor/reset/dispatch均0，不跑zero/fallback，不计为物理失败。
delivery失败、有资格但未认证不是no_witness：prepare阻断整次新部署，不自动缩分母。

执行命令仅 `original_prefix5 + delivery.commit_raw5`。commit必须逐位等于选定源raw_edge前5行，
finite little-endian f32、shape(5,2)、raw[-1,1]；保存全edge/commit/lookahead原bytes和SHA。
禁止clip/tanh/inverse-normalize/重选/重映射。H3余10仅模型lookahead，永不物理dispatch。
H1与H3部署都只commit5；不能把H3模型末端当实物到达点。

## 3. 同reset → warm10 → prefix5 → exact cp5 guard

每个可执行组独立constructor，原swm/PushT-v1、224 RGB、render_action=False、off条件，
原constructor/seed/reset/set_state/goal绑定顺序照冻结native pipeline，不hydrate imagined cp5。
复用原place/episode/native_hooks/dispatch/observe的低层行为，不修改环境或其函数/全局。
原reset recipe中的set_state仅用于原初始reset流程，不允许跳过warm10/prefix5直达poststate。
原moving warm_raw10和原prefix_raw5 literal bytes逐控制步重放；保留contacts/cache自然演化。

在control6（absolute16）第一个public dispatch intent之前，必须写入且fsync accepted guard：

1. 原warmup0..10、plan0..5的完整51份off/on/off观测全五字段exact，包括PNG file SHA、
   decoded pixels、native state、snapshot及原receipt/时间身份；不只比末帧或object投影。
2. cp5 full native状态/goal/renderer、原prefix raw5、warm10实际intent/return bytes全部相等。
3. 三个actual历史锚abs5/10/15，即control−5/0/5，核原RGB/state/snapshot/source SHA；
   context role=`actual_observed_abs5_10_15_control_minus5_0_5`，不接旧control0历史。
4. 实际warm[-5:]+prefix组成incoming raw10；原task mean/std一次normalize的history tokens
   与delivery完整root_context.action_history逐位相等。latent3只能引用原认证编码及RGB绑定，
   本物理adapter不编码，不假称现场重新测得latent；完整root_context/task/provenance精确绑定。

guard复用旧v2.guard_proof的检查语义，但新写secant plan/producer身份验证和own proof schema。
无allclose/ULP宽容，无省略fullpublic/RNG字段；差异durable保存并阻断commit。
旧P5 coverage末位问题不能成为本协议放宽门的理由。
旧v2.PRODUCER/schema/status/assemble/material/run/consume写死旧R，不能直接调用整runner，
不能global monkeypatch或把secant对象改标签塞入旧consumer。

## 4. 原生执行、停止与独立计量

guard通过后control6..10=absolute16..20，原raw first5；每步native intent/return/failure、
实际raw bits、native7/state/snapshot/info/reward/terminated/truncated与PNG全保存。
沿原native早停语义，不抑制termination、不补步；partial保全部attempted/returned和缺槽。
guard失败无commit intent；工程失败不重试该组、不fallback，后续组留明确未派发原因。
close在finally执行且独立记账；cleanup失败不覆盖首错，不退款。

下面是沿原episode完全执行的逐行计数，不是本轮已付费用。
设e为具备认证delivery的组数，e≤4；当前资格至多e=1，是否通过尚未知。

|资源轴|每完整执行行|当前最多e=1|全四行设计上限|
|---|---:|---:|---:|
|constructor / close（各）|1|1|4|
|reset|2|2|8|
|set_state|4|4|16|
|goal_render（含原生额外goal bodies）|2|2|8|
|goal_bind|1|1|4|
|goal_setter|3|3|12|
|public_step：warm10+prefix5+commit5|20|20|80|
|physics_step：2×reset+10×public|204|204|816|
|explicit_render / 保存PNG（各）|66|66|264|
|NN/encoder/predictor/AD/training|0|0|0|

66=3×(warm0..10十一帧+plan0..10十一帧)，51=3×(11+6)为guard前完整prefix。
goal_render单列，不混作这66帧。observe/getobs和额外内部调用必须在实现时按绑定原源码
实测hook范围单列；不得把未hook的API写成0或据public步数假造实际调用。
resource receipt按setup/warmup/plan/closing及每个nested call计intent/return/failed，
分清wrapper public、内部physics和render；实际少于上限必须可由earlystop/failed prefix解释。
inherit模型搜索/delivery资源整向量另列，不混入本轮physics或称等FLOPs。

工程约束前瞻沿旧物理v2：output64MiB含final reserve8MiB、floor200MiB、wall900s、
prepare元数据128MiB。实现前必须全4行264张224RGB PNG（按编码最坏界）+journal重复元数据+
report/binding/completion完整shape-byte census；不足则先报MAIN，不缩分母/精度或暗加cap。
report fsync/hash后独立completion绑定inclusive wall和文件census；completion自写时间边界明确，
其后最终wall检查超限需durable failure且strict拒绝，不能沿用publication前wall冒证。

## 5. 接口、saved验证与批准前停止线

拟新增 `prepare/verify/run/consume`；本协议不提供可执行生产命令，不假装runner已存在。
prepare参数为第1节全部实际delivery authority，加原bridge路径/SHA和本协议SHA。
verify完整重建saved authority及plan；run另要求新MAIN专用physics authorization。
输出own execution_binding、canonical exactly-one-newline journal、PNG、四行report、completion。
所有文件exclusive新建，封存partial不覆盖；prepared/report/receipt/completion SHA均外部绑定。
新consume重建四组selection/guard/全动作/时序/计量/hashchain、全PNG字节与原native任务指标，
检查before/after source闭包以及complete文件census；不重新运行模型/physics。

成功定义不换：native joint agent+object XY与原角度条件；native7 reward含velocity，
不能把object-only距离或RGB Q当完整成功。保留native成功/早停/工程失败/no_witness四类，
四组结果全部展示；不从唯一派发组推广全四组收益，不称与旧baseline公平排名。
P1_admission/P3_admission/scientific_admission/physical_release/execution_authorized/freshness/
full_C3_demonstrated/independent_confirmation全部false。操作授权只允许MAIN实际调用，不打开科学门。

实现验收RAM fixtures：全四分母、no_witness全null且零constructor、唯一selected不fallback；
raw/domain/signed-zero/source/model/task/context/goal mutation拒绝；guard在control6 intent前durable；
51观测任一字段差异拒绝；H3只first5；earlystop/guard failure/cleanup与完整partial；
真实Store双event→canonical reader/hashchain→consumer；全264PNG上界和completion census/超时拒绝。
下一步仅待MAIN批准本协议后实现独占source/tests；当前无新fixture、无生产、无future SHA。
