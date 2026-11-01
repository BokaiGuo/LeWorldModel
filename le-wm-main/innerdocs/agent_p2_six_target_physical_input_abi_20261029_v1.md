# Six-target物理adapter输入ABI连接说明 v1（非runner）

范围仅为保存已知来源和接线约定，等待Pauli六槽物理协议定稿与MAIN另行授权。没有physical prepare/ready标记，没有未来paid report/receipt SHA；不实现driver、不运行NN/环境/physics/fullstrict。旧四槽结果不能拼入新六槽物理记录。

## 1. 已冻结的模型付费接口

以下路径相对 `/home/gbk/code/LeWorldModel/le-wm-main/`，本次只读核对三件套SHA，未改字节：

| 文件 | SHA256 |
|---|---|
| experiments/spectral_reachability/agent_p2_six_target_paid_replay_20261029_v1.py | 84ebc96da0f7e351132bf697a90a7eaab2e2aa296646fe62c0013db65d9157d2 |
| tests/test_agent_p2_six_target_paid_replay_20261029_v1.py | 0dd44a082334e2b0fe9ad315293fdd310913b716944982d3032ead4fd0b83876 |
| innerdocs/agent_p2_six_target_paid_replay_20261029_v1.md | 03589d48dc2de4fd8c9ff747d192fe4600f0bb5b7e492ca48f6908cc00c541cf |

已完成six controller authority固定于该模块REFS：prepare cccfadc20911dc1be008a047c57bca7400272f6602deb435fb0341a0c8bbc8f9；report b81f12ee3602835bc14fd2e22d7e3644b8376f7ba8b0b265948bfb18429b6934；receipt 592062bded7d6eb1716594b21fac4b5ef6255cf93d5285c2361f0ec2686c0103。这些不是未来paid authority。

未来物理adapter显式接受**新paid** prepare/report/receipt各绝对路径+实际SHA；domain路径/SHA从paid prepare.domain获取并与receipt.domain强等，不从猜测文件名取得。paid receipt不包含raw数组或物理许可，不能只读receipt作动作来源。

## 2. 实际层级和严格验证接点

令P为新paid prepare，R为report，C为完整receipt，D为独立domain，T_i为slot i trace。以下是现有源码ABI，不是已运行状态：

| 对象 | 真实字段/路径 | 要核的连接 |
|---|---|---|
| P | schema=`agent_p2_six_target_paid_replay_20261029_v1.prepare`；domain、sources、export、expected_state、expected_construction | domain独立文件；完整425/源码闭包，不把P当旧paid含raw_artifact的schema |
| D | schema同stem`.domain`；controller_prepare.slots[0..5]、rows、normalizer、tasks、imported_complete_receipt | 新六controller全部目标/真实历史/task/blind来源，blocked_slot_ids必须为空 |
| R | 同stem`.report`；manifest/manifest_sha256、rows[0..5]、files、write_journal、prediction_seal | 完整六returned、无errors；各slot输入/trace/数组、425前后和source后核 |
| completion | 同目录completion.json，schema同stem`.completion` | report SHA、complete=true、report fsync/hash inclusive1350s及精确boundary；不能只核beforefooter |
| C | 同stem`.receipt`；manifest_sha256、report_sha256、completion_sha256、domain、controller_authorities | 精确指上述实际文件；整份重建强等，不只核status |
| C.local_checks | slot/root/arm、blocked、predict_API/predictor_samples、action_encoder/low_projection/AE_tokens/predictor_context_tokens/projection_rows | 顺序0..5、各槽blocked=null、6/6、6/6/24/18/18 |
| C.denominator/module_resources | total6/returned6/blocked0/missing0；36/36/144/108/108 | 全六屏障，不能拿旧24/96或任何子集充数 |

receipt status为 `all6_actual_deployment_token_replays_verified_not_physically_released`，raw_deployment_authorized/scientific_admission/original_Hi_scaler_certified仍false，physical_approval_required=true。不能把这些false翻为true或猜造其他flags。只有MAIN后续批准新物理协议/driver才构成工程执行授权，不构成P1/P2科学准入。

可供未来MAIN-only物理prepare使用的纯saved接口：`paid.verify(P,PH)`返回**二元组(P,D)**，不是旧paid的单dict；随后 `paid.validate_files(R,folder)`、`paid.check_completion(R,completion,RH)`、`paid.validate_saved(P,D,R,folder)`。这几项本身不写新receipt；另外按paid.consume中实际update字段补manifest/report/completion/domain/controller_authorities/inclusivewall后与外部C整份比较，并校验显式report→P路径/SHA及最终source闭包。`paid.consume(...)`则会排他写receipt，不能当无副作用getter，也不能覆盖MAIN原C。本worker本轮没有执行这些完整验证。

## 3. 每slot实际raw NPY路径，不从mean重算部署动作

固定i=0..5，root=i//3、condition=off、arm按prior_FIRST/program_first_postcondition/arbitrary；两个root各三臂，不压缩/重排。

真实取数链为：

`R.rows[i].trace → T_i.arithmetic[1] → kind=add/output=array_key → R.rows[i].arrays[array_key] → 实际raw NPY`。

必须先核arithmetic完整顺序mul/add/sub/div、第二项确为返回的add，trace/array描述符均在R.files清单，file/file_bytes/file_sha256/root一致且文件在本新paid目录；原NPYv1/header/EOF/data_sha256/dtype `<f4`、shape `[15,2]`、C-order120bytes、signedzero、finite及[-1,1]全部核实。使用 `six.load_array` 的slot0..5 reader，不能调用仅允许slot[0134]的旧controller reader。

raw须与R.rows[i].inputs.sealed_raw实际NPY及D.rows[i].checks.array_bits.raw逐位相同；前者是输入参照而非CUDA执行结果，物理命令以**真实add输出**为来源。mean、CPU raw_candidate、JSON commands不能替代真实raw NPY authority。若物理执行计划另存commands，需同时保raw bits与data SHA及原描述符，反解后120bytes强等，不重新inverse/normalize、clip、tanh、选mean或额外/100。

T_i.actual_tokens引用的是div返回后reshape的 `[1,1,3,10]`，用于paid模型重放验证，不是native动作。R.rows[i].inputs中z/a/target/mean/mean_xy/std/sealed_raw都应按原字段核来源；z/a/target精确绑定D.controller_prepare.slots[i].controller_payload及input_sha256。新slots的来源位于`.source`，不是旧delivery四槽`.origin`，不可盲用旧assemble。

## 4. 原真实root/history/goal映射和首动作guard

D.tasks[root]提供task_file/task_sha256/task_id、warm_raw10_bits、history_sha256、normalization；D.rows[i].task_identity提供task_id/task_sha256/blind_sha256。原program receipt plans[2*root+1]、原physical prepare对应planner_plan及native source_material的source_roots/warm_traces应整份source-bound导入，与六槽task身份逐项相等，不按terminal收益匹配。

低层真实输入仍来自原packet：root.packets.off.rgb_history三帧、raw_action_history十条、goal_rgb；root原native7 goal/goal_pose/root_sha256、renderer及warm trace。必须核其RGB→latent192来源，使z是原三帧context，a是这十条实际warmup raw按原公共stats一次sub/div后的两条chronological token。原task goal192/native goal始终与controller latent target分开，不能把arbitrary target设置成物理goal。

每槽新环境reset/place后实际warm10，保存warmup0..10 off/on/off；三帧warm0/5/10 off及current图、state/fullsnapshot、renderer、goal_pose、raw bits逐项对原authority。warmup原生终止保失败，不pad。随后bind原task goal；原state/snapshot保持应保持的字段，goal_state/goal_rgb按原binding语义变化，不盲等所有pre/post，也不用state-only冒充fullsnapshot。

plan cp0 triplet完成后，在plan step1 **public intent/callback之前**持久化并fsync own initial_history_guard；拒绝或写失败绝无step1 callback。proof至少绑定slot/task/raw SHA/history pixels/action2/原native root与goal/source closure。之后执行这次认证的raw15，无反馈重选；full15或真实早停均保六分母，缺cp保持null，不把terminal补作cp15。

## 5. Reviewed旧低层可复用边界

参考冻结 `agent_p2_commoncontroller_physical_20261029_v1.py` SHA `7e715a9005a4408fbc7899184f338aa4ae6ee6b20dd414356744aa830ceff5f0`。place/bind_goal/native hooks/observe/dispatch/原phase labels及episode的原生执行逻辑可作为显式低层接点；输入root/source_plan/plan/context必须真实适配。

旧 `initial_proof(plan,p,events)`核warm33+plan0三帧、十个实际warm returns、goal_attached、fullsnapshot/history/a；其返回schema带旧NAME。未来own proof如何封装该算法须由Pauli最终协议明确：不能改名后声称旧consumer验证，也不能用旧schema冒充新own proof。旧Meter.context要求(row,plan,p)，且guard须成功持久化后才accepted；这一因果顺序必须保留。

不能直接复用旧paid_complete、material/assemble、check_plan、rows_for、execute、audit_chain或validate_saved为新six验收：它们含旧4/两missing、ACTIVE=(0,1,3,4)、closed<4、旧authority/schema和reset8等固定合同。新six需自有分母/plan/receipt/hashchain和全文件消费，禁止global monkeypatch。旧episode本身无四槽选择，但其全局helpers/schema/context接缝仍须在新协议和RAM生命周期中逐项确认，不据函数名宣称drop-in。

即使raw15等于旧动作，必须6个新constructor/close、12reset、至多150public/1524physics/486PNG，全部新日志/source/guard/原native指标；不能继承旧四row或只跑新arbitrary两槽。旧四物理receipt仅是历史来源/比较，不计新执行量；错误attempt不退款，unknown不补0。

## 6. 当前待定和实施顺序

当前只等待MAIN实际完成new paid domain/CUDA/全consume后提供实际外部SHA，以及Pauli最终六槽物理协议/source-bound输入约定；本说明不臆测这些未来状态或hash。own physical NAME/schema、proof封装、具体prepare CLI、资源cap/partial消费边界由该最终协议与MAIN审批确定，不在这里预造ready或runner。

审批后最小实施才是独立six adapter/RAM全生命周期/guard持久化失败无step1回归；MAIN审读后生产。当前仅此新innerdoc，保存文件字段/来源与接线要求，无新实验、结果选择或科学gate变更。
