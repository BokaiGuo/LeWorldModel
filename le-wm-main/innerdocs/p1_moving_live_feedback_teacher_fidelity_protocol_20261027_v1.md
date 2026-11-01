# Moving live反馈：固定12条自身轨迹teacher-only factual fidelity预注册

状态：仅本独立新doc交付，尚无teacher runtime/prepare/预测/结果。
依据外层 `docs/D2P_MOVING_OBSERVED_FEEDBACK_PROTOCOL_20261027.md` 第6节。
登记发生在live控制结果已知之后、teacher预测尚未运行之前；不冒称live盲预注册。
experiment-design用于固定输入、评价、错误分母和预算，不采用其训练/调参/显著性模板。
不改封存analysis v2、live driver或其它共享文件；MAIN后续审读和实际CUDA执行。

## 1. 唯一问题与固定cohort

问题仅为：原冻结模型在**本episode真实prefix context＋本episode真正执行的suffix动作**下，
对本episode真实terminal successor的factual预测有多大误差？
没有新控制、候选选择、反馈部署、反事实动作、Qfit或成功阈值。

固定12 rows，顺序严格为：(slot0,GN)、(slot0,D2P)、(slot1,GN)、(slot1,D2P)，
每层O/I/R；episode_id分别e00_O/I/R、e01_O/I/R、e02_O/I/R、e03_O/I/R。
prefix_arm为`line_search_GN`或`full10_conditional_D2P`。
独立原任务仍2个，不是12个独立样本；即使图/命令/latent相同也不去重、不退款。
I也使用自己的actual prefix latent做此**离线诊断**，不再使用其部署时imagined prefix；
这不能倒写成I控制实际见过真实latent，不能改变已封存O/I/R解释。

禁止用O的原suffix搭配R的terminal，或用某模式预测搭配别的模式label；
禁止根据已知Q改善、native失败或误差大小筛根/模式、追加seed、换预算。

## 2. 必需外部authority（MAIN显式提供路径，不搜索替代）

固定live report：
`/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_live_feedback_physical_20261027_v1_main01/report.json`
SHA `990843e8bba15955db5163b9b5d77c1f8eecaa1870d3745b49d46cc22ca895e3`。
原prepare SHA `bc3fbc91d7a19ae256b24db22ad05b2526fcbe454e3b55d904136eba2f8fdff5`；
prepare路径由其execution_binding.json解析并与MAIN传入绝对路径精确闭合。
driver source SHA `6742a1b35102e433d7772e4a208b4f97b37b76f815b991cfc0e472763754ace9`。

MAIN已完成的描述性analysis v2：
`/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_live_feedback_analysis_20261027_v2_main01.json`
SHA `5df7dd90b3da010d1bafbb66198334272ec6470b7b0f7bc62991e4b0dae0b5d8`。
它支持24 endpoints完整及配对身份核对，**不替代原NPY/commands/task的直接读取**，
不得把其terminal Q或native结果传给模型。不得调用封存analysis v1的错误state5接口。

原task通过原moving encoder的strict `load_task(path, external_sha)`或等价完整saved校验接入：
slot0 SHA `a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b`；
slot1 SHA `101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74`。
必须是moving native192/target-free/原history、同task_id/原model/原normalizer；
不得用旧zero-only loader、从其它run补图或重编码。

新prepare封存以上文件、原manifest/journal/必要PNG、24 encoding NPY/receipt、实际raw ledger、
task/model文件与source闭包、新runtime/tests/doc及本协议SHA；checkpoint大文件仅记录SHA，
不复制模型和旧report。运行前后复验输入与冻结tensor hash。

## 3. 最小严格saved joins

每条identity恰为 `(episode_id,source_slot,prefix_arm,mode,parent_task_id,parent_selection_sha256)`。
从prepare.layers以 `(source_slot,arm)`唯一join，不按列表位置猜task；
从report.rows按 `(source_slot,prefix_arm,mode)`唯一join，必须完整12且无未知/重复身份。
同层虽共享原plan，实际episode receipt必须各自读取。

1. 外部report/prepare SHA、execution_binding、prepare.schema/固定layers/三模式顺序闭合；
   terminal run无工程错误、12 full10、model_final_check unchanged，journal SHA/hashchain和run_finished闭合。
   prefix_pairing及原source/task/goal权限沿原live合同核对，不以future outcome选择输入。
2. `checkpoints`唯一phase=plan/step=5、10，及其对应fresh_observation/guard_passed，
   PNG file/pixel SHA与本episode各自encode_request的png身份精确一致。
   PNG仅核来源不进新encoder，缺图/缺证据不得补造。
3. `encoding5/encoding10`为driver `.encode_response`；episode/observed_step/request SHA匹配
   根目录的本episode encode request/response文件，CUDA/frozen/source/各一次encoder与projector回执成立。
   NPY分别来自`episode_id_encode5/native192.npy`和`episode_id_encode10/native192.npy`，
   验file/array SHA、float32[192]、有限、无pickle/尾随数据，不借其它episode的相同值顶替。
4. 原warmup实际10步raw从本episode warmup_return的1..10原float32 bits复原；
   原prefix实际raw从control_return step1..5原raw_bits复原，逐条等于prepare选中原prefix。
   实际suffix从同episode `control_return` step6..10的command/raw_bits复原，
   必须等于本episode authorized_dispatch及dispatch_seal.commands/commands_sha256。
   `suffix_returned=5`，每条都有完成receipt；仅dispatch的未执行命令不能当factual action。
   O再核原plan suffix；I/R核自己的suffix_request/response/dispatch seal，不重算或重选suffix。
5. latent5和latent10分别与其own observation/encoding事件相连；
   原live因果链及控制seal在teacher任务开始前全已终结。
   新teacher调用只能读取已固定factual动作，不能发布任何动作或修改原控制结果。

状态/goal/final_state实际均7维；若必要用于来源或native标签核对，严格保7维，
native_distance用全7维，success仍原前4位置＋第5angle，velocity不能丢弃。
但这些state/native/reward/成功/terminal label都不得出现在teacher forward请求中。

## 4. 唯一H1模型输入：真实过去，不含未来label

设原task root_context.latent_history为source绝对步`[z0,z5,z10]`，
W为本episode真实warmup raw10，P为本episode真实prefix raw5，S为真实suffix raw5。
控制相对prefix5=source绝对15，terminal10=source绝对20。
唯一teacher输入为：

```text
emb3 = [原z5, 原z10, 本episodeactual_z15]          # float32 [1,3,192]
past_raw10 = concat(W[5:10], P[0:5])              # float32 [10,2]
history2 = normalize_float32(past_raw10).reshape(2,10)
raw5 = S[0:5]                                   # float32 [5,2]
token = normalize_float32(raw5).reshape(10)
outgoing3 = concat(history2, token)               # float32 [1,3,10]
```

public normalization只作一次原float32 subtract/divide，time-major；
mean=`[-.0087,.0068]`、std=`[.2019,.2002]`以原task完整float32字面值为authority，
不重新round常量、不/100、不tanh/atanh历史、不clip、不zero padding，不拟合；训练provenance仍unknown。
S必须是原实际raw动作，不是normalized encoder token，也不是原candidate logits。

窄forward请求白名单建议仅含schema/identity/input_digest、emb3、history2、raw5及modelbinding身份；
不含goal、actual terminal latent/path、未来RGB、native state/reward/success、witness或其它模式结果。
原goal/Q只在预测封存后的评价域使用，不进入此transition。
预处理/prepare可以封存terminal label的文件路径及SHA元数据，但label数组加载与评价在隔离阶段完成。
所有12个预测输出封存后才启动label评分阶段；运行域与评价域不共享携带label的请求对象。

## 5. 原模型接口与计量

原冻结`pusht_full_packet_compiler_20261027.make_transition`的H1语义为：

```python
# 示意合同，不是新可执行runtime；MAIN后续实现。
successor_all = model.predict(emb3, model.action_encoder(outgoing3))
predicted_terminal = successor_all[:, -1]  # native float32 [1,192]
```

必须原`model.predict`与既有pred_proj路径，不另造线性readout/独立GD或绕过projector。
可复用make_transition的原normalize/transition，或薄独立wrapper严格同语义；
不能调用live compile_suffix（会执行五forward/J/SVD并发布新动作）或suffix verifier凑此预算。
每次真实action_encoder/predictor/pred_proj模块各1，predict API各1，batch1/context3；
保三token完整module输入输出和最后native192，不能把3token当3次forward。

MAIN HOST CUDA-only；一次加载原模型，eval/frozen/requires_grad=false，
inference_mode、MATH SDPA、TF32off/autocastoff，tensor hash前后等于原prepare.modelbinding。
任何CPU神经fallback、encoder/图像projector调用、AD/J/VJP/backward、SVD/solve、训练或第二次forward
均predispatch拒绝并保失败；不是调用后从计数中删除。
torch AD API应有禁止guard，module/API meter保存真实attempt/return及CUDA同步完成receipt。

## 6. 固定新增预算与工程cap

新增恰12 predict API、12 predictor、12 action_encoder、12 pred_proj；
AD/J/SVD/solve=0、encoder/image projector=0、public/physics/render/reset=0。
API与子模块receipt不能相加为24 forward；不偷加nominal/zero/signflip/imagined对照或重复replay。
原live的24编码、40predict及旧source/plan成本单列继承，不能改报本实验免费获得真实标签。

建议未来runtime冻结的独立工程边界：单job全流程wall300s（run入口含source校验/model load/forward/评分），
CUDA allocated≤3GiB/reserved≤4GiB，完整输出≤8MiB＋200MiB磁盘floor，
prepare≤256KiB，普通写保256KiB final reserve；不复制旧大闭包，不缓存新PNG。
这只是预声明cap，不是已经验证实际工作量或性能。
逐dispatch/写入检查，deadline/显存/输出失败不提高cap、不缩分母、不补跑；MAIN若需改cap必须在teacher运行前另行冻结新版本。

全12 rows在第一次预测前durable预置，依次记not_started→attempted→predicted→scored，
异常保first/secondary、原输入证据、完整已发生计数；尝试不退款、禁止retry/替代CPU/替根。
缺自身prefix/rawsuffix/terminal或不完整seal的row保blocked/null，不以其它episode填充。
结构性authority/hash/identity错误拒绝prepare；若因此没有可执行任务，明确no run，不写假完成report。
正常本cohort应有12对完整inputs/labels，实际计数与成功行数仍须独立核对，不能预填cap。

## 7. 唯一评价与解释

各episode输出：`factual_MSE=mean((predicted_terminal-actual_terminal)^2)`（float64算术/native192原值），
原goal下predicted_Q与actual_Q以及二者差；可附`mean((actual_prefix-actual_terminal)^2)`为
纯保存数组变化量，不新增prediction，也不暗设改善阈值。
所有误差逐episode保留；按固定root/prefix层列O/I/R仅描述，不把不同suffix误差当同动作反事实差。
可记录predicted/actual终态与goal的几何差，但不据此判native成功、机制已修复或调控制器。

输出必须独立new schema/report、external input/prepare/source/model SHA、完整12行、
原请求与完整module trace/预测NPY、标签NPY来源receipt、metrics、真实资源及partial错误。
全预测封存后才允许score，metrics不反流任何selector；未知缺项null，输出固定denominator12。
不拟合、不校准阈值、不报告显著性、独立确认或P1/P4科学通过。
native0/12与R-I观测Q改善只属于已完成live结果，不能预设teacher误差大小或科学结论。

## 8. 给MAIN的窄实现建议（未授权本agent实现/运行）

后续若MAIN分配新stem，仅需四个接口：

1. `prepare_saved(external_authorities)`：纯保存join与输入/标签分域封存，固定12行。
2. `predict_one(forward_input, model, device, ledger)`：只做一次上述H1，无label、无env、无动作发布。
3. `seal_predictions(rows)`：确认所有已发生attempt/return、落盘预测/module trace、保partial并封存。
4. `score_saved(prediction_seal, label_bindings, original_goal)`：纯NumPy评分，无模型，再写独立完整/partial report。

MAIN先RAM测试自身轨迹join、I/R互换label拒绝、5/7状态错配、raw/normalized混淆、
未来token污染、第二forward/encoder/AD禁止、缺label/partial计账与故障保存；
完整审读freeze后MAIN才可prepare/实际CUDA。本文不运行任何上述步骤。
