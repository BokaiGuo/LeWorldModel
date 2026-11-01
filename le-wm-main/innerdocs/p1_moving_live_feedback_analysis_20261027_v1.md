# Moving live feedback saved描述分析 v1

本stem仅三新文件：source、对应tests、本innerdoc。不改driver/sealed/shared文件，
不import任何producer，不执行CUDA、物理或CPU神经；NumPy只处理已保存数组。
MAIN审读和实际运行。experiment-code用于固定合同、保存、RAM故障回归；
本任务不采用其训练、下载、绘图、调参模板，也不要求结果为正。

## Authority与确定schema

已只读核对Arendt最终driver source SHA：
`6742a1b35102e433d7772e4a208b4f97b37b76f815b991cfc0e472763754ace9`，
stem `p1_moving_live_feedback_physical_20261027_v1`。
由MAIN转交接口，工具无send_input，未宣称直接发送消息或审计driver。
MAIN报告的actual prepare SHA为
`bc3fbc91d7a19ae256b24db22ad05b2526fcbe454e3b55d904136eba2f8fdff5`；
actual job启动不等于完成，本实现不读取其live输出、不推测report SHA。

入口 `build(report_path, report_sha, prepared, prepare_sha)` 四个显式参数；
CLI均必填，必须actual terminal report，不能猜近似路径或扫描其它run。
接受driver `.report` 的 `live_feedback_calibration_recorded` 或 `partial_or_failed`；
`.startup_failure`/running/其它schema拒绝，不能将尚无report当完成。
`.prepare`通过execution_binding.json和外部prepare SHA绑定；
prepare.bindings必须包含上述driver SHA，并只读核其真实文件字节。
原moving两task SHA固定（task00 a42bfbb1...，task01 101add21...），逐层核
task_export/task_id/source_slot/identity192/原float32 goal/with_target=false。
不是旧zero-only loader，也不重拟合normalizer、投影或Q。

driver report.rows身份为 `(source_slot,prefix_arm,mode)`，不是root_slot/arm。
固定顺序：(0,GN)、(0,D2P)、(1,GN)、(1,D2P)，各O/I/R，episode_id为e00_O至e03_R。
GN=`line_search_GN`，D2P=`full10_conditional_D2P`。固定12 episodes/4配对层/2原任务。
缺row仍生成该slot的missing行；重复row、未知mode/arm/schema/来源拒绝，不能事后子集。

## 观测Q、native与partial

逐row、逐step5/10：从checkpoints的phase=plan/step读取真实checkpoint，
从encoding5/encoding10读取driver `.encode_response` 和latent receipt。
编码回执必须CUDA、原model前后hash相同、source/model true、encoder/projector各1真实completion。
NPY必须位于本次output的`episode_id_encode{step}/native192.npy`，
核file SHA、array bytes SHA、dtype=float32、shape=[192]，allow_pickle=false、有限值、无尾随数据。
拒绝跨episode/step借图借latent；不调用模型补缺。

Q=mean((native192.astype(float64)-原goal.astype(float64))**2)，不求和、不除像素尺度、不normalize。
实际report有actual_Q5/10则要求精确等于相同算术复算值；
driver只在full10末尾写actual_Q字段，故partial的step5若有合法编码仍可独立得到Q5。
缺checkpoint/encoding/NPY分别输出null和缺失原因；即使report原数存在，也不冒充已复核观测Q。
原数另外保留original_report_Q。存在但SHA/shape/值损坏是拒绝，不伪装普通缺测。

native_transition仅在driver row的final_metrics保留最后原生指标，
这里原样保存original_final_metrics、native_success/terminated/truncated/full_horizon及first/secondary errors。
早停final_metrics不冒充terminal10。prefix没有单独native receipt：
endpoints.native_recomputed明确标`recomputed_saved_state_not_native_receipt`，
使用自己的checkpoint.state/goal纯数组复算原native_distance、native_success、object_position_error、
object_pose与wrapped诊断；step10有原final_metrics时要求与复算及final_state精确一致。
不新增阈值：原xy范数<20、原native angle<pi/9仅忠实复算旧定义。
native angle保持原`min(abs(delta),2pi-abs(delta))`无mod缺陷，
negative_native_angle单列；独立wrapped_angle=abs(atan2(sin(delta),cos(delta)))只作诊断，
不得改变原native success或将object_success未知值填为成功。

prefix_pairing必须精确对应自己checkpoint5的state/snapshot/pixel SHA/goal digest与原固定prefix raw bytes SHA。
同row各checkpoint goal不可变化；各O/I/R配对检查initial_exact与完整prefix_pairing相等。
原protocol配对层不成立时，contrast全部null并标原因，而不是仍报matched收益。
这只是保存身份与算术分析，不声称独立证明physical时序或runtime无泄漏。

## suffix模型实耗与描述contrasts

journal.jsonl由report.journal_sha256/chain_tail封存；按原default-spaces JSON digest复核链，
run_started.prepare必须等于提供的完整prepare。operation_id的attempt/return严格对应，
按episode_id累计全部资源，按region=`compiling`累计suffix资源，保attempted/completed和pending。
全journal计数必须等于report.counts且不超prepare cap；缺journal则实际账为null，不以理论数代替。

理论完整suffix预算单列：O全0；每个I/R为predict API/predictor/action_encoder/pred_proj各5，
其中ad_forward1、Jacobian1、SVD1、jacobian API1、内部grad API1、数学VJP directions192。
理论与实际是否相等单独标记，partial失败不退款、不补齐；O的原GN/D2P计划与来源成本是继承已付，
不称免费或全算力公平。predict API/predictor是两个观察层级，不相加为双倍forward。
完整全实验24 encoder/projector、40predict/AE、8J/SVD和1536方向是协议预算，不是本分析器新工作。

每层输出I-O、R-I、R-O，左右两模式actual endpoint Q/native标量逐项相减；
每项仅在对应数据存在且prefix_pairing=true时有数，否则null，绝不只汇总available-case均值。
同时输出Q10-Q5及suffix资源attempted/completed的同向差值（与物理paired前提相同门控）。
负Q/距离差仅表示描述性更低；success布尔差为-1/0/+1，不做显著性、置信区间或独立样本推断。
R-I有匹配编译预算背景，R-O包含额外编译；不冒称普遍因果结论。
未知预测不补算，不将不同suffix的terminal当成同动作factual标签。

## 输出、保存边界与CLI

输出schema `p1_moving_live_feedback_analysis_20261027_v1.report`，
status `saved_descriptive_analysis_not_admitted`；这只表示分析可执行，
不意味着源run无错误或所有24 endpoints存在。
结果保source_status/source_errors、fixed_episode_denominator12、observed_endpoint_count、
全部12 rows、12 contrasts（4层各3项）、census及实际读过的input SHA map、分析器三文件SHA。
P1/P4_admission=false、independent_samples=false、significance=null、new_neural/new_physics=0。

只复验report/prepare/binding/task/NPY/journal及driver identity，
不重hash巨大model闭包、不调用producer.analyze/execute，也不将此当另一个完整runtime代码审计。
输入大小有界；输出NEW独占JSON，cap1MiB、200MiB floor，fsync，不覆盖旧结果。
失败在输出创建前抛错，MAIN保留CLI错误；不产生伪completed报告，不下载、不删、不重试。

cwd `/home/gbk/code/LeWorldModel/le-wm-main`，MAIN待actual terminal且获得report外部SHA后执行：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_moving_live_feedback_analysis_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_live_feedback_analysis_20261027_v1 \
  --report /ABS/results/p1_moving_live_feedback_physical_20261027_v1_main01/report.json \
  --report-sha ACTUAL_REPORT_SHA \
  --prepared /ABS/ACTUAL_DRIVER_PREPARE \
  --prepare-sha bc3fbc91d7a19ae256b24db22ad05b2526fcbe454e3b55d904136eba2f8fdff5 \
  --output /ABS/NEW_analysis_main01.json
```

目前仅RAM mock通过，无实际saved分析结果、无新神经/物理。
goal API仍被已取消的旧offline-fidelity blocked goal占用，新create_goal被API拒绝；
不假标旧goal完成，也不以本实现交付声称整个研究或P1完成。
