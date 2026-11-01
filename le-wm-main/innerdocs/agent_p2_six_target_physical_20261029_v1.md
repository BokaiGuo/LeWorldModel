# Six-target actual-token raw15：own-six native adapter

仅实施MAIN批准协议 `agent_p2_six_target_physical_comparison_protocol_20261029_v1.md`，SHA `6e45721309e9a1fe1107918c988c170daf65ebe032163d0beace73d2bcb118f0`。worker只RAM/fixture和保存文件ABI检查，无实际环境/NN/physics/fullstrict或子代。旧封存不改。experiment-code技能只用于固定循环、失败账和RAM回归，不采用训练、调参、效果选择或图表建议。

## 实际输入和工程许可

源码ACTUAL_PAID固定实际prepare `9f367dbfdf91dc69ff4ca79562868a8bf16074bff55b91699f5894356a1dc5df`、report `c3733a752d1e9b80929edf9726bc0eba2ad1c2ae39b6f05550652ff7b3b74dbd`、完整receipt `6e5fa7c57721cd4765430c7379d501d19978bdb44eebe96dce94f7bc68d9a3db`，均仍由CLI显式传入并强等。domain经prepare.domain真实路径/SHA取得，不猜未来hash。

MAIN prepare/verify/consume调用paid.verify(P,PH)得到(P,D)，核completion inclusive边界、validate_files和validate_saved，再按原paid.consume的真实字段补全并整份比较已有C；不写/覆盖C、不运行producer。将paid全source/model425/原task/全部输出文件、原native recipe及本三件套闭包绑定并最终再次核hash。此认证是MAIN生产步骤，worker没有调用真实material/prepare/verify/consume。

P/D/R/C独立six schema，domain.blocked=[]，6/6returned、36API/36samples、144AEtokens/108predictcontext完整。原raw_deployment_authorized/scientific_admission/Hi_scaler_certified=false保留，physical_approval_required=true不改；它们不阻止MAIN另行明确授予本工程AUTH，也不升级科学门。

每槽只取实际CUDA trace.arithmetic[1]的add output对应原NPY（six reader，原file/data SHA/dtype/shape/header/EOF）；raw120bytes与domain精确、包含signedzero，不用CPU重算或normalized mean替换。后续模型误差只用新paid实际tokens whole/seg预测；本物理driver不计算latent、不重编码RGB、不反馈重规划。

## 新六环境、因果guard与资源

固定0..5两根off×prior_FIRST/program_first_postcondition/arbitrary。所有槽新constructor/reset/place/warm10/bind原goal/raw15/close，绝不继承旧四row或只跑新两槽。target不设成native goal。真实warm0/5/10/current RGB、全snapshot/state、renderer/goal_pose、source/task/blind和一次normalize history2精确认证。

初始proof为自有 `.initial_history_proof`，显式source-bound嵌套未变旧initial_proof算法输出（旧schema只作为低层见证，不伪装新旧report）。guard持久化/fsync完成后才accepted，并且在plan step1 intent/callback之前；写失败或失配绝无plan dispatch。保留phase-goal labels与完整on/off还原，不能state-only代替snapshot，也不能将bind前后goal字段盲等。

复用common_resource.Store.check与Meter的**相同六分支caps/900s资源策略**，原native hooks/observe/dispatch/labels；外层assemble/plan/execute/hashchain/row validator/consume自有six schema，不调用旧fixed4 validate_saved，不patch globals。独立标量来源固定summary source `95eb0fa6da34b7cabe1b1ea64bcba6eba98185081a5767651f973f8d1d6c1562`；原native7、XY、raw/native/wrapped angle与agent velocity并列，不修原angle行为。

满程上限：6constructor/close、12reset、150public、1524physics、24set_state、18goal_setter、12goal_render、6goal_bind、486renders/PNG。128MiB含16MiB终态reserve、floor200MiB、每metadata128MiB；wall900s从verify前起至实际report fsync/hash。completion.json独立绑定report SHA和inclusive elapsed，ack写自身不计该wall但计输出cap。调用before/after hash、完整journal/PNG均保留；早停立即停，缺cp保持null，native terminal不冒cp15。失败不退款、unknown不补0，无法日志闭合只partial不颁完整receipt，不扩预算。

## 固定saved报告

每branch完整public native7/reward/termination/原指标/动作bits/状态/RGB及cp0/5/10/15。consumer核全部事件白名单/嵌套intent-return/hashchain、全PNG文件/像素SHA与sourceclosure。六槽×四cp=24位置，两根×三对×四cp=24pair，missing保null；各指标差值right−left不挑winner，不称fresh/equalFLOPs/独立六根/P1/P2成立。真实latent/RGB encoder需后续独立授权模块，本版不实现。

## MAIN CLI：审读后亲自执行

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_six_target_physical_20261029_v1.py
MOD=experiments.spectral_reachability.agent_p2_six_target_physical_20261029_v1
P=results/agent_p2_six_target_physical_20261029_v1_prepare_MAIN01.json
O=results/agent_p2_six_target_physical_20261029_v1_MAIN01
.venv/bin/python -B -m "$MOD" prepare --output "$P" \
 --replay-manifest /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_six_target_paid_replay_20261029_v1_prepare_MAIN01.json \
 --replay-manifest-sha 9f367dbfdf91dc69ff4ca79562868a8bf16074bff55b91699f5894356a1dc5df \
 --replay-report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_six_target_paid_replay_20261029_v1_cuda_MAIN01/report.json \
 --replay-report-sha c3733a752d1e9b80929edf9726bc0eba2ad1c2ae39b6f05550652ff7b3b74dbd \
 --replay-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_six_target_paid_replay_20261029_v1_consume_MAIN01.json \
 --replay-receipt-sha 6e5fa7c57721cd4765430c7379d501d19978bdb44eebe96dce94f7bc68d9a3db
# PH填上述实际prepare SHA，RH填实际run report SHA，禁止未来占位证书。
.venv/bin/python -B -m "$MOD" verify --prepared "$P" --prepare-sha "$PH"
.venv/bin/python -B -m "$MOD" run --prepared "$P" --prepare-sha "$PH" --output "$O" --authorization MAIN_SIX_TARGET6_FULL15_PHYSICS_ONLY
.venv/bin/python -B -m "$MOD" consume --output "$O" --report-sha "$RH" --receipt results/agent_p2_six_target_physical_20261029_v1_consume_MAIN01.json
```

最终RAM回归：59 passed / 44.08s，exit0（session35812，上述测试命令）。首次worker回归为1 failed/57 passed / 43.94s；唯一失败是测试误将list传给bytes-only的paid.core.digest，已改为对实际compact JSON bytes求SHA，与own spaced JSON codec比较；source pins和验证条件未放宽。MAIN首次独立回归同一测试失败记录为1 failed/57 passed / 43.87s，保留该记录。

本三件套至此冻结。worker未执行真实prepare/run/fullconsume、NN或physics。真实paid/recipe的saved ABI smoke不调用paid.verify/consume或native.source_material，不冒生产证据；生产验收由MAIN按上述CLI执行，当前无生产结论。
