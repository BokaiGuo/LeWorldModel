# P3固定8次actual-history预测：MAIN HOSTCUDA

独占新source/tests/本文。不改compat、resolver、encoding、planner或任何封存结果。
本实现只补8次A下一步预测；原R实际IO仅继承已认证复用，新增R/encoder/projector/
AD/physics/训练/拟合全部0。known2诊断，不签P1/P3/C3，不是物理策略收益。
worker仅RAM算术callbacks与临时保存文件；不生产prepare/NN/physics/strict。

## 实际compat authority

只接受 `main_p3_history_preconditions_compat_20261029_v1.prepare`，全8ready/0blocked，
外部SHA必须MAIN实际 `d51467dabc0fb2625016ff21229f33b04fde93b966e4a5c8e77408d9cef675c8`。
compat源码固定 `f909d207b880d0fc5e4dc23b1f4f378f99b68f932edc6c26a6f4858be6be3ba1`。
拒绝旧resolver schema、status-only、ready子集、未认证R和替代SHA。
新prepare/verify只重建自己的saved接线，不调用compat.build或任何旧consume/verify。
继承MAIN完整encoder/planner消费与R三边原IO的信任边界显式保存，不能冒称重签全上游。

合并compat全部bindings、原planner prepare.sources、encoder prepare.bindings、
本三件套及原生helper/loader/preflight源码；SHA冲突拒绝。
所有源文件、weights/config、原始reports/receipts/PNG/latent/journals按原绑定逐文件
streaming SHA256检查，保存size并前后复查。1MiB块复hash不对132MiB report错误套128MiB
读取cap；读取prepare仍128MiB，原plannerreport不重新解析或扩大其他metadata上限。
原模型policy/weights/config精确绑定，实际加载的每类module源必须在预封存closure。

## 固定8输入/原生ABI

原顺序四branch（s0/endpoints、s0/midpoints、s1/endpoints、s1/midpoints），
每branch start10→15、15→20。原prefix/raw20/winner/task/goal/identityQ不变。
job的task.root_context换成认证A完整3×192 latent/2×10 action history；goal/Q/model/normalizer不变。
原base task先过core.validate_task，原binding单列original_task_binding；新local binding
显式更新context_sha256以及由本schema/job/context产生的task_id/task_sha256/task_contract_sha256，
再过core.validate_task。不能更换context却携带旧context seal，也不冒用原planner的node receipt。
原normalizer mean/std float32，raw5严格原bytes，标准化一次，无clip或投影。
核A/R action history逐位相同、causal raw10/raw20 bits和事前A_predict_input配方。

运行复用封存 `NativeProgram.replay`，每job直接调用一次raw5 helper（不是H3重新规划），
其原 `NativeSignal.observe/call` 保存真实action_encoder/predictor/pred_proj/predict IO。
保留模型native full3输出，仅末行作为端点，按原shift生成下一完整context。
每job helper调用1，共8；没有外层`ledger.call('replay')`，故resource账的
replay_calls轴为0（只计显式该ledger operation），不把helper执行说成没有动作预测。
一次predict对应底层action_encoder/predictor/pred_proj各一次，完整四操作均付费。

每job 10数组事件、19数组、25840有效float32 bytes；8job共206720bytes。
另有64个resource attempt/return事件、16个intent/return及3个seal/finish事件，共163。
resource事件仅metadata，原80 native事件预算不被冒充成全部事件数；每event metadata
64KiB硬限与64MiB整输出cap/final reserve仍全部执行。

## 盲接口/封存顺序

新prepare.jobs不传endpoint labels、R预测或R_proof到神经callback，只含原task/goal、
A观察context、固定raw5和身份。原实际future labels位于pinned compat文件，研究者先前
已经见过结果，不能称研究者盲法；代码不使用它们选动作/root/bank/超参。
运行顺序：binding/initial_jobs→全部8A真实intent/IO/return→同步→model/inventory/
precision after与完整bindings复hash→durable prediction_seal→predictions_sealed事件→
labels_unlocked事件→读取compat标签/R并算分→scores文件→全部复hash→finish/report。
seal含所有8输出、实际资源账与模型前后证据，scores不混入prelabel seal。

原目标float64 meanQ，score调用冻结纯数值decomposition核心分别算A/R MSE、Q与signed/
absolute误差及内积分解；差一律A−R。实际标签相同，误差变小不是物理收益。
不新增R replay，不dedup相等输入，不读取outcome挑配置，无任何可训练参数。

## 守卫/预算/partial

MAIN显式flag才进入run（授权检查先于I/O），HOST CUDA:0 offline、原loader/frozen eval、
所有parameters requires_grad=false、buffers在CUDA；torch.inference_mode、TF32off、
autocastoff/SDPA MATH，实际before/after flags与模型含buffers hash严格一致。
实例forward wrappers复用原封存helper，退出finally恢复，encoder/projector进入即拒绝。
NoAD临时守卫拒绝grad/backward/Tensor.backward、jacrev/jacfwd/hessian及solve/SVD，
finally恢复，进入失败也恢复已安装守卫；不patch旧源码或compat globals。

固定predict API/sample8、context-token24；action_encoder/predictor calls/sample8/token24、
pred_proj calls8/rows24。J/VJP/grad/rawbackward/solve/SVD/encoder/projector/physics全0。
墙钟从run入口含saved verify/load/forward/source复hash/评分；wall600s、输出64MiB、
final reserve8MiB、floor200MiB、metadata128MiB、CUDA allocated3GiB/reserved4GiB。
封存后的正常写入仍受wall；失败final写可绕过wall以保存首因/partial，但绝不绕过
输出cap/floor，不把收尾超时标成complete。report.elapsed_seconds在finally开始取样，
最终durable journal/report fsync为故障保存收尾，不声称其耗时包含在该数值中。
验证在Store创建前失败则没有生产outputdir；Store建立后失败保全部8初始槽、已付费
资源、事件与partial report，无retry/refund。partial不能通过完整consume。

## own saved consumer

prepare/verify/run/consume使用全新schema。完整消费验证外部reportSHA、own prepare重建、
全artifact census/SHA/bytes、链每record与tail、初始roster、prelabel seal、全部163事件
严格次序及每操作ID/返回状态、原始float32 blobs、实际token/act_emb/full3/shifted
context接缝、完整resource账与model/precision/inventory/peak。unknown事件不能过滤。
独立从保存输出重算A/R分数并比较scores；最终再核全部bindings/artifacts/report。
不CPU神经重算，不称新模型native-vs-reference parity；不重复原上游fullstrict。
本轮8A own IO完整验收与继承R的旧权威范围分开。

## MAIN CLI

在 `/home/gbk/code/LeWorldModel/le-wm-main`，使用新独占输出路径。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p3_same_action_actual_history_cuda_20261029_v1.py

.venv/bin/python -m experiments.spectral_reachability.agent_p3_same_action_actual_history_cuda_20261029_v1 prepare \
  --preconditions /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_action_history_preconditions_20261029_v1_prepare_MAIN02.json \
  --preconditions-sha d51467dabc0fb2625016ff21229f33b04fde93b966e4a5c8e77408d9cef675c8 \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_action_actual_history_cuda_20261029_v1_prepare_MAIN01.json

.venv/bin/python -m experiments.spectral_reachability.agent_p3_same_action_actual_history_cuda_20261029_v1 verify \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_action_actual_history_cuda_20261029_v1_prepare_MAIN01.json \
  --prepare-sha "${ACTUAL_A_PREPARE_SHA:?actual generated SHA required}"

.venv/bin/python -m experiments.spectral_reachability.agent_p3_same_action_actual_history_cuda_20261029_v1 run \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_action_actual_history_cuda_20261029_v1_prepare_MAIN01.json \
  --prepare-sha "${ACTUAL_A_PREPARE_SHA:?}" \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_action_actual_history_cuda_20261029_v1_cuda_MAIN01 \
  --authorize-host-cuda

.venv/bin/python -m experiments.spectral_reachability.agent_p3_same_action_actual_history_cuda_20261029_v1 consume \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_action_actual_history_cuda_20261029_v1_cuda_MAIN01 \
  --report-sha "${ACTUAL_A_REPORT_SHA:?actual generated SHA required}" \
  --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_action_actual_history_cuda_20261029_v1_consume_MAIN01.json
```

RAM测试覆盖真正封存NativeSignal.call/observe在非神经module callbacks上的入口与恢复、
完整八槽IO/资源链、history/动作/unknown/splice/label顺序篡改、临时Store单LF/hashchain/
fsync/timeout final、compat schema/SHA/closure篡改、零AD/视觉拒绝、partial不退款。
后续实际prepare/模型加载/8A CUDA/完整own消费全部交MAIN，不把fixture当真实神经证据。
