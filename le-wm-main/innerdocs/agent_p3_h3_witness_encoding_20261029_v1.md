# P3 H3 same-action witness：四分支真实RGB编码

仅本stem source/tests/本文；不改physical/shared-panel/旧编码工具或封存结果。使用experiment-code组织实现与RAM回归，按任务约束不执行训练、绘图、production prepare/fullstrict、NN或物理。实际CUDA由MAIN在源码审读后执行。技能旧reference路径不可用，遵循本仓库冻结原生tensor与saved ABI。

## 权威与目的

四个固定known2 × endpoints/midpoints分支。物理阶段执行原prefix5 + adaptive b5原H3 winner完整15raw，实际control20；不是原MPC在b10/15换动作的终态。只做cp5/10/15/20的HOST CUDA encoder/projector，最多16次各batch1，不去重相等PNG、不重新渲染、不选winner、不调整动作/Q/goal/normalizer。

MAIN已给实际物理report SHA `c8fa06be8eaaf6f7515e6e9550c900c71e04f2141d745699b33d1585eb21a2ce`，完整consume SHA `48a1a5ecadf7043e3162ca104d8f43b0d59baf88655a076f675c9f96d697b9f7`，physical prepare SHA `0e31243a683a7383066ea3b495840610a4b1f2df811fcae318764d3b5acbac4e`。源码不预造未来encoder report/receipt SHA，也不把它们写成已执行结果。

prepare必填physical output/reportSHA/receipt/receiptSHA/sourceSHA，先核真实bytes与冻结physical源码`f06772639f5477d9bd9bbe962a7b37ece34878226c30b0b43a0efa3a30eb7fe0`，再调用其完整`consume`并与持久化receipt canonical bytes完全比较。仅接受自己的4branch physical schema与完整recorded状态；不是status-only，也不冒用旧P2 receipt。producer.consume已完整重建其prepare和shared-panel strict；随后对其bindings中已核验的physical/shared-panel prepare做SHA绑定读取，不再无谓额外调用相同producer.verify。new verify仍完整重做new build。

完整bindings继承physical strict的全部模型/原任务/PNG/上游源码与installed inventory，再合并本三件套和实际使用的原生编码/预处理/trace/guard工具及checkpoint文件。run/consume前后重hash；原model hash来自shared-panel prepare，原task/Q/normalizer均逐分支比对。prepare不加载模型；worker不运行生产build或strict。

## 对齐语义与null

|真实control时间|模型侧参照|结果字段|
|---|---|---|
|5|原b5 request实际观察latent|observed_anchor_reencode_MSE；必须exact重编码一致，不叫预测误差|
|10|原H3 winner.whole[0]最后native192|prediction_vs_encoded_MSE、原goal Q误差|
|15|原H3 winner.whole[1]最后native192|同上|
|20|原H3 winner.whole[2]最后native192|同上；与原winner terminal cost复核|

不比较native预测3行与shifted context的错误时间语义，不拿cp5 observed当forecast，不把早停终态放在缺失cp。`fixed_rows`核原selected H3、原raw20=prefix5+15suffix、whole路径每边context/action历史闭合、原identity192 projection与terminal原Q。原请求raw10必须逐位等于warmup tail+已执行prefix5。物理cp5 state/snapshot/pixel须与原request观测对应的cp5完全一致；10/15/20严格逐时间checkpoint/trajectory/response联接，native7包含速度，goal/reward/成功门不改。

固定16行，即使早停。不可用PNG没有encoder调用，source/latent/actual_Q/误差/native metric保持null；预测存在不代表有真实标签，不能输出零误差。不同时间的model_Q可作描述性数值，不能称cp10为原terminalQ。完整真实physical未来outcome不进入任何选择路径。

编码与原P3相同：RGB224 → 原preflight图像变换 → `(1,1,3,224,224)` → encoder输入`(1,3,224,224)` → 完整hidden`(1,257,192)` → CLS`(1,192)` → 原projector → native192。没有额外L2归一化、白化、拟合或goal重编码。保存原PNG pixels、preprocessed、encoder完整输入/输出、projector完整输入/输出、latent与每个array SHA/shape/dtype。strict独立重算原像素预处理和CLS/projector接线，强制全float32。

saved consumer同时派生`actual_context_chains[4]`：从原b5 observed完整context出发，cp5只exact重编码核对；cp10/15/20用实际RGB latent逐帧shift，并用实际causal raw10按原float32 mean/std只减除一次。缺任何必需实际latent后完整context为null，不用预测latent补位。这是观察历史重建，不额外调用predict，也不是新反馈策略。

## runtime/strict守卫与预算

仅MAIN `--authorize-host-cuda`后进入run。原checkpoint loader，CUDA:0、eval、所有模型parameters禁梯度、buffers在CUDA。源码class inventory每个实际module类路径必须在预封存bindings中且SHA匹配，before/after清单一致；原state_dict（含buffers）hash前后等于expected_model_hash。无CPU模型fallback，无网络下载；使用offline环境变量。

原生Trace保存真实module IO，独立native guard保存encoder/projector intent/return；predict/action_encoder/predictor/pred_proj及AD API均禁止。所有image/module/guard事件按完整生命周期exact顺序验收，未知event不filter。每个available图像独立计费；missing保留槽但不补调用。运行失败保留初始roster、已写arrays/events和partial report，不重试。

|资源/边界|事前固定上限|
|---|---:|
|计划/检查点槽|4 / 16|
|encode API / encoder calls/samples/frames|16 / 16 / 16 / 16|
|projector calls/samples|16 / 16|
|新predict/action encoder/predictor/pred_proj/AD/physics/训练|全部0|
|wall（从run入口，包含输入verify）|600秒|
|输出/终态reserve/disk floor|64MiB / 8MiB / 200MiB|
|metadata cap|128MiB|
|CUDA allocated/reserved|3GiB / 4GiB|

实际attempted/completed逐独立轴报告；不是FLOPs相同，inclusive wall不相加，FLOPs与未独立计量的hidden engine成本不能编造。TF32两类off、autocast off、SDPA MATH，运行前后真实flags完全验收；每个callback检查wall/disk/CUDA上限，记录peak allocated/reserved。输出native seal先于消费者算误差；saved strict完整文件census、每文件SHA、binding/初始roster、原生seal、真实数组、dual ledger、模型/precision/memory证据和完整source闭包都必须通过。partial只能保留，不签完整结果。

own receipt：`rows[16]`含actual192/goal192/predicted192、same-time model_Q/actual_encoded_Q/有符号及绝对Q误差/endpoint MSE；cp5有单独observed重编码字段。另含`actual_context_chains[4]`、实际模块counts、physical report/receipt SHA、继承原planner/physical资源各轴。所有P1_admission/P3_admission/C3/physical_release为false，known2不是fresh4；不声称同动作预测正确、控制收益、泛化或科学准入。

## MAIN完整命令

在`/home/gbk/code/LeWorldModel/le-wm-main`。P、R分别由MAIN从本轮实际prepare/run返回取得，不能预填未来hash；所有写出路径须新建且本stem前缀。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p3_h3_witness_encoding_20261029_v1.py

.venv/bin/python -m experiments.spectral_reachability.agent_p3_h3_witness_encoding_20261029_v1 prepare \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_h3_witness_encoding_20261029_v1_prepare_MAIN01.json \
  --physical-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_h3_witness_physical_20261029_v1_MAIN01 \
  --physical-report-sha c8fa06be8eaaf6f7515e6e9550c900c71e04f2141d745699b33d1585eb21a2ce \
  --physical-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_h3_witness_physical_20261029_v1_consume_MAIN01.json \
  --physical-receipt-sha 48a1a5ecadf7043e3162ca104d8f43b0d59baf88655a076f675c9f96d697b9f7 \
  --physical-source-sha f06772639f5477d9bd9bbe962a7b37ece34878226c30b0b43a0efa3a30eb7fe0

.venv/bin/python -m experiments.spectral_reachability.agent_p3_h3_witness_encoding_20261029_v1 verify \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_h3_witness_encoding_20261029_v1_prepare_MAIN01.json --prepare-sha "$P"

.venv/bin/python -m experiments.spectral_reachability.agent_p3_h3_witness_encoding_20261029_v1 run \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_h3_witness_encoding_20261029_v1_prepare_MAIN01.json --prepare-sha "$P" \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_h3_witness_encoding_20261029_v1_cuda_MAIN01 --authorize-host-cuda

.venv/bin/python -m experiments.spectral_reachability.agent_p3_h3_witness_encoding_20261029_v1 consume \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_h3_witness_encoding_20261029_v1_cuda_MAIN01 --report-sha "$R" \
  --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_h3_witness_encoding_20261029_v1_consume_MAIN01.json
```

测试仅合成数组/非神经Module callbacks、临时磁盘Store/arrays/event与自己schema的saved consumer。覆盖all16与missing/earlystop、H3路径/原目标/normalizer/causal raw10、hidden197拒绝、禁止predict/AD、完整事件顺序、pixel/CLS/projector/latent/模型/precision/memory/文件census变异。夹具复用原测试的两个root原本有不同假model hash，已统一为同一合成checkpoint以满足真实原模型单一性；未修改生产或旧fixture。

最终RAM/临时磁盘测试46 passed（10.65s），冷CLI --help通过。另只读MAIN实际physical report/receipt及原shared-panel prepare，纯fixed_rows ABI联接确认16/16 checkpoint可接；没有调用任何生产consume/fullstrict或写生产prepare，这不认证新的编码结果。三SHA随冻结交付。worker不执行真实physical receipt.consume、production prepare或CUDA；MAIN尚需实际审读、prepare/verify、HOST CUDA和完整saved strict。旧physical三件套保持冻结。
