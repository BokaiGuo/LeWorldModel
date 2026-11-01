# 固定双臂 raw normalizer 数值诊断 v1

独占新 source/tests/innerdoc；不修改原 FD、normalizer、模型、Q、gate 或旧封存。worker 不生产 prepare，不导入/执行 torch，不跑神经、CUDA、物理或训练。MAIN 审读冻结后主机执行。

## 研究边界与唯一干预

只比较 public 原 task mean/std 与 archive_sample_stats。后者只取已封存 action-only 报告的 statistics.mean/sample_std，各转一次 float32；不是 checkpoint→training dataset→normalizer 身份证书。即使改善也不认证训练血统、不替换部署统计、不选择 arm/h/root/action。若不改善，也不排除未知训练域，只检验此次两套固定数值对本固定 cohort 的影响。P1/freshness/training_provenance/deployment_allowed/scientific_admission 恒 false。

MAIN 冻结运行前协议：
/home/gbk/code/LeWorldModel/docs/D2P_FIXED_NORMALIZER_ABLATION_PROTOCOL_MAIN_20261028_v1.md
SHA 8f602e5228c7f4de0d5aa693ff669ec0c9e91b8b78e8269f8ffe1b1665689fdc。
prepare/verify 与运行前后 closure 强绑定其实际字节，绝不编辑。

固定输入全部为 results/ 内真实封存：

|对象|文件|SHA|
|---|---|---|
|164 jobs prepare|agent_observed_raw_coordinate_fd_20261028_v1_prepare_MAIN01.json|a1a87327d230af7e02e2379909305c9fbbfb8a05768ba85eb4fcbdac85f95ef5|
|原完整模型 report|agent_observed_raw_coordinate_fd_20261028_v1_cuda_MAIN01/report.json|df21e0cd0a235a6e6f43572cc02614f8c62b4704bac2aabb8ec4e2886104aa15|
|原 full strict|agent_observed_raw_coordinate_fd_20261028_v1_consume_MAIN01.json|c2c454cf51b11e5f28df12e2c2f9ef12c8c7cd1eba7bf99b9a9cf9c97977e2b0|
|action-only archive stats|pusht_d2p_archive_action_statistics_v1_20261026.json|c816c535e08a946a6f7464c659b23e98022b0375359bf5a7add4edd6e018cdda|
|deferred 真实 RGB strict|agent_observed_raw_coordinate_rgb_20261028_v1_consume_MAIN01.json|d6b6ed08732d7595e740f98e0dc2450adf439e726aa3078854badb78f7d8b4b2|

prepare 完整重建原 FD.consume，并逐字节比较原 strict；保留全部164 job 及原预测 reference。installed class source、checkpoint/source/stats/本协议/新三件套均进 forward bindings。真实 RGB receipt 只存固定 path/SHA 作为 deferred evaluation authority，prepare/run 不读取它，不把其 state/outcome 传给 forward。consume 后段才读取用户给定 SHA 的完整 strict receipt 做评价；这是复用已认证 receipt，不伪造或重新签发物理/编码证书。

## 实际 kernel 与 ABI

顺序 public164 后 archive164；每臂 panel0..3，每 panel 原41项：baseline、h(.01,.001)→coordinate0..9→sign(-,+)。原 observed latent3、goal192、identity meanQ、全部 raw5/observed_raw10 不变。没有 selector/gradient/encoder/新 AD。

incoming history 两个 token 都从 job.observed_raw10[10,2] 按 NumPy float32 (raw-mean)/std 重建并 chronological reshape(2,10)，再复制到 CUDA。这与原 history 创建路径一致，不是归一化旧 token。outgoing suffix5 采用原 CUDA float32 ((raw-mean)/std).reshape(10) 表达式，再 cat 为 [1,3,10]。无 /100、clamp、tanh、重复归一化。

只复用原 NativeProgram 的 instance-local observe/call/IO hooks；独立 Backend 显式调用原 model.action_encoder→model.predict→predictor→pred_proj，原 task 不被改写为 archive task，不修改任何 globals。保持原 full3、pred_proj flatten(3,192)、shifted latent3/action2 和原 CUDA meanQ。

public 必须 ALL164 full shifted context 与完整 objective arrays/diagnostic 都精确等于原 report，不是仅4 baseline。prepare 先核重建 public incoming 与原 context 精确一致；每次 public 返回即时做 exact guard，失败终止后续调度、保留原结果和费用。consumer 独立验证完整 IO 链/所有 tokenbits/goal/资源/次序，并再次精确比较 public164。结构失配或 public mismatch 为工程失败，不加容差，不解释成科学负结果。原 mean reduction arithmetic bound 仅复用既有客观浮点算术核验；public 对照本身没有 tolerance。

## 留证、资源与异常

8个独立 native journal（a0_p0..a1_p3），每项 request intent→原生资源/全部 IO→request return；每行保存41个固定索引、job hash、fullcontext、native objective，journal packet 保存 raw10/history/raw5/mean/std/token/outgoing 的 float32 blob+形状/哈希。endpoint192 即 fullcontext 最后一项。无需新造/伪装旧 report schema。

完整新预算 predict/AE/predictor/pred_proj 各328，encoder/AD/physics/training=0。每 panel 独立41 cap，consumer 汇总全328；嵌套计量不加成 FLOPs。冻结模型所有参数/buffer 前后 hash 相同；TF32/autocast off、MATH SDP、no_grad、GPU allocated3GiB/reserved4GiB 前后/每调用检查。wall1800s 含 verify；输出128MiB（16MiB final reserve）、floor200MiB，metadata cap64MiB，独占新路径，无重试/退款。

任一异常保留 failed_partial、已 entered/returned/not_started 的全部41槽及其它7行的 not_started；run errors 和实际 ledger 进入 report。可闭合的 partial trace 消费仍保留每臂164/80分母、未算项 null/missing。损坏 journal/不闭合原生事件/public 不精确则 strict 直接失败，不伪签通过。unknown events、未计量 IO、跨 job 拼接、额外文件或旧值替代均拒。

## post-run 评价

每臂保留164 endpoint误差及四 baseline；每 panel 两个h分别10坐标，合计80对/臂，无最优h/arm。报告 scalar native float32 Q 中央 secant（实际 float32 raw spacing 与名义2h），真实端另列 NumPy float32 Q 与原 RGB float64 Q，不声称 NumPy reduction 是新 CUDA reduction。192维 secant/名义secant、L2/Linf/relative/cosine均逐对保存。same_sign、strict_reversal、both_tie、actual_tie、predicted_tie、missing 分列；真实 exact-zero vector response 与零坐标另列，零范数 cosine=null，不能算同向。真实 native state/physical success 不参与 forward 或评分。

## MAIN 可执行命令

工作目录 /home/gbk/code/LeWorldModel/le-wm-main。source/tests/doc 冻结并审读后：

```bash
.venv/bin/python -m pytest -q tests/test_agent_raw_normalizer_ablation_20261028_v1.py

.venv/bin/python -m experiments.spectral_reachability.agent_raw_normalizer_ablation_20261028_v1 prepare \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_raw_normalizer_ablation_20261028_v1_prepare_MAIN01.json

.venv/bin/python -m experiments.spectral_reachability.agent_raw_normalizer_ablation_20261028_v1 verify \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_raw_normalizer_ablation_20261028_v1_prepare_MAIN01.json \
  --prepare-sha PREPARE_ACTUAL_SHA

.venv/bin/python -m experiments.spectral_reachability.agent_raw_normalizer_ablation_20261028_v1 run \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_raw_normalizer_ablation_20261028_v1_prepare_MAIN01.json \
  --prepare-sha PREPARE_ACTUAL_SHA \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_raw_normalizer_ablation_20261028_v1_cuda_MAIN01 \
  --authorization MAIN_FIXED328_NORMALIZER_ABLATION_HOST_CUDA_ONLY

.venv/bin/python -m experiments.spectral_reachability.agent_raw_normalizer_ablation_20261028_v1 consume \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_raw_normalizer_ablation_20261028_v1_cuda_MAIN01 \
  --report-sha REPORT_ACTUAL_SHA \
  --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_raw_normalizer_ablation_20261028_v1_consume_MAIN01.json
```

不要覆盖任何已有路径；MAIN 实际 prepare/report SHA 必须由新产物取得，无未来 SHA 默认。run 是唯一导入 torch/加载模型入口，必须先通过 MAIN authorization，且无 CPU fallback。worker 未运行上述生产命令。

## 测试说明与当前证据

采用 TDD：首次 RED 为新模块未存在；基础6 passed。扩展 saved arithmetic 测试曾以内部 ndarray 调用只接受 JSON 的 FD.packet，导致1 failed/34 passed；修复内部 float32 latent 校验，不放宽 literal JSON input。随后40 passed。full328 fixture 初次 factory 少传 RAM check 参数导致1 failed/40 passed，修复测试 factory 接缝；不涉及生产 Backend lambda。

测试使用实际封存 metadata/IO 的无模型 replay fixture、手工 token arithmetic、篡改事件/上下文/参数/public late output，以及完整328调度/strict saved consume 和失败无退款分母。Replay 复用旧输出作确定性 fixture，archive fixture 绝非新模型输出或研究证据，不能拿其误差/方向写科学结论。最终本文件测试命令通过：42 passed in 24.13s；MAIN 回报独立复核：42 passed in 23.81s，并已审读 innerdoc 与 native_numpy_Q 修正。模型 source-faithful CUDA kernel 的实际数值验证仍必须由 MAIN 本次真实 run 完成；无 worker CUDA/物理/训练结果。

