# MAIN v2：最终真实物理 ABI 修复（保留原v1冻结文件）

原v1只通过mock，绑定了旧physical SHA并使用obsolete operation_attempted/completed事件。MAIN全文审读后创建此v2，不修改原v1、已封存planner或physical。v2使用最终physical SHA、receipt内operation_index/resource/status与嵌套dispatch_intent/returned日志，新增actual producer RAM seam和变异拒绝测试，并在build调用physical.consume完整验证真实物理产物。只修代码接口，不改模型/动作/目标/阈值，不将接口失败当成理论负结果。实际运行、SHA和新测试数由MAIN独立记账。

# Prediction-matched sampling：四计划真实RGB评价

独占此stem的源码、RAM测试与本文。sampling CUDA/physical、overlay evaluator、原source和
模型等旧文件全部只读；MAIN亲自审读、prepare及HOST CUDA运行。实现者不执行实际
prepare、模型或物理，不另派审计agent。按experiment-code技能固定实验/日志/失败边界；
依用户三文件限制，不生成额外训练、绘图或共享文档。

## 输入与固定分母

明确输入新 `moving_prediction_matched_sampling_physical_20261027_v1.report` 的实际路径、
外部report SHA、physical prepare SHA及physical source SHA。没有默认未来report路径/SHA，
无自动发现“最新结果”。目前physical source pin为
`50552d50ce3d97c9a25f938ef2b47e99bc66c88fa1644659611b5bb32843884a`；
sampling producer pin为
`72195df2a7f1d10948c2bf5c494dd26e66400eaa2e269737332cbd876e599d40`。
MAIN随后实际完成的planner报告固定为
`results/moving_prediction_matched_sampling_cuda_20261027_v1_cuda_MAIN01/report.json`，SHA
`60872ad114700731d82d293a616a4dede54a53b7be6514ad0cdf231d838d3ed8`。
physical manifest authority与四条plan均必须引用此报告；实际ownconsume仍由physical.verify
执行saved-only验证。576API/samples/AE/pred_proj/errors[]为MAIN报告，不是作者运行CUDA所得。
若MAIN最终冻结SHA不同，本版本明确拒绝，不能静默放宽authority。

固定root-major四行：root0 CEM/nominal-MPPI、root1 CEM/nominal-MPPI。不能从真实效果中
筛算法、筛根或减少分母。每行自身真实plan checkpoint0/5/10完整且只对应自己的selected
commands/selection_digest。缺图、earlystop缺endpoint、engineering failure、SHA/join不齐
均在prepare前fail closed，不补图、不复制末态、不填零或减少分母。

本版复用Hegel evaluator的无状态array/NPZ/PNG/config-shape校验与Artifacts，并用自有
build/assemble/evaluate/run/consume及有界Trace，不调用它的overlay8 build/run/contract，
不monkeypatch其globals，不把off/ontarget28图搬过来。其冻结source SHA为
`70c03508d775b8b488c12133da929bae607d08d6163d5e1c6578edd9a2422bcc`。

## Goal来源去重与实际调用

每个root的goal读取原sourcepack中该slot的blind NPZ **实际goal_rgb**，绝不重新render、
插goal_pose或用默认轮廓替代。通过physical自己的saved `verify`重建source/ownconsume/
manifest/全部bindings，绑定原task export、task contract、blind SHA、root+goal identity。
planner消费只能是新sampling自己的consume，不重标旧sampling结果；它不调用模型。

goal只按 `(原NPZ绝对path, NPZ文件SHA, goal像素SHA)` 相同的真实来源去重；同源goal在
两算法之间共享一次编码。不同source即使goal像素相同，也不凭像素合并。正常原两根是
**2 goal + 4×3 checkpoint =14张，14encoder/14projector**；prepare保存每个goal的
source_slots与每行goal_image，真实calls也从模块entry/return统计。合同支持实际共享一个
来源时13张，绝不凭空添加调用。12 checkpoint即使同像素也均独立编码，不能减少调用。

原moving history严格float32 raw(10,2)，前5步[.1,0]、后5步[0,.1]。零、时序反转、缩放
或float64均拒绝。验证NPZ完整RGB3/raw哈希、实际warmup10 bits与旧task；不是zero-only
loader，不再次除100。physical所有22帧/branch的文件SHA与88帧journal/guards均绑定，
选中的12帧另外严格核PNG/pixel/state/snapshot/returned own command对应。

## 模型与模块原始证据

模型weights/config SHA与tensor SHA沿原冻结模型。当前config为image_size224/patch14，
由绑定config计算 `(224/14)^2+1=257` hidden tokens，非197。保存原PNG/NPZ bytes、原
RGB uint8、实际transform `[1,1,3,224,224]`、encoder input `[1,3,224,224]`、**完整hidden
`[1,257,192]`**、真实CLS/projector input `[1,192]`、projector output `[1,192]`及native
latent `[192]`。纯saved逐位校验transform reshape、hidden[:,0]、projector到latent链。

HOSTCUDA-only，原 `image_transform/_image_tensor/model.encode`，MATH SDP、float32、
TF32off、autocastoff、eval/requires_gradFalse/inference_mode。禁止predict、predictor、
action_encoder、pred_proj及AD API；不训练、不拟合、不Qfit、不新控制/physics，不CPU
neural fallback。NumPy/PIL仅用于saved解析与算术，非CPU模型。

使用冻结native module guard，限定它只在本实例的授权encode期间临时封住禁用入口。
Trace保存dispatch前durable intent、实际callback entry、同步返回与完整inputs/outputs；
intent持久化失败不冒充模型调用；实际callback返回后若保存output失败，保留真实completed
计数，但工程状态失败、Q全部null。失败不重试。restore在最终model/source verify和seal/Q
之前完成，restore异常也不能释放指标。

## Seal、Q与消费

先收齐全部14张（或实际来源导出的13张）编码，关闭guard、复核冻结模型与全部输入SHA，
再durable `encoding_seal.json`，之后才进行所有Q算术和保存metrics。Q为自身实际goal
与checkpoint native192的float64平方差均值；不混sum192，不换goal、不改阈值、不使用state
参与神经编码或计划选择。报告保留原physical native initial/final metrics，不宣称native
成功。任一编码/封存前模型或源验证失败，完整四行保留、所有Q=null、不产metrics；保留
未开始image/partial及first/secondary错误列表。底层磁盘不可写时只能尽力保存已有证据，
不承诺必能写最后report，也不删除资产重试。

`consume(output, external_report_sha)`只做saved/CPU数组，不调用模型。校验自己的外部
prepare及整链、完整artifact人口、module intent/return、NPY/pixel/PNG/NPZ原字节、完整
257token链、encoding seal与metrics时序关系，重算全部12个checkpoint Q，返回：

```text
schema: moving_prediction_matched_sampling_rgb_cuda_20261027_v2.consume
rows: 固定4条 root_slot/arm/selection_digest/task_id/goal_image/physical_status/
      initial_metrics/final_metrics/checkpoints
checkpoints: 固定0,5,10各含 image, latent{file,sha256,array_sha256,shape,dtype},Q,status
counts: encoder/projector实际attempted/completed
P1_admission:false; new_independent_samples:0
```

## 工程预算与MAIN命令

CAP32MiB包含原图、完整hidden/transform/模块raw NPY.gz、receipt、seal、partial、metrics
和report；普通write预留2MiB供final。启动要求32MiB+200MiB可用空间，逐write保持200MiB
floor。prepare≤2MiB、NEW独占文件与output，禁止覆盖。600s从verify前开始，在
verify/load/encode/finalverify检查，不能抢占已进入的阻塞CUDA/系统调用；超时仅允许有限
failure/final持久化，无新神经或Q。规划576predict/AE等历史费用、source/physical成本
单列继承，不能加到本次14图调用或退款；同predict预算不意味着等FLOPs或整个管线等成本。

在 `/home/gbk/code/LeWorldModel/le-wm-main`，MAIN填实际external SHA；以下命令本次未执行：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_moving_prediction_matched_sampling_rgb_cuda_20261027_v2.py

.venv/bin/python -B -m experiments.spectral_reachability.moving_prediction_matched_sampling_rgb_cuda_20261027_v2 prepare --physical-report ACTUAL_ABSOLUTE_REPORT --physical-sha ACTUAL_REPORT_SHA --physical-prepare-sha ACTUAL_PHYSICAL_PREPARE_SHA --physical-source-sha 50552d50ce3d97c9a25f938ef2b47e99bc66c88fa1644659611b5bb32843884a --output results/moving_prediction_matched_sampling_rgb_cuda_20261027_v2_prepare01.json

.venv/bin/python -B -m experiments.spectral_reachability.moving_prediction_matched_sampling_rgb_cuda_20261027_v2 verify --prepared results/moving_prediction_matched_sampling_rgb_cuda_20261027_v2_prepare01.json --prepare-sha ACTUAL_RGB_PREPARE_SHA

.venv/bin/python -B -m experiments.spectral_reachability.moving_prediction_matched_sampling_rgb_cuda_20261027_v2 run --prepared results/moving_prediction_matched_sampling_rgb_cuda_20261027_v2_prepare01.json --prepare-sha ACTUAL_RGB_PREPARE_SHA --authorization MAIN_PREDICTION_MATCHED_SAMPLING_RGB_V2_CUDA_ONLY --output results/moving_prediction_matched_sampling_rgb_cuda_20261027_v2_cuda_main01

.venv/bin/python -B -m experiments.spectral_reachability.moving_prediction_matched_sampling_rgb_cuda_20261027_v2 consume --output results/moving_prediction_matched_sampling_rgb_cuda_20261027_v2_cuda_main01 --report-sha ACTUAL_RGB_REPORT_SHA
```

RAM/mock测试只验证工程schema/算术，注入cuda_verified=true的schema fixture不是真实CUDA
证据。本实现没有实际physical/神经结果，未来MAIN运行仍只是knownroot校准，不是泛化、
独立样本、P1准入或任务成功证明。goal API仍由被取消但unfinished/blocked的旧审阅目标
占用；本实现没有假标其完成来建新goal，也不把此工程交付说成整个研究完成。

本轮验收38项RAM测试，包括14图/四分母/goal来源共享、非零history与dtype/尺度/时序反例、
完整257链、seal先于Q、pure saved消费、模型/源/restore失败时Q未知、真实callback与
intent/output/returned持久化故障的attempt-return区别、严格artifact集合、wall/cap/floor。
测试没有创建磁盘tmp/prepare/result，没有神经调用。
