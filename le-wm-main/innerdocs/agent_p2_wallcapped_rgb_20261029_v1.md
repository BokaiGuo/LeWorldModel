# P2 wallcapped：真实checkpoint RGB192 companion

独占NEW source/tests/本文，不改wallcapped planner、physical、预算分析或旧three-program RGB封存。遵循 experiment-code 的固定实验、原始数组、失败保留与回归结构；不扩展绘图/训练/参数探索。worker仅运行NumPy/RAM及已封存JSON/PNG只读ABI检查，无生产prepare、NN/CPU neural/CUDA/physics/下载。MAIN审读冻结后执行。

**这是所有动作已封存并实际执行后的后验表示/任务诊断，不是预注册的新选择方法或原科学假设的事前确认。** MAIN已告知真实stop信息；本桥只据此诚实保留不可观测时间槽，不据结果改变动作、阈值、候选、Q或根。

## 固定12槽、实际最多11图

固定顺序 `(root0,joint),(root0,composition),(root1,composition),(root1,joint)`，每plan cp5/10/15，共12槽、2 known roots、6配对。MAIN已通知前三分支15步、root1 joint在14步native success，故它的cp15不存在。保留该槽原模型预测和modelQ，但source/encoded latent/realQ/native同时间评价均null；不把step14终态顶成step15，不补图，不删除分母。编码max11，每张batch1/no dedup/no retry，0 predict/action_encoder/predictor/pred_proj/AD/新physics/新训练/新选择。

原plan是 `agent_p2_wallcapped_cuda_20261029_v1.plan`，使用 `contexts[0/1/2]` 对应cp5/10/15、`warm_raw10`；**不存在旧edges接口，不构造假edges/旧plan/旧receipt**。整个physical prepare与own strict先完整重建，外部实际报告/receipt/source SHA分别为：

- report `d0708d3105e2b9523d6e90f92fb94c06dfd18f48b4860f78f84c98934b08a691`。
- strict `47ee2abeb1ee5805c47a204efc7afca07e5f98cfca8144591bfd59b995dc376a`。
- physical source `217c27a2f9398ee4909d91afb6c6681599f6a4791ba80253eb3c4bcfb06bb0f4`。

这些作为MAIN实际CLI输入，不用标签定方法。原四动作已在模型封存中确定，native state只用于同时间评价与身份认证，绝不传入encoder输入或planner。现有strict的2known roots不变成fresh根。

## Authority与ABI

`build(physical_output, physical_report_sha, physical_receipt, physical_receipt_sha, physical_source_sha)`：首先核实际source/report/strict字节SHA，再真实调用own physical.consume并全对象比对持久strict；physical.verify完整回溯own wallcapped planner consume/prepare。继续核实际model report/prepare文件SHA处于原bindings，冻结checkpoint policy/model identity一致。

每plan核原compact+LF model digest与spaced physical digest各自编码、完整原task/root/goal/context/warm/raw15；不更换codec。每checkpoint核原action_history归一化/chronology、同step checkpoint/实际off PNG/response/state/snapshot、原native7 reward/native success。goal192直接用原封存float32值，不新编码目标，不改变Q。

复用冻结低层 `bridge.Trace/native.guard/visual_bridge/load_pixels`，而非调用旧RGB高层prepare/run/consumer。native guard临时实例hooks在finally恢复；不monkeypatch旧globals，不伪造旧schema。

## 真正HOST CUDA编码

prepare/verify是saved-only；run须 `--authorize-host-cuda`，授权检查先于输入读取、torch导入或分配。只在run导入torch/模型，加载原compatible pretrained到cuda:0，eval/frozen权重及buffers前后tensor hash相等，TF32 off/autocast off/MATH SDP，原projector完整执行。

每图保存原像素uint8 NPY、原preprocessed float32 NPY、encoder完整input与hidden `[1,257,192]`、CLS projector input/output、最终native float32 `[192]` NPY，均保留NPY字节与数组SHA。独立 `pixels.preprocess_pixels` 精确重算PNG→原224预处理，而非只检预处理张量等于encoder输入。encoded输出先native_seal再saved分析。

资源：真实hook intent/return、禁止API guard、全部event文件逐条对账；expected_calls由固定12槽presence导出且≤11。wall600s、64MiB输出/8MiB final reserve/200MiB floor、metadata128MiB；实际CUDA allocator peak allocated≤3GiB/reserved≤4GiB，逐encode/hook边界同步检查。失败保留原12行、已付attempt/partial数组/errors，不退款、不补图、不重试、不将partial签strict。若在初始化/底层存储写失败不能完整收尾，原已落盘文件仍保留，不能声称完整receipt。

## 严格saved消费和分析

`consume(output, expected_report_sha)`重建完整自身prepare及上游own strict；核新schema、完整artifact集合/每个文件SHA、全部初始roster/native seal、逐事件exact chronology与全部encoder/projector/forbidden guard；unknown event、重封额外未引用artifact、坏预处理、197-token替代、缺array、native/input/latent不一致均拒绝。strict不重新做NN。

每槽原样分列：同时间 `model_Q`、`real_Q`、model−real Q、predicted-vs-encoded MSE、原native metrics/reward/done/物理长度、missing及原图来源。所有Q均float64 identity192 mean squared distance，无新权重/拟合。cp5/10使用对应预测context，不拿full15 terminalQ冒充较早预测。

6pair按root×checkpoint固定，方向 `composition - joint`；双方modelQ/realQ/native metrics与差值全保留。root1 cp15的joint真实标签为null，配对真实差值也null，而两侧模型Q仍可报告；不跨时段补齐。real_pairs_available单列5/固定6，不替换完整主表。模型Q/encodedQ好坏不是native success或机制证明，P1/P2/fresh/physical release/equalFLOPs均不准入。

## MAIN完整CLI

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。prepare后的 `ACTUAL_RGB_PREPARE_SHA`、run后的 `ACTUAL_RGB_REPORT_SHA`仅填实际返回值，不预造。

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p2_wallcapped_rgb_20261029_v1 prepare \
  --physical-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_wallcapped_physical_20261029_v1_MAIN01 \
  --physical-report-sha d0708d3105e2b9523d6e90f92fb94c06dfd18f48b4860f78f84c98934b08a691 \
  --physical-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_wallcapped_physical_20261029_v1_consume_MAIN01.json \
  --physical-receipt-sha 47ee2abeb1ee5805c47a204efc7afca07e5f98cfca8144591bfd59b995dc376a \
  --physical-source-sha 217c27a2f9398ee4909d91afb6c6681599f6a4791ba80253eb3c4bcfb06bb0f4 \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_wallcapped_rgb_20261029_v1_prepare_MAIN01.json

.venv/bin/python -m experiments.spectral_reachability.agent_p2_wallcapped_rgb_20261029_v1 verify \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_wallcapped_rgb_20261029_v1_prepare_MAIN01.json \
  --prepare-sha ACTUAL_RGB_PREPARE_SHA

.venv/bin/python -m experiments.spectral_reachability.agent_p2_wallcapped_rgb_20261029_v1 run \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_wallcapped_rgb_20261029_v1_prepare_MAIN01.json \
  --prepare-sha ACTUAL_RGB_PREPARE_SHA \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_wallcapped_rgb_20261029_v1_cuda_MAIN01 \
  --authorize-host-cuda

.venv/bin/python -m experiments.spectral_reachability.agent_p2_wallcapped_rgb_20261029_v1 consume \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_wallcapped_rgb_20261029_v1_cuda_MAIN01 \
  --report-sha ACTUAL_RGB_REPORT_SHA \
  --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_wallcapped_rgb_20261029_v1_consume_MAIN01.json
```

测试：`.venv/bin/python -m pytest -q tests/test_agent_p2_wallcapped_rgb_20261029_v1.py`，仅RAM/NumPy/临时磁盘及真实已完成JSON/PNG的只读ABI smoke。首次fixture缺task_file失败，已补齐fixture真实字段，未放宽生产validator。最终测试记录随冻结交付。

最终上述命令 **43 passed in 2.74s**。包括真实新Store→NPY/event文件→saved consume、11次编码hook/guard全序列、12槽/6pair及缺项、原像素预处理、197形状/未知事件/输入和资源篡改拒绝、partial失败不继续dispatch，以及实际sealed own report/strict/PNG只读ABI smoke。磁盘完整consumer测试仅将本模块verify边界替换为RAM prepare，不声称RAM对象具有真实工程authority；生产build仍必须完整重建own strict。未做生产prepare或任何模型编码，实际新RGB结果仍待MAIN执行。

冻结核查：原physical source SHA仍 `217c27a2f9398ee4909d91afb6c6681599f6a4791ba80253eb3c4bcfb06bb0f4`；wallcapped CUDA source仍 `5e733e75f56b3dd2f5598d6e9c4ae7878c1e3dd0d613c5b4777c1ecd891b9776`。旧RGB及所有原封存文件未编辑。
