# Observed-suffix new physical terminal RGB — own fixed4 HOST CUDA bridge

## 新来源合同，不冒用旧schema

消费已冻结 `agent_observed_suffix_physical_20261028_v1` 新四槽 root0/off、root0/on、root1/off、root1/on。physical source SHA `246739a509c8727bb9381d0c5df3c9649ab6dd9b484ec8f4b8f9bb5891e45a65`；其tests `051178dd1476da87576ff9b3a86644c6f297061878127f76bdaf8b2e3f30b269`、doc `e70127c0f1582a69286e0f1b7014136c2713dd570803c328929e80dcd00fc5c4` 由实际physical prepare闭包继承。本桥不改任何旧模块globals、source、results、MAIN文档，也不加载物理或训练模型。

build先校验CLI真实physical output/report/prepare/receipt/source SHA，再实际调用新physical.consume，完整重建并逐字节canonical比较持久receipt。该consumer内部verify→material→observed compiler v2.consume/full build→midpoint/terminal/原physical source完整闭包；不是status-only或给旧schema换标签。本桥自己是独立 prepare/report/consume schema，只读新physical cp10，不复用旧latent/旧末帧。

各digest按原模块领域分别认证：**compiler.digest（compact JSON）**用于compiler-owned chosen selection、goal、observed_context；**producer.digest = mixed.core.digest（原physical编码）**用于execution plan、plans/reference/cp5 guard proof。两者不能互换。本桥不拿自己的digest代算跨schemaauthority；rawfloat32 SHA/数组hash始终实际bytes。

逐行绑定原task/model/Q/normalizer、observed midpoint context、原prefix5+新suffix5/fullraw10、compiler plan/winner/Q/选择digest、native root7/goal7/goal_pose/同condition goalPNG。新physical实际cp5 guard必须认证完整51prefix帧/返回、原cp5/state/snapshot/PNG exactguard，且位于任何step6 dispatch之前。本桥调用producer.audit_guard并核prefix_match；新analysis row明确输出compiler_selection_digest、observed_context_sha256、prefix_reference_sha256、actual_cp5、actual_cp5_guard、prefix_reference_exact。

cp10须来自新physical report的真实plan10 checkpoint，并与同condition/overlay trajectory/state/snapshot/final_state逐字段一致；snapshot原时间字段完整保留，不 invent新时间。仅genuine native stop缺cp10允许source/latent/actualQ/endpointMSE=null，保留全4分母；不padding、不用cp5替终态、不借旧latent，不把工程遗漏当native stop。若warmup即停止，则仍要求真实physical consumer认证该停止与缺观察，不伪造root-state返回。

## CUDA与保存/strict

每可用新cp10只执行1encoder+1projector，最多4+4，0predict/AE/predictor/pred_proj/AD/newphysics/训练/goal重编码。encoder输入原224RGB→原p0 transform `[1,1,3,224,224]`→flatten `[1,3,224,224]`→hidden `[1,257,192]`（绑定224/patch14 config，拒197）→CLS `[1,192]`→projector→192。model.encode输出通过其实际projector输出逐值绑定。四槽不dedup。

所有原PNG uint8 pixels、原float32预处理、encoder输入/完整hidden、CLS、projector输出与最终192均即时持久化原shape/dtype/数组与文件SHA。run用原p0.image_transform/_image_tensor，同时独立NumPy/PIL逐值/字节复算；saved strict再次读取原PNG复算并验证完整module接线，不能靠“数组内部一致”绕过真实输入。评分保留新compiler predicted_Q、actual_Q、差值、endpointMSE与actual192，使用原goal192/identity Q、float64 MSE；native7/eval结果独立保留，不将latentQ等同native成功或物理收益。

MAIN授权后才导入torch/原兼容loader，HOST CUDA only，eval/frozen、原模型tensor与全部buffers前后hash一致，实际installed class源码必须属于完整绑定清单（继承observed compiler原模型source closure）。MATH SDP、matmul/cudnn TF32 off、autocast off、inference，before/after precision守卫持久匹配。复用冻结encoder-only instance guard/Trace/Store/PNG helpers，不改旧module globals；guard拒predict/AE/AD。完整image/module/guard events顺序、资源attempt/return、native seal、初始四槽、execution binding、全部artifacts及source/report/prepare前后SHA严格核对，不过滤unknown events。

分列新encoder/projector计量、已发生physical资源、observed recompile72forward/32reverse和更早资源；消费自身0neural/physics不代表原数据免费。未测FLOPs=null，不宣称equalFLOPs。预算output64MiB（含final8MiB）/floor200MiB/wall600s；来源metadata cap128MiB沿用physical，报告final8MiB。所有输出新排他路径，无覆盖、删除或自动重试；失败保留partial，无科学负结果冒充。

## MAIN精确命令

MAIN通知新physical实际四槽full10已完成：80public/816physics/264PNG、errors[]、cp5 exact全true、native success0/4；这是MAIN证据，不是worker自行运行。真实report/prepare SHA如下；strict receipt仍待MAIN实际终态提供，不填未来SHA。变量PHYSICAL_PREPARED须是该report.execution_binding.prepared的真实绝对路径，PHYSICAL_RECEIPT/PHYSICAL_RECEIPT_SHA须取实际落盘值。

```bash
M=experiments.spectral_reachability.agent_observed_suffix_terminal_rgb_20261028_v1
RGB_PREPARED="$PWD/results/agent_observed_suffix_terminal_rgb_20261028_v1_prepare_MAIN01.json"
RGB_OUTPUT="$PWD/results/agent_observed_suffix_terminal_rgb_20261028_v1_cuda_MAIN01"
.venv/bin/python -B -m "$M" prepare --output "$RGB_PREPARED" \
  --physical-output "$PWD/results/agent_observed_suffix_physical_20261028_v1_MAIN01" \
  --physical-report-sha 5a05a79996d49a5e90dae7002cc09d07d04291029e74d988e9b18211bdff9a26 \
  --physical-prepared "$PHYSICAL_PREPARED" \
  --physical-prepare-sha 6fb3da89aef4096a80686e4e5facbbcb10d24ca733fc576bffa505c80a8f88a2 \
  --physical-receipt "$PHYSICAL_RECEIPT" --physical-receipt-sha "$PHYSICAL_RECEIPT_SHA" \
  --physical-source-sha 246739a509c8727bb9381d0c5df3c9649ab6dd9b484ec8f4b8f9bb5891e45a65
# RGB_PREPARE_SHA取上述实际stdout，不预填
.venv/bin/python -B -m "$M" verify --prepared "$RGB_PREPARED" --prepare-sha "$RGB_PREPARE_SHA"
.venv/bin/python -B -m "$M" run --prepared "$RGB_PREPARED" --prepare-sha "$RGB_PREPARE_SHA" \
  --output "$RGB_OUTPUT" --authorize-host-cuda
# RGB_REPORT_SHA取真实report.json SHA
.venv/bin/python -B -m "$M" consume --output "$RGB_OUTPUT" --report-sha "$RGB_REPORT_SHA" \
  --receipt "$PWD/results/agent_observed_suffix_terminal_rgb_20261028_v1_consume_MAIN01.json"
```

## RAM测试与未执行项

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_agent_observed_suffix_terminal_rgb_20261028_v1.py \
  tests/test_agent_nonlinear_suffix_terminal_rgb_20261028_v1.py
```

仅RAM metadata/native hooks/数组、模拟env fixture（非真实physics）、临时saved文件、独立冷进程禁止runtime import；测试不是科学证据。首轮51passed/20.30s；新增真实build/verify/receipt/sourceSHA、全saved consume/census与CLI回归，最终实际结果随冻结SHA交付。worker未生产prepare/verify/consume、GPU、neuralCPU、physics或网络。前置为MAIN真实strict receipt、最终审读及CUDA调度；结果unknown。P1/freshness/scientific_admission始终false，不打开保留根、不给物理收益预测。三件套冻结后不再append，后续修复必须新版本。
