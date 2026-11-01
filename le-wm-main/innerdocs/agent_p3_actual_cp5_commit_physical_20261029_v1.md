# Actual-cp5 R FIRST5 原生 commit driver v1

Pasteur 独占新三件套；不改旧源、typed projection、gate 或封存。遵循 MAIN `main_p3_actual_cp5_R_delivery_protocol_20261029_v1.md`，SHA `deed88fd70073c1aaca96d32c81968b7f02a7a3841e3d8a7be00d4e4d573275f`。worker 仅 RAM/native-array mock/临时文件测试，不调用生产 NN/physics/fullstrict。此实现不是执行许可，MAIN 审读后实际验证运行。

## 与 Bohr 的实际接缝

没有可用 agent 消息工具；按 Bohr 共享源码及 innerdoc 的真实 ABI 对接，未另造 planner wrapper。producer 是 `agent_p3_actual_cp5_r_delivery_20261029_v1`；其最后冻结 source SHA、实际 report/prepare/receipt SHA 必须由 MAIN 显式 CLI 提供，本文不猜未来值。

|真实位置|本 driver 用法|
|---|---|
|`consume(output, report_sha, prepared, prepare_sha)`|MAIN prepare/verify 完整调用并与已保存 receipt 全量相等，不仅信 status；无模型/physics 新计算|
|`prepare.tasks[root] / provenance[root]`|rebased cp5 task 与 `original_task` 双身份、raw10/prefix5、真实时刻 5/10/15；不替换旧 control0 R group 的任务|
|`receipt.arms[i].slot/root/horizon`|固定 root0H1/root0H3/root1H1/root1H3；显式整数身份，全部四槽|
|`arms[i].status/complete/model_delivery_verified/no_witness`|必须完整 settled；no_witness 不造 fallback，不构造 env/不派发动作，outcome/checkpoints 留 null/missing|
|`arms[i].delivery.task/base_task/*_sha256/provenance_sha256`|与 cp5 bridge 的实际 task/context 精确连接，显式使用 producer compact digest；own physical plan 保留 native spaced digest，两个 codec 不混用|
|`delivery.raw_edge/commit_raw5/lookahead_raw` 及各 float32 SHA|完整5或15步见证；本次仅取原 prefix5 + commit_raw5，共10个 plan raw；H3 后10步不执行|
|`delivery.verification`、`path`|whole/segmented 实际返回和完整路径精确一致；H1 `[5]`/`[5]`，H3 `[15]`/`[5,10]`；不拿诊断 suffix 拼执行计划|
|`delivery.source_closure_sha256/root_context_role`|完整 producer source closure 与 actual abs5/10/15 角色，不接受 control0 重命名|
|`arms[i].resources`|全27轴原 ledger 原样保留/核 ledger；未知 scalar_backward_calls/FLOPs 保持 null，不用 predict 计数代替全部成本|

prepare 另完整重建已冻结 `agent_p3_physical_commit_authority_bridge_20261029_v1` 输出；要求 actual3frame join=true、planner=null、BLOCKED。它是**历史来源**，不是 R release。合并原 physical strict bindings、history prepare/NPY/三帧 PNG、模型/producer/本 driver/source/constructor 闭包。不会伪装旧 observed_suffix/H3 schema 或修改其 globals。仅显式复用 reviewed native `episode/dispatch/hooks/Store` 原语、prefix 比较和 native label 核验器。

## 真实 causal guard

每个有 witness 的槽独立 fresh env，原 reset/place/warm10，再原 prefix5。`Meter.call` 在 control6 **operation_dispatch_intent 之前**执行并 fsync `cp5_guard`：

1. 完整51个原生 warmup0..10/plan0..5 off/on/off receipt 与原 authority 的 pre/post renderer、goal、native7 state、snapshot、PNG字节 SHA、pixel SHA 精确相等；唯一 cp5 与原参考一致。
2. 当前真实完成的 warmup10 和 prefix5 public operation receipts 连续，逐 float32 raw bits 完全匹配；不是计划自述，也不把 prefix incoming raw10 当完整 commands。
3. 实际 warmup 尾5 + prefix5 组成 causal raw10，与 provenance/bridge 精确相等。按原 mean/std float32 各算一次历史 tokens，与 actual cp5 模型输入逐位相等；无 `/100`、clip、二次 normalization。
4. 当前 warm5/warm10/plan5 三个实际 off RGB/native anchor 与已认证 abs5/10/15 三帧来源精确连接，完整 latent3 context/task 仍是实际模型见证身份。此处不新编码，而是复现原已编码的同字节图像历史。

proof 记录 ref/prefix/cp5、raw10、tokens、model context、local task、bridge SHA。`emit` 写入/fsync 成功后才调用原 Meter 的 intent/native callback。失败写拒绝事件，绝无 control6 及后续 commit intent；journal 自身失败也不得回调。strict 验证 guard 在全部51 observation 和15个 public return 之后、所有 control6..10 intent 之前；拒绝 guard 必须可由 saved mismatch 重现。结束时仍保留 finally prefix 闭包核验。

新 driver 只执行 control6..10。H1 预测终点 control10 可同动作对照；H3 delivery 预测终点 control20 **本次没有对应实际终点**。不得把 cp10 当完整 H3 终点、假造 latent、签 C3/P1/P3 或宣称预算公平。

## 资源、失败和消费

四槽最多4 env、8 reset、80 public、816 internal physics、264 PNG；这是有 witness 四槽全10 raw 时的上限，不伪填 no-witness/early-stop 实际次数。物理资源按实际 nested entry/return 完整计数，不退款，模型成本分栏继承；无新 predict/encoder/AD/训练。原生 native7 reward 包括 velocity；success 为原 jointXY+angle，分别保留，不改阈值。

900s wall、64MiB 总输出、8MiB 收尾 reserve、200MiB floor、128MiB prepare cap；不重犯把完整 metadata 塞进8MiB的错误。计数、wall、hashchain、artifact census 全保存。实际 callback 异常/拒绝 guard 保留四槽及 attempted/completed；原生早停不 padding，cp0/5/10 缺项保留 missing。

`consume` 只核自己的 schema/event whitelist/完整操作栈/PNG/source，拒绝额外/漏失事件、原生标签篡改、未知事件和控制步越界。可消费完整闭合的 `partial_or_failed`，但不冒称成功执行。磁盘或 journal 写入失败若导致 intent/terminal/bookend 不完整，原件保留但**不能签完整 saved consume**；reserve 不保证硬件 IO 故障可恢复。prepared/source/full receipt 缺失或 R partial 不通过 prepare，不能先造 ready。

所有产物 `physical_release/execution_authorized/P1_admission/P3_admission/C3/scientific_admission/freshness=false`，known2/fresh0，不宣称等 FLOPs。CLI 的 MAIN authorization 只允许受控原生执行，不改变这些科学/发布标志。

## MAIN CLI

cwd `/home/gbk/code/LeWorldModel/le-wm-main`；以下变量只能填实际冻结/生产值。bridge 实际 SHA 已知；R SHA 未知不填假值。所有输出新文件，禁止覆盖。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p3_actual_cp5_commit_physical_20261029_v1.py

.venv/bin/python -m experiments.spectral_reachability.agent_p3_actual_cp5_commit_physical_20261029_v1 prepare \
  --delivery-output results/agent_p3_actual_cp5_r_delivery_20261029_v1_cuda_MAIN01 \
  --delivery-report-sha "$P3_R_REPORT_SHA" \
  --delivery-prepared results/agent_p3_actual_cp5_r_delivery_20261029_v1_prepare_MAIN01.json \
  --delivery-prepare-sha "$P3_R_PREPARE_SHA" \
  --delivery-receipt results/agent_p3_actual_cp5_r_delivery_20261029_v1_consume_MAIN01.json \
  --delivery-receipt-sha "$P3_R_RECEIPT_SHA" \
  --delivery-source-sha "$P3_R_SOURCE_SHA" \
  --bridge results/agent_p3_physical_commit_authority_bridge_20261029_v1_MAIN01.json \
  --bridge-sha 9c47743431166ddb2564bf5dea5d5335e794d98ec0873a8b90417e632eefc97d \
  --output results/agent_p3_actual_cp5_commit_physical_20261029_v1_prepare_MAIN01.json

.venv/bin/python -m experiments.spectral_reachability.agent_p3_actual_cp5_commit_physical_20261029_v1 verify \
  --prepared results/agent_p3_actual_cp5_commit_physical_20261029_v1_prepare_MAIN01.json \
  --prepare-sha "$P3_COMMIT_PREPARE_SHA"

.venv/bin/python -m experiments.spectral_reachability.agent_p3_actual_cp5_commit_physical_20261029_v1 run \
  --prepared results/agent_p3_actual_cp5_commit_physical_20261029_v1_prepare_MAIN01.json \
  --prepare-sha "$P3_COMMIT_PREPARE_SHA" \
  --output results/agent_p3_actual_cp5_commit_physical_20261029_v1_MAIN01 \
  --authorization MAIN_P3_ACTUAL_CP5_R_COMMIT5_PHYSICS_ONLY

.venv/bin/python -m experiments.spectral_reachability.agent_p3_actual_cp5_commit_physical_20261029_v1 consume \
  --output results/agent_p3_actual_cp5_commit_physical_20261029_v1_MAIN01 \
  --report-sha "$P3_COMMIT_REPORT_SHA" \
  --receipt results/agent_p3_actual_cp5_commit_physical_20261029_v1_consume_MAIN01.json
```

## RAM 验证记录

首轮40 passed/1 failed，50.31s：action mutation fixture 把原零动作又写成零，未改变字节；修为不同 float32 bits，无 validator 放宽。下一轮44 passed/67.10s，含真实 Store→disk→reader→264 PNG 全生命周期（仅原生 NumPy mock；prepare authority 用明确 fixture seam）。随后对齐 Bohr 新显式 slot/root/source closure 字段，最终回归 **44 passed/66.46s**。测试涵盖无 witness 全4/混合槽、earlystop、cp5 七类 mismatch 零 commit、拒绝 guard partial 消费、事件/action/native label/counter 漏改、journal fsync 故障不回调、不同 digest codec 和无 torch/pymunk/producer import。

冻结源码 SHA `32e485063b98e5da27599e91115e92e3238ec39e761e9860a92de15390cd815d`；测试 SHA `b3503782c83471aff5be19b3fb9eebbd6933f403e454fdb9e00683af1a9048b4`。本文 SHA 由交付消息提供。旧 bridge 三件套 SHA 与上一冻结版相同。

尚未生产 prepare/run/consume，实际 R authority/真实 replay guard/物理结果均待 MAIN 验证。RAM 通过不意味着原生环境通过或物理 release。
