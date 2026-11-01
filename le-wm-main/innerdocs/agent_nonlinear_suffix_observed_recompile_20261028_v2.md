# 实际 cp5 RGB 上的 suffix 重编译 v2

独占生产 adapter；冻结 v1 numerical core 原样保留。MAIN 审读、主机 CUDA 执行；本工作位不执行神经/物理。4 槽是原 2 known roots × off/on，不是四个独立根；非 fresh，不放行 P1/物理/科学结论。

## 固定算法与边界

原 nonlinear library v2 全 168 候选严格消费后的 4 selected plans，前 5 个 rawfloat32 动作已经真实执行。只重编后 5 个动作；保留原 goal192、identity mean Q、模型、normalizer、尺度、cap 与 prefix bits。输入不是 imagined prefix context：使用 Bohr 冻结 midpoint 的 actual_full3（warmup5/root0/cp5），incoming2 为 warmup6..10 和已执行 prefix1..5，各自按原 float32 mean/std 归一化一次。

从每条原 selected suffix 的保存 logits/rawbits 开始，不做 inverse tanh 或 iterate0 修补。迭代 8 次、eta=1，action scalar gradient 经 tanh chain 一次、沿冻结 inward projection 更新；在 incumbent + 全部 8 iterates 的实际 nonlinear terminal meanQ 上 stable(Q,index) 选择。与冻结 library 的 float32 raw chart 相同，不声称与更早 float64 tanh 候选生成逐位一致。原 float64 ridge solver 合法、cast 后 norm 略超 cap 的 incumbent 仍由原 solver/cast witness 认证；postcast norm 仅诊断，不加 epsilon、不静默投影。原 selected projected iterate 则完整核原 9/8 recurrence。

**预算事前固定 72 predict / 32 scalar reverse**：每槽 9 score forward + 8 AD forward + 1 独立 winner replay = 18/8；原 v1 core 的 68/32 数值成本留在 nested numeric 结果中，新 adapter 明示额外 4 winner replay。不新增 encoder、goal encode、prefix prediction、Jacobian/SVD、训练或物理。原费用与 midpoint 4 encode/4 predict 另栏继承；FLOPs unknown，不称等 FLOPs。72 次 action_encoder/predict/predictor/pred_proj、216 temporal rows，32 grad/VJP/raw callbacks，完整原 resource axes 都按实际 entry/return 记录。

## 权威和实际 ABI

生产 build 调用 `midpoint.consume(output, reportSHA)` 完整重建，逐字节比较 persisted receipt；之后才用公共 `load_saved_array` 读取 actual_full3、全部 feedback inputs、native3 prediction。只取 native 输出末行构成下一个 shifted context。独立重播 iterate0 必须逐位复现 Bohr 的同 suffix actual-mid prediction/Q，独立 winner replay 也必须精确等于选中 context；不把前两行 native prediction 当未来历史。

准备文件绑定 Bohr prepare、report、receipt、全 artifacts、全部 inherited bindings；从 `physical_authority` 接原 library v2 prepare/report/receipt，核完整 168、原 selected prefix/iterate/selection digest/full10/goal/model-task。优化器仅接 whitelisted model context/raw/goal，不接 state/reward/native metrics。frame state 仍只留上游证据，不入新 model request。自身 source/tests/doc 和实际 installed 类源码也入 SHA 闭包；verify 每次完整重建，没有伪称只读 receipt 为重新执行。

冻结依赖：numeric v1 source `231bf17d98f4e5dcb61845a8bff4bed1ebbc108b58f6fbeddb722c340ce5b1df`；library v2 source `0a3e475af9271a0769a3829b4e714fa4e2d6f00eaed38ec8c3ea5d7b9a163fbc`；midpoint source `b1371a0de4d805bac1b4b64cfb625b7d327a3772d61d68c092804430f94a7ece`。

MAIN 已有真实 midpoint 输入：prepare SHA `6dd4ef991ebf2947388ae7c139a23776592b200e11032fdc106be281ae9cc5e8`；report SHA `1639745d33424a28e39a36924b4fe161a00aed2f5a6ae035352c898a9c8e88a0`；receipt SHA `127c650eace2c41e43dc0361aa13009981acfc8e9509dfff7a1f924283f87521`。本 adapter 参数接受实际 authority，不预填未来自身产物 SHA。

## 实际 ledger 与 strict

复用冻结 F.Backend，仅实例方法显式转译 observed_ label 前缀，不 patch globals。每槽一个真实 NativeJournal，compile intent → 数值 intent/全 native start/resource/IO/return/end/AD → 全九 score 选定 → independent winner replay → compile return。strict 同时重算数值算法和原生完整 module/数组/资源账，拒绝未知事件/字段、跨 phase splice、AD 标量伪装和漏操作。runtime loss 必须 0d；原 codec 将其传为 shape[1]，仅此 shape 接受；raw leaf/gradient/callback 严格 [5,2]。bool 不可作为 numeric、slot/index/Q/cap，所有数值必须有限。

prepare 任一 actual history 缺失保留 4 槽且 production_ready=false；run 全槽拒绝，不补图或根。dispatch failure 保留已用 ledger、failed_partial、未到达槽 null，不重试。consume 重建整张 prepare、输出 census/SHA、journal chain、实际数组/操作和全部新 plan；partial 不输出可部署 plans。plan 中 parent selection/result/candidate 均单列；`new_suffix_Q`、`new_terminal_context`、`observed_context_sha256`、`new_witness_sha256` 与新动作密封，旧 parent prediction 不作为新 claim。

CUDA-only、原 checkpoint hash、eval/frozen；MATH SDP、TF32/autocast off。1800s 从 verify 前起计（边界 watchdog，不是抢占式 kernel timeout）；output128MiB、final reserve16MiB、磁盘 floor200MiB、CUDA allocated3GiB/reserved4GiB。失败使用 finalizing 保留 final reserve。前后重核模型 parameters/persistent/nonpersistent buffers 和 bindings，未变才 complete。

## MAIN 命令

以下在 `/home/gbk/code/LeWorldModel/le-wm-main`。只有 MAIN 可执行 run；先审读/回归，冻结后 prepare。`PREPARE_SHA` 和 `REPORT_SHA` 是相应实际输出的 SHA，不是字面占位 authority。

```bash
.venv/bin/python -m pytest -q tests/test_agent_nonlinear_suffix_observed_recompile_20261028_v2.py tests/test_agent_nonlinear_suffix_observed_recompile_20261028_v1.py tests/test_agent_p1_nonlinear_suffix_library_20261028_v2.py

.venv/bin/python -m experiments.spectral_reachability.agent_nonlinear_suffix_observed_recompile_20261028_v2 prepare \
  --midpoint-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_nonlinear_suffix_midpoint_rgb_20261028_v1_cuda_MAIN01 \
  --midpoint-report-sha 1639745d33424a28e39a36924b4fe161a00aed2f5a6ae035352c898a9c8e88a0 \
  --midpoint-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_nonlinear_suffix_midpoint_rgb_20261028_v1_consume_MAIN01.json \
  --midpoint-receipt-sha 127c650eace2c41e43dc0361aa13009981acfc8e9509dfff7a1f924283f87521 \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_nonlinear_suffix_observed_recompile_20261028_v2_prepare_MAIN01.json

.venv/bin/python -m experiments.spectral_reachability.agent_nonlinear_suffix_observed_recompile_20261028_v2 verify \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_nonlinear_suffix_observed_recompile_20261028_v2_prepare_MAIN01.json --prepare-sha PREPARE_SHA

.venv/bin/python -m experiments.spectral_reachability.agent_nonlinear_suffix_observed_recompile_20261028_v2 run \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_nonlinear_suffix_observed_recompile_20261028_v2_prepare_MAIN01.json --prepare-sha PREPARE_SHA \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_nonlinear_suffix_observed_recompile_20261028_v2_cuda_MAIN01 \
  --authorization MAIN_OBSERVED_SUFFIX4_RECOMPILE72_HOST_CUDA_ONLY

.venv/bin/python -m experiments.spectral_reachability.agent_nonlinear_suffix_observed_recompile_20261028_v2 consume \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_nonlinear_suffix_observed_recompile_20261028_v2_cuda_MAIN01 --report-sha REPORT_SHA \
  --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_nonlinear_suffix_observed_recompile_20261028_v2_consume_MAIN01.json
```

## 验证边界/后续

RAM 原生 ABI fixture 不是假执行证据。另有 actual MAIN metadata/保存数组接缝测试，明确不重复上游 strict，不据此签署生产 authority。无神经或物理在此工作位运行，未生成本 adapter 生产 prepare/report/receipt。

产物是四个新 full10 plans，但 prefix5 已执行；未来物理 adapter 必须复原完全相同根/warmup，执行原 prefix，核 exact cp5 start/history/goal 后才执行新 suffix，不可从别的 trajectory 偷接，不能伪造旧 receipt。新的 terminal RGB、native outcome 及科学效能均未知，由 MAIN 后续授权执行。本文档与测试通过不是 P1/fresh/C2 通过。
