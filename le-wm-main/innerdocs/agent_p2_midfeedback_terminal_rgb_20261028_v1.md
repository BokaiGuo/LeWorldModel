# P2 新 suffix terminal RGB 窄桥 v1

只新增同名前缀 source/tests/innerdoc。复用已冻结 RGB/native 数组 machinery，不改 recompile 或任何 MAIN 结果；生产由 MAIN 审读后 HOST CUDA 运行。worker 没有神经/物理/网络运行，也未重复生产 consume/prepare。

## 入口复用与不复用

`agent_p2_contextual_midfeedback_recompile_20261028_v1.consume_physical` 完整重建实际物理严格 authority；`physical_material` 重建新编译原生数组/事件/solve/计划及 exact mid replay gate。其 source 固定 SHA `e195749c68747f3d1d820cb62e53d0b109865c4d3c87ead9f040610c953d94ac`。

安全复用 RGB bridge 的 load_pixels/preprocess_pixels、patch14→257 config_shapes、原冻结 NativeTrace/visual_bridge、Artifacts/array SHA 加载和模型 hash。再复用 native encode-only guard 拒绝 predict/action_encoder/predictor/pred_proj 及 AD/backward/torch.func。**不复用旧 fixed_rows/build/run/consume**：旧 roster 要求父 `predicted_Q/witness`，且会编码0/5/10并预测；本次只有新 terminal 最多8编码，不重复 root/mid或goal，不做任何预测或物理。

## 输入 authority 与真 schema

prepare 必须提供 MAIN 实际新 physical output/report SHA 和严格 receipt/SHA。尚未终结的报告或未来SHA不能准备。build 实际调用冻结 `consume_physical`，与整个外部 receipt 作 canonical 精确比较，而非只看 hash/status；之后复核 `physical_material` 全编译 authority。不借用旧 physical report 充新动作结果。

新严格 receipt 的每行是 `{index,control,intervention,new_minus_control,...}`；新 PNG 在 `intervention.rgb_encoding_inputs`，原 helper 不提供 absolute path，本桥用经过校验的 flat `png` 名与实际 output 构造 absolute path。全8固定顺序：2 known roots × off/on × 原joint80/conditional40 arms。完整保留所有行、ties、missing及原native成功/负结果，不过滤，不新增独立样本。

只接 step10，同 condition；cp10 必须 own_returned=10/full_horizon=true，raw_action_history10 bits 必须等于这次**新 full10**（不是父 old suffix），task/blind/model identity、compiler selection seal一致。缺step10则latent/Q/MSE=null，不拿early-stop最终帧当step10，不拿off冒on，不补图、不dedup。缺0/5无需再编码，它们已经在新编译严格authority中绑定。

## 新预测身份与数值

新计划不得含旧泛称 `predicted_Q/witness`。本桥读 `new_suffix_Q`、`new_suffix_context`、`new_suffix_witness`，核 actual-mid suffix5-only scope、原goal192、raw5、新context及独立suffix replay；只用 context.latent_history[-1] 对actual endpoint。父 `parent_predicted_Q` 仅作父身份，不当新动作预测，不假造从root出发的imagined新graph。

每计划保存并输出：

- new suffix predicted Q、actual terminal Q、有符号Q误差、预测endpoint对实际latent的MSE。
- 原 fixed suffix 在**同actual-mid context**的预测Q及其已严格编码actual terminal Q、原Q误差。
- new-minus-old predicted Q与actual Q，所有精确数值不加epsilon/调阈值。
- 完整原 physical pair（原生jointXY/agentXY/objectXY/velocity/wrapped-angle/native reward/success/unknown coverage）并列，不据此选动作。

Q/MSE为原native192、原goal的float64 mean square，不重编码goal、不改Q/normalizer、不训练拟合。native success包含agent+object的原whole-body合同，不冒作object-only coverage；unknown保留null。latent Q变化/误差下降不等于物理收益，`physical_benefit_claim=false`。

## 原生数组、执行与成本

每个available terminal保存原pixels、精确preprocess、encoder输入/完整hidden `[1,257,192]`、CLS projector输入/输出和native192。197拒绝。完整文件SHA/arraySHA/dtype/shape、image/module/encode-only guard attempted/returned双ledger及逐事件次序严格重算，未知事件不filter。native seal在saved数值分析前持久化，完整artifact census/native source/model/input闭包前后核对。模型只加载原checkpoint，CUDA-only、frozen eval/inference、MATH SDP、TF32 off、autocast off；无CPU神经fallback。

最多8encoder/8projector；新predict/AE/pred_proj/AD/physics=0。encode-only guard的`image`是available顺序ordinal（0..available-1），Trace另有原全8row身份，缺帧不改变原row分母。原24encoder+8same-suffix预测、新编译实际resources/AD_API/predict_API attempt/completed各轴和wall、原planner/物理费用与新增物理counts分别继承，不把inclusive API相加作FLOPs。总FLOPs未知null，不称等总成本。

整run600s，从verify前开始计，包含saved authority重建/加载；cap64MiB、final reserve8MiB、disk floor200MiB。失败保留全8roster/已用账/返回数组，strict拒partial，不重试补根。`P1_admission=P2_admission=physical_release=scientific_pass=false`；仅2 known roots，不是fresh证据。

## MAIN 可执行命令

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。`<NEW_PHYSICAL_OUTPUT>`、`<NEW_PHYSICAL_REPORT_SHA>`、`<NEW_PHYSICAL_RECEIPT>`、`<NEW_PHYSICAL_RECEIPT_SHA>` 由 MAIN 完整物理strict后填实际值，不是预填未来SHA。输出必须新建且使用本stem。

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p2_midfeedback_terminal_rgb_20261028_v1 prepare \
  --physical-output <NEW_PHYSICAL_OUTPUT> --physical-report-sha <NEW_PHYSICAL_REPORT_SHA> \
  --physical-receipt <NEW_PHYSICAL_RECEIPT> --physical-receipt-sha <NEW_PHYSICAL_RECEIPT_SHA> \
  --output results/agent_p2_midfeedback_terminal_rgb_20261028_v1_prepare_MAIN01.json

.venv/bin/python -m experiments.spectral_reachability.agent_p2_midfeedback_terminal_rgb_20261028_v1 verify \
  --prepared results/agent_p2_midfeedback_terminal_rgb_20261028_v1_prepare_MAIN01.json --prepare-sha <ACTUAL_PREPARE_SHA>

# 只有 MAIN 审核后 HOST CUDA：
.venv/bin/python -m experiments.spectral_reachability.agent_p2_midfeedback_terminal_rgb_20261028_v1 run \
  --prepared results/agent_p2_midfeedback_terminal_rgb_20261028_v1_prepare_MAIN01.json --prepare-sha <ACTUAL_PREPARE_SHA> \
  --output results/agent_p2_midfeedback_terminal_rgb_20261028_v1_cuda_MAIN01 --authorize-host-cuda

.venv/bin/python -m experiments.spectral_reachability.agent_p2_midfeedback_terminal_rgb_20261028_v1 consume \
  --output results/agent_p2_midfeedback_terminal_rgb_20261028_v1_cuda_MAIN01 --report-sha <ACTUAL_ENCODER_REPORT_SHA> \
  --receipt results/agent_p2_midfeedback_terminal_rgb_20261028_v1_consume_MAIN01.json
```

prepare/verify/consume不加载模型；consume只打印小路径/SHA，全部8行写独占receipt。使用实际返回SHA，不修改冻结prepare。

## 已测试 / 未执行

```
.venv/bin/python -m pytest -q tests/test_agent_p2_midfeedback_terminal_rgb_20261028_v1.py
35 passed in 0.79s
```

RAM测试在fake numpy tensors/modules上**实际调用生产 Trace 和原 encode-only guard**，验证8encoder/projector与完整双ledger/原生pre→hidden257→CLS→projector接缝、guard恢复、predict/AD/AE禁止、缺首/末/多个/全8帧仍保留全部8行、new/parent计划schema、错误raw/model/task/condition/context/cost/witness、197/坏数组/丢attempt/乱序/unknown事件/虚计数failclosed。fake设备标签不是CUDA回执，测试不是科学证据。

未执行生产prepare/verify、CUDA或生产consume。未决依赖只有MAIN实际新物理终态与完整strict report/receipt SHA，以及MAIN审读/主机CUDA授权；没有新增root/模型/物理采集需求。冻结后不得原位编辑，后续修复新版本。
