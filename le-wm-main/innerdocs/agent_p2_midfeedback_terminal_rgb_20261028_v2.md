# Safeguarded v2 terminal8 RGB bridge

只新增本 stem source/tests/innerdoc，terminal v1 和 recompile v1/v2 保持冻结。MAIN 完整审读后在 HOST CUDA 执行；worker 仅 RAM tests，没有生产 prepare/verify、模型/物理/网络操作，不创建下级代理。

## 实际绑定与窄复用

绑定 `agent_p2_contextual_midfeedback_recompile_20261028_v2`，source SHA `e459cdc5de507353873346d6f86bc223afed291982904c4c4c6ce9bdfb7eaeab`；共享 terminal v1 source SHA `e7bcd2d9a4f0920eb4c715ddd0d095b22d4d6ba3b2b1bd6c49599836b6ab68ac`。两者在 prepare 校验且纳入闭包，不修改它们。

新 prepare 参数为**实际 v2** physical output/report SHA/strict receipt path/SHA；没有 future SHA 常量。必须调用 `v2.consume_physical` 完整重建真实8计划/原数组/计量/source/model/native gate，再逐字节比较整个外部receipt。接着 `v2.physical_material` 完整 consume compile，核 prepare/compile 的 v2 schema和完成状态，得到实际选中的 full10/Q/context/witness。物理 receipt 没有 `control_report_sha` 不是缺项：本接口根本不读该字段，对照来自原已strict RGB authority与 v2 完整paired rows。

对每槽核 `safeguard == physical_pair.safeguard`、selected_index 为整数0..4、domain合法、chart为声明 float64 tanh→float32、selected_Q等于new_suffix_Q且不大于incumbent_Q；incumbent_Q还需等于原 actual-mid same-suffix 的模型Q。该匹配不使用真实终态latent、native reward/success作选择。新 suffix context/witness只表示actual-mid→terminal，不拿旧parent graph冒充新预测。

复用 terminal v1 的 policy-free `process`、`fixed_rows` 共用frame ABI校验、`validate` 全量tensor/native guard校验。为了调用冻结纯函数，在**已检查新v2 schema/contract之后**构造不修改原对象的本地兼容view：只将共用row validator的schema标签、共用tensor validator的contract常量投影为其原预期。没有猴补globals，没有修改真实authority字节/SHA/schema，没有把v1 receipt当v2；所有新receipt/prepare/report仍用v2 schema。plan/row/event/arrays/counters一字不改传给原validator；v2差异的guard/selection身份在新入口额外校验。新run单独接自己的verify/Store/AUTH，不调用旧run。

## 固定8槽、真实历史与编码

固定2 known roots × off/on × 原两臂，全部8槽。即使8个选择都为incumbent，也对本次真实physical step10 PNG分别执行8 encoder+8 projector，不读取旧terminal latent替代、不按raw/PNG相同去重。step10来自相同condition；缺checkpoint10则该槽null，不以early terminal、另一condition或父结果填补，仍保全8分母；n个可用时真实encode/projector各n，nominal上限8。

共享validator逐项绑定本次plan selection/compiler digest、parent身份、root/goal/task/blind/model/condition、真实full10 rawfloat32因果history、PNG文件bytes/像素SHA及原preprocessing。新预测明确读取 `new_suffix_Q/new_suffix_context/new_suffix_witness`。原goal192/identityQ/normalizer不变，无重新goal编码或Q拟合。

图像路径使用实际v2 output，不复用v1文件。真实输入 `1×1×3×224×224`、native ViT `1×257×192`、CLS→projector `1×192`、最终latent192。197明确拒绝。继承 actual Trace 与 encode-only native guard，持久化每个encoder/projector的完整输入输出NPY、scope、actual counts、attempt/return/forbidden-API guard及完整事件；无predict/AE/predictor/pred_proj/AD/physics。

native_seal先于saved分析。consumer完整重建prepare/source/model/physical/compile authority与所有artifact census/SHA、绑定、roster、seal、native数组、preprocessing和事件精确次序，不是只看hash/status。unknown/缺事件/未完成/计量不闭合一律拒绝；missing和失败不当零。原raw10/native taskgoal/whole-body success/coverage由物理strict authority认证，本encoder不重新模拟或根据指标选计划。

## 输出语义和成本

全8行输出 actual terminal Q、所选predicted Q、endpoint MSE、预测/真实new-minus-old Q、原fixed实际mid预测和actual Q、全部physical paired指标，另外附safeguard/selected_index/incumbent_selected。缺latent保持null。incumbent_selected不意味着免执行/免编码，也不意味着fresh。

guard只保证同模型输入下的已评估model-Q不劣于旧点，**不保证真实latent Q、预测排序或原生任务收益**。这些保证字段明确false，`physical_benefit_claim=false`。不能用terminal v1已见结果、native标签或本次encoding再调选择器。机制失配和工程完整性分开，不把latent负结果当C2或任务失败。P1/P2/physical_release均false，2known roots不当8独立样本。

新增仅至多8encoder/8projector，重编译真实56predict/8Jacobian及原编译、原RGB、物理费用都继承分轴列出；API inclusive轴不重复求和，总FLOPs未知null，不称等FLOPs。run从verify前开始计600s（非抢占watchdog），64MiB输出/8MiB final reserve/200MiB磁盘floor，无重试/CPU neural fallback。model before/after hash及source闭包必须不变，TF32关闭且finally恢复调用前标志，MATH SDP/autocast关闭/冻结eval。

## MAIN CLI：真实v2物理完成后才prepare

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。`<V2_PHYS_OUTPUT>`、`<V2_PHYS_REPORT_SHA>`、`<V2_PHYS_RECEIPT>`、`<V2_PHYS_RECEIPT_SHA>`必须来自实际v2 native physical+strict，不使用terminalv1或recompilev1的SHA。输出全部是不存在的新路径。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p2_midfeedback_terminal_rgb_20261028_v2.py

.venv/bin/python -m experiments.spectral_reachability.agent_p2_midfeedback_terminal_rgb_20261028_v2 prepare --physical-output <V2_PHYS_OUTPUT> --physical-report-sha <V2_PHYS_REPORT_SHA> --physical-receipt <V2_PHYS_RECEIPT> --physical-receipt-sha <V2_PHYS_RECEIPT_SHA> --output results/agent_p2_midfeedback_terminal_rgb_20261028_v2_prepare_MAIN01.json

.venv/bin/python -m experiments.spectral_reachability.agent_p2_midfeedback_terminal_rgb_20261028_v2 verify --prepared results/agent_p2_midfeedback_terminal_rgb_20261028_v2_prepare_MAIN01.json --prepare-sha <ACTUAL_PREPARE_SHA>

# 仅MAIN审核并授权HOST CUDA：
.venv/bin/python -m experiments.spectral_reachability.agent_p2_midfeedback_terminal_rgb_20261028_v2 run --prepared results/agent_p2_midfeedback_terminal_rgb_20261028_v2_prepare_MAIN01.json --prepare-sha <ACTUAL_PREPARE_SHA> --output results/agent_p2_midfeedback_terminal_rgb_20261028_v2_cuda_MAIN01 --authorize-host-cuda

.venv/bin/python -m experiments.spectral_reachability.agent_p2_midfeedback_terminal_rgb_20261028_v2 consume --output results/agent_p2_midfeedback_terminal_rgb_20261028_v2_cuda_MAIN01 --report-sha <ACTUAL_ENCODER_REPORT_SHA> --receipt results/agent_p2_midfeedback_terminal_rgb_20261028_v2_consume_MAIN01.json
```

CLI只打印小路径/SHA；完整表和数组留durable输出。consumer无新neural/physics。

## RAM测试和边界

复用原RAM fake tensor/modules，真实 Trace/native guard/process/validator 接线：混合及全8incumbent仍8encoder、null缺项、197拒绝、v1schema拒绝、guard/selected/context/Q/actual history变异、完整数组/事件/计量变异、禁predict/AD/AE、whole receipt重建、AUTH-before-I/O、兼容view不修改原authority/globals。首轮34passed/1failed：raw变异测试把原本全零的rawbits又写成零，未形成变异；已改成确实不同的bits，不放宽源码检查。RAM不是模型/物理或科学证据。

最终联合回归：本v2 36项 + terminal v1 35项 + safeguarded compiler v2 43项 + MAIN guard 11项，**125 passed in17.38s**。含完整RAM prepare成功接线（真实v2字段，无control_report_sha；strict重建→material→全8→56predict继承账→8encoder预算），不是只测拒绝路径。模块 `--help` 退出0。冻结前再次核验旧terminal v1及compiler v2 source SHA未变，三SHA见交付消息，本三件套至此冻结。

worker未执行生产prepare；精确未决依赖是MAIN真实v2 compiler+物理strict全闭包及其实际report/receipt SHA，然后由MAIN执行HOST CUDA和strict。没有外部批准补件，不伪造任何未来authority。
