# Contextual8 actual-mid 重编 suffix5：最小配对干预 v1

本三件套冻结后由 MAIN 审读/运行；worker 只运行 RAM 测试，没有生产 prepare、CUDA、物理或网络操作。P1/P2/fresh/physical_release 一律 false。只涉及2 known roots × off/on × 原两臂的8计划，保留全部分母、零和负结果。

## 不是重复 teacher 预测

旧 `agent_p2_midpoint_teacher_suffix_20261028_v1` 只替换最后一个 latent、重预测原 suffix，原四根/八臂无新物理动作。新 contextual RGB bridge 虽用真实 warm5/root10/actual15 history，也仍执行原动作，只增加同 suffix 预测。二者均没有本次“actual-mid 重解并实际部署新 suffix”的干预。

本版最小新增：从已经 strict 编码的真实 mid context 重新求一个原 H1 bounded-ridge suffix5；随后复用冻结 `agent_p2_contextual_composition_physical_handoff_20261028_v1.execute`、原 episode/Store/Meter/native hooks/dispatch/strict assemble，执行同 root、同 prefix5、同 goal/Q/原生 success 合同的 full10。**没有重写通用闭环框架，没有接 P3 full20 任务。** 对照直接复用已真实执行且严格消费的原8 full10 fixed-suffix 结果，不再跑一组相同对照。

这是分阶段的 observation-conditioned 编译 + exact replay-gated 部署，不是单进程在线 latency benchmark。模型在原真实 mid RGB 后编译，物理重放时必须在 suffix 进入前证明重现了该 mid 的全部精确输入；不符合就拒绝，不近似恢复、不重新开根。没有状态 setter 用于 mid 恢复。

## 真实 authority 和固定输入

- RGB report `results/agent_p2_contextual_physical_rgb_encode_20261028_v1_cuda_MAIN01/report.json`，SHA `b57fa03560696c3c0b75fe842ec8263d6ec3961341d6736a4f6e782d1f0e92ea`。
- RGB strict receipt `results/agent_p2_contextual_physical_rgb_encode_20261028_v1_consume_MAIN01.json`，SHA `3d3d020c113959aec25f4e310b8512b814f7a78976f2bd7d5e58e0e435027f4b`。
- 原 physical report SHA `c9e8c37984926a68bf2e4d3a92fe47737c03e719ed51917da58b72c888eb038d`，strict receipt SHA `cc2de96eb6569319228644b62afe45218d0e4df1261c2bdabf6c0a9b052ad818`。
- 冻结 RGB source SHA `fa1d44fb2b753b2deb2c1743cfa0d0496b4fdea42f92d4190cce231260d5375e`。

prepare 完整重建 RGB consume、整个外部 strict receipt、原 physical prepare/planner strict v2/source/model 闭包。不只看 hash/status。只从原 winner 的 candidate_id 取原 logits，并严格核 `np.tanh(float64 logits).astype(float32)` 等于原 full10 bits；不从 rounded raw 反推原 logits。全8计划顺序不改，缺0/5任一输入阻止整个编译准备，不静默删槽。

compiler 实际 whitelist：原 warm5 latent + 原真实 root10 re-encode + actual mid15 encoding、cp5 incoming raw10、原 prefix/old suffix、原 goal192、identity Q、原 normalizer、剩余 logit radius。terminal actual latent、native状态/成功/reward/metric不进入 compiler；control native metrics只在封存新动作之后用于配对报告。

## 动作域与不同的生成映射

保持原 joint20 logit radius `CAP=.75*sqrt(10)`；prefix 固定，suffix solve 的 radius 是 `sqrt(CAP²-original_prefix_logits@original_prefix_logits)`。不改成另一个 packet radius。剩余半径非正即 fail closed，全8分母保留，不扩大半径。解的是原 identity-Q、damping .002 bounded ridge，无拟合和阈值变化。

**映射并非逐位相同生成器**：原 frontier 候选是 NumPy `tanh(float64 logits) -> float32`；本次新 suffix 继承 `moving_history_observed_suffix_cuda` H1 helper，为 `tanh(params.astype(float32))`。只继承同半径约束/原 normalizer/原Q，不声称逐位候选生成器相同。原 prefix raw5 完全不变，不 clip/rescale 新 raw。H1 helper 在 old diagnostic 预测前已经封存新 solve；无条件部署新解，即使 old diagnostic 的预测更好也不回退、不选 winner。

新计划明确拆分 provenance：旧 `predicted_Q/witness/selection_digest/compiler_selection_digest` 仅保存在 `parent_*`；不把旧图挂作新动作见证。新 `new_suffix_Q`、`new_suffix_context`、`new_suffix_witness` 只陈述 actual-mid→suffix-terminal（非从原 root 出发的 imagined full10 graph），附原生数组/独立 replay/context/raw/goal 与 compile row/report digest。新 full10 rawbits/SHA 和部署 selection digest 重新封存，parent 不改写。

## exact mid gate 与物理计量

`ReplayGateStore` 是原 native Store 的窄扩展，直接接真实 `fresh_observation` 的 branch、receipt.phase/step/condition/overlay_index。新物理 step5 选定 condition 的 PNG **文件字节 SHA、像素 SHA、7维 state、完整 snapshot** 必须与已 strict 编码的 cp5 相同。失败在持久化实际观测后抛错；step6 的 `operation_attempted` 还要再次检查 gate 才进入 native call。全8固定 prefix/rawfloat32 由原 episode 的每步 actual/latest_action 与 paired.check_return 以及 saved strict assemble 校验；不以 requested action 冒 actual。

旧原生 full10 runner 保留原 placement/reset/warm10、goal bind、goal_pose outline、off/on/off、全部 native metrics。warmup 旧 goal reward 不用于当前 task goal。原 success 是 agent+object XY<20 且原 angle term<pi/9，不是 object-only coverage；unknown coverage 保持 null。不得编辑 goal_pose/状态/目标制造图。各臂若 early terminal 则按原 runner 停止，不补动作，不将 nominal10 冒成实际10。

物理 strict consumer 直接调用冻结 `physical.assemble` 重建全 raw/原生 state/goal/native reward/停止/PNG triplet/每 operation reserve-return 嵌套账，另核 gate、执行 binding、完整 artifact census、journal SHA/hashchain/source闭包。返回全部8项 control 与 intervention 原 metrics、有符号 new-minus-control、各自 native success；不成功过滤。工程故障 report/已用计量保留但 strict 不放行，pending 不能当零成本。

## 完整预算与失败边界

每个新编译：5 native predict（nominal、AD forward、old diagnostic、new、new replay）；AE/predictor/pred_proj各5；Jacobian1（192×10）、SVD1；AD API jacobian1/grad1，192 mathematical VJP directions。总8 compile、40 predict（inclusive API另轴）、8 Jacobian/SVD、1536 mathematical VJP。新 encoder/projector=0（复用 paid actual RGB 编码），新 physics 编译阶段=0。完整模块 inputs/output NPY、math/AD/API ledger与attempt/input-save/return、全事件durable存储；独立saved重算bounded solve。AD的grad return先于jacobian return，不能错当 ledger entry 顺序。未知事件不 filter，缺 attempt/输入/return/分母拒绝。

新增物理最多8实例、16 reset、160 public（80warm+80control）、1632 physics、528 PNG，原 caps不变。两阶段分别 cap64MiB / final reserve2MiB / disk floor200MiB / 900s。900s 从阶段入口、验证前开始计，包含 saved authority 重建、加载和实际执行，非抢占watchdog；独立 prepare/verify 的耗时不冒作模型 forward。原 compiler1328 predict、RGB24 encode/8同suffix预测、旧物理费用继承另列，未知上游搜索/总 FLOPs=null，不称等 FLOPs 或更省资源。相同控制长度不等于相同总计算。

## MAIN CLI（冻结后，future SHA只能取实际返回值）

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。所有输出必须不存在、使用本新stem。

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p2_contextual_midfeedback_recompile_20261028_v1 prepare \
  --rgb-output results/agent_p2_contextual_physical_rgb_encode_20261028_v1_cuda_MAIN01 \
  --rgb-receipt results/agent_p2_contextual_physical_rgb_encode_20261028_v1_consume_MAIN01.json \
  --output results/agent_p2_contextual_midfeedback_recompile_20261028_v1_prepare_MAIN01.json

.venv/bin/python -m experiments.spectral_reachability.agent_p2_contextual_midfeedback_recompile_20261028_v1 verify \
  --prepared results/agent_p2_contextual_midfeedback_recompile_20261028_v1_prepare_MAIN01.json --prepare-sha <PREPARE_SHA>

# 以下新神经/物理仅 MAIN 在 HOST 分别授权。
.venv/bin/python -m experiments.spectral_reachability.agent_p2_contextual_midfeedback_recompile_20261028_v1 compile \
  --prepared results/agent_p2_contextual_midfeedback_recompile_20261028_v1_prepare_MAIN01.json --prepare-sha <PREPARE_SHA> \
  --output results/agent_p2_contextual_midfeedback_recompile_20261028_v1_cuda_MAIN01 --authorize-host-cuda

.venv/bin/python -m experiments.spectral_reachability.agent_p2_contextual_midfeedback_recompile_20261028_v1 consume_compile \
  --compile-output results/agent_p2_contextual_midfeedback_recompile_20261028_v1_cuda_MAIN01 --compile-sha <COMPILE_REPORT_SHA> \
  --receipt results/agent_p2_contextual_midfeedback_recompile_20261028_v1_compile_consume_MAIN01.json

.venv/bin/python -m experiments.spectral_reachability.agent_p2_contextual_midfeedback_recompile_20261028_v1 run_physical \
  --compile-output results/agent_p2_contextual_midfeedback_recompile_20261028_v1_cuda_MAIN01 --compile-sha <COMPILE_REPORT_SHA> \
  --output results/agent_p2_contextual_midfeedback_recompile_20261028_v1_physical_MAIN01 --authorize-physics

.venv/bin/python -m experiments.spectral_reachability.agent_p2_contextual_midfeedback_recompile_20261028_v1 consume_physical \
  --output results/agent_p2_contextual_midfeedback_recompile_20261028_v1_physical_MAIN01 --report-sha <PHYSICAL_REPORT_SHA> \
  --receipt results/agent_p2_contextual_midfeedback_recompile_20261028_v1_physical_consume_MAIN01.json
```

run_physical 内部重新完整 consume_compile 后才开环境，不能用简化receipt绕过实际模块数组验证。consume CLI只打印小路径/SHA，完整8行写独占receipt。

## 实际轻量测试 / 未运行

```
.venv/bin/python -m pytest -q tests/test_agent_p2_contextual_midfeedback_recompile_20261028_v1.py
30 passed in 11.33s
```

测试直接复用原 H1 evaluate/Modules/native-buffer validator和原物理 execute/episode/strict assemble，在RAM fake arrays/native-interface doubles上执行；真实emit字段接线通过，mid任一精确身份变异使全8在5步后停、不dispatch新suffix；new/parent provenance、全部native/module/math/AD/API事件、坏输入/输出/unknown/丢attempt/丢return/乱序、原suffix更优仍新解、原始logits/domain、AUTH-before-I/O均覆盖。初轮测试暴露并修复AD嵌套return顺序及RAM Store继承接线；失败未当科学负结果。

worker未执行真实 build/prepare、生产 consume重建、CUDA或物理；MAIN尚需实际 prepare→verify→CUDA→consume_compile→物理→consume_physical。RAM不验证科学效能；实际mid误差/排序改善也不等于控制任务收益。所有原封存三件套和结果保持不变，本版本冻结后修复另开新版本。
