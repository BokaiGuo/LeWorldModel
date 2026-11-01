# P2 contextual physical 真中间帧编码桥 v1

本版为独占工程接口，源、测试和本文一起纳入 prepare 输入闭包。固定已执行的 8 plans、2 known roots、off/on × joint80/conditional40；不是新根，不是确认性 P2，不产生新的物理 rollout。未经 MAIN 审读与显式 HOST CUDA 授权不运行神经。RAM 单测不是科学证据。

## 真实输入 authority

- Physical output：`results/agent_p2_contextual_composition_physical_handoff_20261028_v1_MAIN01/`。
- Physical report SHA：`c9e8c37984926a68bf2e4d3a92fe47737c03e719ed51917da58b72c888eb038d`。
- Physical strict receipt：`results/agent_p2_contextual_composition_physical_handoff_20261028_v1_consume_MAIN01.json`。
- Receipt SHA：`cc2de96eb6569319228644b62afe45218d0e4df1261c2bdabf6c0a9b052ad818`。
- 原 contextual planner report SHA：`498b45ed9f4c580064d8f2a5dfd00ac8542b7ce9d0b86e6587d9ab6a9639d49c`。
- 原 planner saved_consume **v2** receipt SHA：`578c5bd93394ae9c3863cb420afe4cfd0c59633fbbea503b4012c029211e063f`。

新桥 `build()` 调冻结 handoff 的 `consume()` 完整重建 physical strict authority，并对整个外部 receipt 作 canonical 相等比较；不是只验 receipt hash/status。再经冻结 physical `verify()` 重建 prepare/source/model/原 planner strict v2 authority。所有输入、源码、模型、原 PNG、模型中间张量及本桥三文件纳入 SHA 闭包。既有 report/receipt 均不改写。完整保存原 compiler resources 与 physical counts，分轴而非相加。

复用：handoff 的全8计划/raw/native/PNG严格验证、原 overlay `NativeTrace.save`/`traced_scope`/`visual_bridge`、RGB `Artifacts`/数组完整性加载/config-shape 校验、teacher RGB v2 原像素预处理、已验证 API 实例 hook 恢复机制。**不复用**旧 RGB 的固定 arms roster/build/run 或 39/40 槽执行器；本桥实际独立 dispatch 编码和同后缀 API，既有产物不会冒作新计算。

## 实际回调与语义

所有 available checkpoint0/5/10 分别编码，同 condition 不混用，不去重。0/5/10 指 plan 执行步，相当于原10步 warmup 后的 public10/15/20。PNG 原字节 SHA、像素 SHA、真实 float32 预处理、encoder hidden `[1,257,192]`、CLS、projector、192 latent 全数组持久化并严格接缝。197 tokens 拒绝。原 goal 为已封存 goal192，保持原 identity Q，不重新编码或替换。

每计划使用真实 checkpoint5 RGB 重建：

```
latent3 = [原 root_context.latent_history[1] (warm5),
           新 encode(checkpoint0/root10), 新 encode(checkpoint5/actual15)]
outgoing3 = [normalize(cp5.raw10[:5]), normalize(cp5.raw10[5:]),
             normalize(原已选 full10[5:])]
action_encoder(outgoing3) -> model.predict(latent3, action_embeddings)
```

cp 的 incoming raw10 必须精确等于 `(原 warm10 + 本计划 executed prefix[:checkpoint])[-10:]`；原 full10、suffix5 bits、原 model selection digest 与新 physical seal、task/blind/condition/goal/context 一并绑定。normalizer 为原 float32 减均值除标准差一次，无额外 `/100`、tanh 或 clipping。无 native state/metrics 进入 model/compiler；没有 compiler、优化器或新动作选择。这里是**真实 mid RGB 上原动作的模型重放预测**，不对机器人反馈施加新动作，不能称闭环任务改进。

完整 native `[1,3,192]` 输出保存；只有最后一行对应 actual terminal。前两行 native 输出绝不和 shifted history 前两行混比。预测模块实际 entry 顺序 AE / predict_API / predictor / pred_proj；predict_API return 在其内部 predictor/proj return 后，consumer 精确检查嵌套 chronology，不过滤未知事件。

输出逐8计划：root 重编码 MSE、imagined prefix MSE、imagined terminal MSE、真实 mid 同 suffix terminal MSE、原 predicted Q、新 feedback predicted Q、各自相对 actual terminal Q 的有符号误差，以及 checkpoint0/5/10 actual latent Q。MSE/Q 都是 float64 mean-squared distance of native192。与原 native jointXY/agentXY/objectXY/velocity/wrapped-angle/native reward/success/coverage 分栏；保持原 unknown coverage 为 null。native success 包含 agent+object，不冒作 object-only coverage。warmup 旧 goal reward 不混入当前 task goal：checkpoint0 用原 strict initial_metrics，后续只取同 plan step metrics。

缺帧保留所有8计划/全部3 checkpoint 槽：latent/Q/MSE=null；缺0或5不预测；缺10仍可预测但 actual terminal 误差=null。partial/failure 保持已用预算/attempts/返回数组及全8分母，strict consume 不当完成；无重试/补帧/补动作/替代根。

## 成本与边界

最多 encoder=24、projector=24、action_encoder=8、predict_API=8、predictor=8、pred_proj=8。available 数量在 prepare 固定，不按结果改变。predict_API 是包含 predictor/proj 的嵌套 API，不得累加冒称独立 FLOPs。实际全 tensor attempts/returns、输入/输出数组 SHA、event、native seal、全部 artifact census 保存；超出任何 cap 在模块 predispatch 拒绝。总 FLOPs/底层隐藏 backward engine 调用数 unknown=null；请求 AD=0，训练/Qfit/优化器/新physics=0；不称等 FLOPs 或等总成本。

原冻结 weights SHA `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`；config SHA `2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09`；原 model tensor SHA `4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546`。CUDA-only/eval/requires_grad=false/inference/MATH SDP/TF32 off/autocast off，无 CPU 神经 fallback。模型 before/after、全部 source/model/input before/after 验证。

整作业 wall600s，output64MiB（8MiB final reserve），disk floor200MiB。包括失败在内保存实际已用数组/计数；不为了完成抹除错误。`P1_admission=P2_admission=physical_release=scientific_pass=false`。latent 误差不是物理收益。当前只接2 known roots，fresh/P1 科学前置仍未满足。

## MAIN 的精确 CLI

从 `/home/gbk/code/LeWorldModel/le-wm-main` 执行；prepare/verify/consume 为 saved-only，但 prepare/verify 会完整重建 authority，可能需要读大 planner trace。worker 未重复执行 MAIN physical consume/生产 prepare。输出必须为未存在的新独占路径。

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p2_contextual_physical_rgb_encode_20261028_v1 prepare \
  --physical-output results/agent_p2_contextual_composition_physical_handoff_20261028_v1_MAIN01 \
  --physical-report-sha c9e8c37984926a68bf2e4d3a92fe47737c03e719ed51917da58b72c888eb038d \
  --physical-receipt results/agent_p2_contextual_composition_physical_handoff_20261028_v1_consume_MAIN01.json \
  --physical-receipt-sha cc2de96eb6569319228644b62afe45218d0e4df1261c2bdabf6c0a9b052ad818 \
  --output results/agent_p2_contextual_physical_rgb_encode_20261028_v1_prepare_MAIN01.json
```

将 prepare 返回的实际 SHA（不可预填未来 SHA）代入 `<PREPARE_SHA>`：

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p2_contextual_physical_rgb_encode_20261028_v1 verify \
  --prepared results/agent_p2_contextual_physical_rgb_encode_20261028_v1_prepare_MAIN01.json --prepare-sha <PREPARE_SHA>

# 仅 MAIN 审核后 HOST CUDA 执行；不是本worker执行命令。
.venv/bin/python -m experiments.spectral_reachability.agent_p2_contextual_physical_rgb_encode_20261028_v1 run \
  --prepared results/agent_p2_contextual_physical_rgb_encode_20261028_v1_prepare_MAIN01.json --prepare-sha <PREPARE_SHA> \
  --output results/agent_p2_contextual_physical_rgb_encode_20261028_v1_cuda_MAIN01 --authorize-host-cuda
```

将 run 返回的实际 report SHA 代入 `<RGB_REPORT_SHA>`：

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p2_contextual_physical_rgb_encode_20261028_v1 consume \
  --output results/agent_p2_contextual_physical_rgb_encode_20261028_v1_cuda_MAIN01 --report-sha <RGB_REPORT_SHA> \
  --receipt results/agent_p2_contextual_physical_rgb_encode_20261028_v1_consume_MAIN01.json
```

consume 只打印小 receipt 路径/SHA，完整8行分析写独占文件，不输出巨量张量。

## 已实际轻量验证 / 未运行

```
.venv/bin/python -m pytest -q tests/test_agent_p2_contextual_physical_rgb_encode_20261028_v1.py
47 passed in 4.22s
```

RAM fake tensor/module **实际调用本版生产 Trace 的 hooks** 与 process/native-chain/strict validate，验证24+8完整 dispatch、所有模块入返 chronology、恢复 hooks、257通过/197拒绝、checkpoint缺失与首图失败全8保留、模型/goal/context/raw10/suffix身份、坏模块/数组/事件/分母拒绝、端点语义、AUTH-before-I/O、输出预算与完整authority而非只看status。fake cuda device 标签仅为受控RAM测试双，不是CUDA运行回执。只读核过实际 physical report/receipt/prepare 及原 slot result 的字段布局，不重复消费或执行生产产物。

未运行：生产 prepare/verify、本桥 HOST CUDA、生产 consume。需 MAIN 审核本三件套并执行上述 CLI；本版本冻结后不得原位修改，后续修复新版本。工程通过也不解锁 P1/P2/fresh/物理效力结论。
