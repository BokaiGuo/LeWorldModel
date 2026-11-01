# Moving 两根无组合 full-H2 谱基线 v1

本协议由 MAIN 在授权实现范围内预声明；仅复用原 moving 两个 calibration tasks，不是独立确认。P1/P2 admission 均 false，不训练/拟合新 Q，不读取 physical outcome、sourcepack state 或 witness。禁止把此本地 comparator 称官方复现、等总算力或组合增益证明。

## 输入和数学冻结

输入仅为 moving encoder 的完整实际报告及其12个 hash-bound artifacts。批准报告 SHA：`e7fb554327b3bb5a96190f6291f8450b2b0723cd981817715cfb7e50aa266bf6`。来源权限 token `c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6` 仅转录 MAIN attestation，不打开 sourcepack。复用 moving sampling wrapper 的 `build(('CEM',))` 仅作已验证输入绑定桥，不调用 CEM，不读 sampling 实际结果，重新构造 own schema/contract/source closure。与旧 zero-only loader/consumer 不兼容。

同原 native projected CLS192 目标、identity mean192 terminal Q、三帧真实 latent、两个非零 incoming history packets。raw 前5条 `[.1,0]`、后5条 `[0,.1]` 的真实历史从封存 moving loader 装载，按原 normalizer 转 token，每步滚动；无二次 /100。原目标及根不变。

零点一次完整 H2 terminal Jacobian192×20，经原 NativeCuda 图同时对两 packet 求导，不在 prefix detach/recompile，不按 Vreach 排序。AD 对精确 float32 raw commands 求导并一次乘 tanh chain；零点 chain=1。192个 scalar autograd.grad API 调用，不冒称旧 vectorized Jacobian 的 API 数相同。

复用 `p2_no_composition_sra_20261027_v2.spectral_bank`：zero1；full20 bounded ridge 三 scale `.25/.5/1`；top9 右谱方向 × 两半径 × ± 符号36，共40槽。joint cap `.75 sqrt(10)`、半径 `.25 sqrt(10)` 和 cap。原 **未归一化** residual ridge damping `.002` 保留，即若全目标除192，等价 damping 为 `.002/192`，绝非 mean192 damping `.002`。

其后逐 candidate 用原 `projected_logits` 投影到每 packet `.75 sqrt(5)` logit 球，再 NumPy float32 tanh 成 raw[-1,1]。这是明确改变支持域的映射，不等价原 joint 球/分布，不是精确双 packet 约束 ridge 解；方向/尺度和重复槽不筛选不补采样。40槽全部实际 H2 非线性原 Q，stable first argmin；赢家 fresh whole+segmented H2 各一次，要求两者和搜索路径逐位相同。

## 固定预算和持久化

每 root：nominal2 + AD forward2 + 40candidate×2 + whole2 + segmented2 = **88predict、88action_encoder**；1J、192scalar VJP、2SVD、1bounded ridge solve。两根176/176、2J、384VJP、4SVD、2solve。bounded solve 内含首个 SVD，不是另一次 dense solve API。CPU 每根40packet-pair projections、40candidate scores、1stable selection；编译包含 residual/projection 构造、SVD 和 bounded 求解。上下文每次固定3latent/2incoming token，batch1。pred_proj 内含于 predictor，不另列 image projector。无新 image/physics。

继承 encoder8/projector8 仅全调用记一次，非免费。run 不重构第二次谱 bank；run 每根 planning Q API 43次（40候选+nominal+2replay），run 内数组验证另41次 Q、第二次 stable argmin。own saved-data consume 为验证每根另付 CPU2SVD/1bounded solve/40packet-pair 投影，Q API84次（choice41+journal43）、stable argmin1次；单独披露，不混 planner 神经计数。不是 FLOPs、AD、总时间或 D2P 全资源 matched。

原 NativeCuda module hooks 实记 predictor/action_encoder entry/return，AD forward 也计；禁止 image encoder/projector。每 native context、AD packet、192raw J行、最终 J、原始 joint bank/投影 bank、候选/选择/replay 均保存。操作尝试先 durable、成功同步后才 completion。lossless gzip journal 每事件 Z_SYNC_FLUSH+fsync；错误保留 partial，不退款/重试/替换根。最终 source 与 model hash 分开终结，首错保留。完整成功才 own consume 释放10raw动作及 task/source/export/selection witness。部分错误文件不是可部署 plan。

NEW prepare≤256KiB；NEW run 总≤8MiB（含partial和final report），保留200MiB磁盘 floor，封存 Store 的 final reserve 不降低。源/测试/doc、旧 primitive 和 encoder artifacts/model bindings 一并封存。模型仅 HOST CUDA，MATH SDP、matmul/cudnn TF32 false；没有 CPU neural fallback。容量不足 fail closed，不降资源/guard。

## MAIN 执行接口

在 `/home/gbk/code/LeWorldModel/le-wm-main`，下列均为 MAIN 命令；worker 仅 RAM/mock tests，不执行 prepare 或神经/物理：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_moving_no_composition_baseline_cuda_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_no_composition_baseline_cuda_20261027_v1 prepare --output results/p1_moving_no_composition_baseline_cuda_20261027_v1_prepared01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_no_composition_baseline_cuda_20261027_v1 verify --prepared results/p1_moving_no_composition_baseline_cuda_20261027_v1_prepared01.json --prepare-sha PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_no_composition_baseline_cuda_20261027_v1 run --prepared results/p1_moving_no_composition_baseline_cuda_20261027_v1_prepared01.json --prepare-sha PREPARE_SHA --output results/p1_moving_no_composition_baseline_cuda_20261027_v1_cuda_main01
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_no_composition_baseline_cuda_20261027_v1 consume --output results/p1_moving_no_composition_baseline_cuda_20261027_v1_cuda_main01 --report-sha ACTUAL_REPORT_SHA
```

不得把 own selected plans 直接交旧 zero-only 或白名单物理 runner；后续 moving consumer 由 MAIN 另接，本三文件不含物理执行。Mock 只证明代码合同，不提供实际预测或物理收益证据。当前实际结果未知；不预设改善。
