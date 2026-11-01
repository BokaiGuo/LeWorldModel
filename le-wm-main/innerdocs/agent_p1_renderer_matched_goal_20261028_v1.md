# P1 actual20→25：renderer-matched goal 固定80对诊断

## 交付范围与证据缺口

独占新增同 stem source/tests/本文。不修改任何 v2/v3/v4 封存、源码、结果、gate 或阈值；无训练、Q拟合、筛根、挑对或 fresh reserved。结论范围是事后固定动作对照，不是新 selector、部署纠偏或 P1 admission。

**所需目标图像/编码已经充分封存，不需要新目标RGB，不需要 CUDA/物理实验。** 原先 v4 `effects` 对 original_endpoint 和 alternate_endpoint 都沿用原 goal；后者不能单独回答“匹配 renderer 后的任务偏好是否变化”。本 consumer 实现真正的终态 renderer × goal renderer 2×2 对照。

证据链（本次实际核读，非推测）：

- actual20→25 固定40行 report：`results/agent_p1_prospective_action_effect_20261028_v2_MAIN01/report.json`，SHA `2e7794d32602ca2b045202099fedb6b99ac5988d6255ee68870032219530084c`。
- MAIN 完整 strict receipt：`results/agent_p1_prospective_action_effect_20261028_v4_MAIN01.json`，SHA `5a5c1948dd37ff0725447887218f6d57e59aaae54a3f1df28925fb8fe32b01a8`。直接复用此接受证明及 frozen authority；不重复跑原 producer 或全量 consume，不抹去原 v2/v3 failures。
- 同物理目标的 RGB：`results/p1_moving_target_overlay_source_20261027_v1_main01/s{0,1}_{off,on}_blind_inputs.npz` 的 goal_rgb。source report SHA `85ca1df3de910628bfd6c5bd1a0a482f61880a2ee76bfd404732a77dd8c7666a`。其 construction 终态20 off/on/off 是真实渲染，signed journal 连接完整 state7/snapshot、预后不变量、默认 outline pose、同 renderer 与 off 恢复，不是合成或重绘目标。
- 四份原目标 latent：`results/p1_moving_overlay_recompile_cuda_20261027_v1_cuda_MAIN01/report.json`，SHA `2b2e5064f74f8aca712ae0d0415fcbfb64a2c8e6325bdffe24ae4fdc3b373381`。固定 image index **3/7/11/15** 对应 root0 off/on、root1 off/on；各自有 pixels/preprocessing/encoder257/CLS/projector/native192 数组，与现 P1 task_export 和 anchors.goal 精确相等。原始 encoder/model tensor hash 与 actual20→25 相同。

独立验证目标路径：PNG bytes/pixels=blind packet.goal_rgb=编码保存 pixels；原 original224 NumPy preprocessing 精确到 dtype/shape/bytes；encoder 输入/输出257/CLS/projector/latent 全桥接；所有数组 SHA、原生 ABI、原封存 module ledger/image scope、task_export 身份及既有物理 goal pixel SHA 相连。source journal 使用其原始 serializer/hash，不把较晚 v2 的 JSON ABI 套用到旧文件。

注意两种 pose：constructed physical goal 是 source 终态的完整 state7；native `goal_pose` 是原环境默认 outline pose。on 渲染保留默认 outline，**不能声称它就是 constructed goal 的绘图位置**。两份 RGB 的物理 state7 完全相同，outline pose 不改；也不把渲染敏感性解释为物理任务变化。

## 事前固定计算

全部8 anchor ×5 word、每 anchor 无序10对，总80/80，unknown0。与旧 teacher15 的76/4 panel 不混同。使用每行已封存的 original/alternate 两 latent、两 root 对应的 off/on goal latent；无新 predict，不将某 anchor off 条件 rollout 与另一 on 条件 rollout 拼成相同动作对照。

令实际同动作终态 latent 为 zᵣᵢ、目标 latent 为 gˢ，r,s∈{off,on}。四种 Q=mean((zᵣᵢ−gˢ)²)，固定 identity192，float64均值，仅除192一次；原模型 seal 的 selected_word 始终不变。报告该固定 word vs zero 的四种 Q差及原 native/XY/angle差，禁止用物理标签或重算 Q 改选 word。

逐对 margin Mᵣˢ=(||zᵣᵢ−gˢ||²−||zᵣⱼ−gˢ||²)/192。全部 margin 以原 IEEE数值转 Fraction 精确计算，报告 numerator/denominator、sign 与 float(exact) 展示值；该展示值不声称等于两次普通浮点 Q 相减。普通 float64 word Q 独立保留为 Q_float64；无 epsilon、无 near-tie 合并。

对原条件 c、alternate条件 a，分别报告：Mᶜᶜ（原匹配）、Mᵃᶜ（原先不匹配诊断）、Mᵃᵃ（新匹配诊断）及所有 strict reversal/tie。目标替换的精确独立贡献：

`Mᵃᵃ − Mᵃᶜ = −2 <zᵃᵢ−zᵃⱼ, gᵃ−gᶜ>/192`。

完整 renderer×goal 交互：

`M_on,on − M_on,off − M_off,on + M_off,off = −2 <(z_on,i−z_on,j)−(z_off,i−z_off,j), g_on−g_off>/192`。

两条恒等式要求 Fraction residual=0。该归因区分终态 renderer 影响、目标编码影响、交互，绝不追求最大 reversal 改善。另与同一行既有 native distance、XY和wrapped-angle float64排序对照，保留 ties（非严格反转），不替换原物理指标或门。

## 剩余科学未知量

这能回答“固定真实终态在匹配 goal renderer 时的偏好”，不能证明冻结模型在另一 renderer context 下也会产生相同动作预测/选择，不能逆转已知物理XY/angle恶化，更不能把两 known roots 当 fresh 确认。没有待补目标图像/编码缺口；若要研究另一 renderer 的完整闭环选择，须另立前瞻合同，不能靠此事后重评分代替。

## CLI 与测试

只有 saved-only `analyze/consume`，刻意无 prepare/run/CUDA 命令，因为新增编码会重复已有证据。从 `/home/gbk/code/LeWorldModel/le-wm-main` 执行：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p1_renderer_matched_goal_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_renderer_matched_goal_20261028_v1 analyze --output results/agent_p1_renderer_matched_goal_20261028_v1_MAIN01.json
sha256sum results/agent_p1_renderer_matched_goal_20261028_v1_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_renderer_matched_goal_20261028_v1 consume --analysis results/agent_p1_renderer_matched_goal_20261028_v1_MAIN01.json --analysis-sha REPLACE_WITH_SHA_ABOVE --output results/agent_p1_renderer_matched_goal_20261028_v1_consume_MAIN01.json
```

结果与receipt均 xb独占，禁止覆盖；consume 从固定输入重新计算整个分析并精确比较，非只信已有 summary。正式落盘由 MAIN 审读后执行。TDD 已先观测缺实现报红，新增 summary 行为也先观测失败再实现。最终本 stem **22 tests** 与 frozen v4 的36 tests 联合 **58 passed in 7.53s**，含真实四目标来源、真实 triplet、目标/时间/条件/词/selection 篡改、像素/preprocess/latent协同篡改及 journal 哈希篡改。

worker `build()` 全量只读 smoke exit0，144865-byte JSON可序列化（文档最终SHA绑定可能令大小不变但内容不同），未生成正式结果/receipt。固定80对全部可用：原终态与 alternate 共用原目标时7对 strict reversal，两端均匹配 renderer 后4对；alternate 单独更换 goal 有7对反转。对既有 XY 排序，original_matched/alternate_original_goal/alternate_matched 分别7/12/7对反转；native distance 为36/43/38对；wrapped angle 为28/30/30对，所有三列均保留40个物理angle ties。无新物理结果；不把任一计数改善作为入门证据，也不回改原结果。只写新三个文件，所有原数据保持只读。source/tests/doc 的冻结SHA由交付消息报告，避免文档自哈希。
