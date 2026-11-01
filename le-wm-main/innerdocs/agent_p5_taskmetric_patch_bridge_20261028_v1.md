# P5 saved-only taskmetric bridge：全 256 patch，不产生新加权 Q

独占新 source/tests/innerdoc。只消费已完成 v4 的四固定词同时间 frame20；无 encoder/predict/AD/physics，无训练、拟合、阈值调节、根/词/patch 筛选。MAIN 审读和实际 saved CLI。旧 v4、失败 renderer、科学门均不变。

## 先冻结 mask 拒绝及后备合同

原 goal RGB 的目标色干预为 null first-stage；不能据此说 encoder invariance。原生 `pusht_env.py:_render_frame` 先白背景、再 goal polygon、后 live space.debug_draw、最后 cv2.resize。goal 与 live object 的目标位置重合时，目标色可能全遮挡。单纯“精确等于 LightGreen”最多取到可见内点，并不能恢复遮挡区域/插值边界；精确等于 block 灰色又不能认证这些像素就是原 goal overlay。背景非白也包括 agent/walls。故拒绝将这些快捷规则冒充合法完整 goal mask，不设色差阈值、膨胀半径、占比门槛或近邻补图。

`mask_decision(goal_rgb, palette)` 只接受 goal RGB 和已保存实际 native goal-color RGBA；没有候选图/coverage/labels 参数。palette 取原 visible/white render 后的 snapshot，不 import pygame 查色表、不造新渲染。记录两种实际颜色的 exact 像素计数只是 provenance 诊断；无论计数多少，本版不认证分割，也不以候选效果改变分支。四词 goal-region/rest-region 聚合全部 null，不把空集合填成 0。

后备为全部 256 patch 按原序输出，无 top-k、无 argmin、无 heatmap 阈值、无“最佳 patch”。它不是 learned probe，不重新拟合 Q。此拒绝不代表不可能存在其它严格 mask；仅说明当前 RGB+固定 palette 证据不足以认证完整 goal-only foreground。本版不自行选择其它分割算法。

## 数据来源与严格绑定

仅接受下列真实三 SHA 的 v4 prepare/report/full strict；路径 CLI 显式传入：

| 输入 | SHA256 |
|---|---|
| prepare | `88e6403b2fd828dcfe08c803cc878fc948b112b2106aa95b4d4e64acf2f1475a` |
| report | `96c369c9b51301d902a489f0063754cce63007ca98aa5227fa54fc52ffc644e5` |
| strict | `542f2717e3860903d9ea87577edef2725a1fd7ae919c5aa216b45b75c0a29684` |

v4 source SHA 固定 `b9139d497cea2aaaaf680790d2bb4d8174c0df39c10f39d7cb2c30db6f8ac059`。调用其 **saved consume** 重放保存 journal/来源/模型身份/计账/全五 NPY；完整返回必须与外部 strict 相等，不运行其 runtime。继承全部原 bindings，加五 NPY、本实现三文件。四个旧 whole-word predicted 三 token NPY 取最后 token，不新预测。源照片、模型、normalizer/预处理及原 raw20 动作身份由 v4 的完整封存链保持。

坐标依据封存官方 DINOv2 `layers/patch_embed.py:PatchEmbed.forward` 的 Conv2d kernel=stride=14、`flatten(2).transpose(1,2)`，以及 `vision_transformer.py:forward_features` 的 patch token slice。原 loader `p5_jepa_official_action_fidelity_20261027_v7.py` 声明 patch14；原预处理是 224、scale/ratio=1、无 flip。保存并绑定这些源 SHA 和 preprocessing/crop 源。224×224 → row-major 16×16，每格名义原 RGB 14×14；边界 y0,x0,y1,x1 为半开区间。**输出 token 经全局 attention，不是只依赖其名义方格像素；贡献图不是局部因果解释。** 不 import DINOv2/torch/环境来实现坐标。

## 固定算术与全量输出 ABI

`words` 固定 xy、yx、x_minus_x、zero_zero，frame20 四分母。每词 `spatial` 三项：

- `observed_goal`：真实 frame20 patch features 对真实原 goal features。
- `predicted_goal`：旧封存全词预测最后 token 对同 goal。
- `predicted_observed`：同一全词预测对对应实际 frame20 features。

各项保存所有 256 个 `patch_mse`（每 patch 对全部 384 channel 的 float64 mean-square），以及 `uniform_full_mean_contribution=patch_mse/256`。全 patch 原始 float64 flatten mean 必须与 v4 对应标量 **exact** 相等。不同 reduction 顺序可能有舍入末位差；本版不 allclose，也不伪称 patch mean 求和 bitwise 等于 flatten mean。用 `Fraction.from_float` 明列：

`exact(reference_float64) = Σ exact(saved_contribution_i) + exact_reduction_residual`

residual 用十进制整数字符串 numerator/denominator 无损保存，属于算术记账，不是调阈值/校正科学度量。`contribution_sum_float64` 是 math.fsum 的独立可读显示，不替代原 scalar。

`pairs` 保留全部 12 有向对（6 无向对双向）。每项三组 256 个 `signed_patch_contribution`=左词 uniform contribution−右词，保留正负、不绝对值排序；同样保存左右原 full scalar 相减的参考值及 exact residual。`layout` 全 256 条，不按结果调整顺序或范围。

每词 `native` 保留公开 double object x/y/theta、原 goal pose、逐轴 x/y/raw theta 残差、atan2(sin(dtheta),cos(dtheta)) 主值、block velocity/angular velocity、完整 native7/其原 float32 object slice、proprio4、agent public、coverage/reward/done/完整 transition。double pose 必须 exact 对应 native info；不拿 float32 state 冒充 double。agent_goal 未定义则 null，不猜目标；object-only 残差不代表完整 PushT 失败。每对另列 object pose/error 各轴有符号差、wrapped error 的算术差（不是新角度距离）、coverage/reward 差，不拟合其组合权重、不造 success。

`mask` 的两区域/评分均 null；`protocol.new_Q=null`。输出为描述性任务度量差距定位，不会宣布哪一 patch 最好/新方法通过，不将四词当独立根或作显著性检验。

## 资源、失败与下一步边界

16MiB 输出 cap、200MiB free floor、120s saved-analysis wall（从显式外部 SHA/来源校验开始；包含 saved v4 consume、分解与最终 hash）。失败抛出并不写成功回执；v4 原四分母不删。只有全部完整输入才出新报告。目录为 results 的本 stem 唯一新 JSON，exclusive create+fsync，禁止覆盖 MAIN/v4。verify 重新绑定三 authority、重消费并全量重算报告，拒绝任意 patch/scalar/native/来源修改；输出无依时间变化的 runtime 字段。

这一步可供 MAIN 检查 uniform-Q 的空间分布与 native 各轴错位，但**不能从当前结果反选权重用于同批“验证”**。下一次真正 task-weighted 谱编译仍需单独事前固定：独立可认证的 goal 区域定义/坐标映射与遮挡处理、所有 patch 权重及归一化、缺 foreground 时 fail/null 规则、相同全四词及预算、模型加载封存与 full native 成功定义。若继续仅有当前 null goal 图，则本桥只支持原 uniform 全 patch comparator，不授权补画/藏物体/挑 patch；新加权合同仍未满足，不伪造可执行加权器。

## MAIN 可运行命令

在 le-wm-main 根目录；都是保存数据操作，不 launch 神经/物理：

```bash
.venv/bin/python -m pytest -q tests/test_agent_p5_taskmetric_patch_bridge_20261028_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p5_taskmetric_patch_bridge_20261028_v1 analyze \
  --prepare results/agent_p5_same_time_frame20_20261028_v4_prepare_MAIN01.json \
  --prepare-sha256 88e6403b2fd828dcfe08c803cc878fc948b112b2106aa95b4d4e64acf2f1475a \
  --report results/agent_p5_same_time_frame20_20261028_v4_cuda_MAIN01/report.json \
  --report-sha256 96c369c9b51301d902a489f0063754cce63007ca98aa5227fa54fc52ffc644e5 \
  --strict results/agent_p5_same_time_frame20_20261028_v4_consume_MAIN01.json \
  --strict-sha256 542f2717e3860903d9ea87577edef2725a1fd7ae919c5aa216b45b75c0a29684 \
  --output results/agent_p5_taskmetric_patch_bridge_20261028_v1_MAIN01.json
sha256sum results/agent_p5_taskmetric_patch_bridge_20261028_v1_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p5_taskmetric_patch_bridge_20261028_v1 verify \
  --saved results/agent_p5_taskmetric_patch_bridge_20261028_v1_MAIN01.json \
  --saved-sha256 ACTUAL_NEW_SAVED_SHA
```

未预填未来报告、权重或生产结果。测试里的随机 float32 fixture 仅测分解恒等式和拒绝路径，不是新科学样本。

## 交付验证状态

29 项 RAM 测试覆盖全部 patch/双向对/精确 rational 守恒、原 scalar 一 ULP 偏差拒绝、同帧/完整 native7/原 double pose 身份、颜色为空或存在时都不造 mask、outcome 改变不影响空间贡献或 mask、坏 dtype/shape/NaN/缺词、完整保存重算、重复 JSON keys/额外写入及输出 cap 拒绝。

实际只读调用 analyze（内存，不写 MAIN 回执）已完成：4 words、12 pairs、256 layout、序列化约 612408B，goal visible exact-palette 像素 0；torch/pymunk 均未导入。未运行神经或物理。该计数不提供 mask、encoder invariance、新 Q 或科学准入。持久化 saved analyze/verify 仍交 MAIN 按上述命令执行；三文件冻结后不原地改动。
