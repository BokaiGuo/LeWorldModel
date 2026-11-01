# Q 一致 conditional full10 固定 H2：独立 guarded physical consumer

只新增本 consumer、测试、本文和独占 prepare/bind；不改 compiler v1、其输出或任何 sealed 旧文件。本轮没有运行神经模型、物理或下载。真实执行仅 main 在 review 后通过显式 `--execute-physics` 进行。四根仍是 calibration；P1/P3/P4 admission 为 false。

## 来源与自身 schema

唯一 planner 来源是已封存 `p3_goal_patch_consistent_compiler_20261027_v1.selected_actions`。compiler prepare SHA 为 `2b32137d1e7facf79335b25c6e6acf976ed10162073b938bff1262b385ba1425`，源码 SHA 为 `006398e5aaf0f6e901a01c2d6779a977c31875413b967fec552245332002396d`。不假接后续 v2，也不使用旧 physical/replay endpoint。

实际 main compiler report 路径为 `results/p3_goal_patch_consistent_compiler_20261027_v1_cuda_main_v1/report.json`，只读核验 SHA `64fd87f48f73674d98b8eb2ddce7415507bf1e6d457df8e78ad91da1c15c4fe1`。其完整成功导出才允许 report-bind。全部八条新计划均保留，包括动作相同的 arms。

本模块自己的 `.protocol/.plan/.branch/.physical_report/.encoding/.report/.compiler_history_audit` schema 均使用 `p1_goal_consistent_conditional_physical_20261027_v1` stem，不把旧 schema 换标签。`plans_from_exports(protocol,exports)` 独立验证 compiler export digest、精确 root/arm census、原 goal0/context、active metric/Q、model bindings、protocol/report/input/selection hash，并生成自己的八条 physical plans。每条固定 H2、10 个原 raw controls；不 clip、补齐、重选、去重或用旧预测 endpoint 代替物理终点。

## v1 成功数据的独立 history 内容 gate

v1 exporter 的 action-history shape 检查不当作内容证明。本 consumer 在 bind/verify/执行前后新增独立 `verify_compiler_histories`：

- 用 AST 读取封存 `experiments/run_pusht_counterfactual.py` 的 public mean/std 字面量，核对既有 warmup normalization；不 import 模型。它们是当前公开实现的来源，不证明训练时 normalizer provenance。
- 从每个 raw packet 按 time-major `float32 [5,2]→[10]` 独立计算 `(raw−tile(mean,5))/tile(std,5)`；无 `/100`、repeat-interleave、transpose 或旧 token 替代。
- 用纯 `_context` 加严格 metadata/schema/dim/type/完整 keyset 检查，重建最后两段 action history，要求内容及顺序精确一致；检查保留的原 latent chronology，不宣称从数组重新生成神经预测。
- 覆盖全部 **80 candidate contexts、24 winner contexts、104 completed replay contexts、8 selected exports**，不只检查 winner。逐 packet replay 的第二段以实际第一段完整 context 为初始状态。

以上对实际 v1 saved 数据已只读通过；它不是修复/否定 v1 的异常处理窗口，不冒充 CUDA 重新运行。main 独立 `p1_conditional_history_arrays_main_20261027_v1` 的 112-check 报告是另一路证据，本 consumer 选择自有独立字面量/内容重算 gate，而非仅信任那个报告的结论。

## 四根复原、guard 与指标

source prepare 认证原四根/root→index、相同 model bindings、相同原始独立 goal0、旧 Q 与 source input SHA；复用 `p3_horizon_residual_physical_20261027_v4_source_prepared01` 的 source references/seed，而不是其物理结果。

main 创建原 `swm/PushT-v1` 后核对 source-sealed installed official evaluator 与 RoyalBlue agent。原 collector count=4、oversample=4、原 seed 不变，四个完整 state/goal fingerprints 必须 upfront 全匹配。每条分支独立 `_reset_and_warmup(history_len=3)`，原 15 个零 raw warmup steps 不变；核对三帧完整 RGB stack、goal RGB SHA、归一化 incoming between-state tokens、完整状态和 goal，两臂 initial identity 完全相同。warmup/collection 与 selected-path 分账。

只复用 `durable.physical_branch` 的低层逐步 executor：初始以及每一 raw command 后真实 RGB，RoyalBlue guard 决定是否允许下一 command。state/goal/pose/official 仅作为 evaluator，不参与动作/guard/replan。每次 attempt、env.step return、观察、PNG/hash、异常均 fsync 留存；guard-partial、环境终止、reset/render/step/evaluation failure 全保留，未开始臂显式 blocked。一次完整 selected path 恰好 10 commands，不以 H3 或旧终点冒充。

`verify_branch` 核对原动作、计数、journal/trace/PNG/selection identity，并用保存的 RGB 独立重放 guard 状态机，禁止 abstention 后继续 dispatch。terminal freshness 必须对应最近真实返回步的真实保存图片；step 或 render 失败后的旧图严格禁止编码。初始或中间 guard partial 的 fresh endpoint 可以编码，但明确标 partial，不混入 full-H2。

两臂使用同一 installed official evaluator 和原 `xy_distance +20*angle_distance` pose metric。只有完整同 root/初始 context、同 H2 的配对才给 `new_minus_old_pose_cost` / `new_minus_old_official_distance`。partials 分类 `partial_same_steps` 或 `partial_or_unmatched`，不补齐、不混入完整比较；不按有利根筛选。

## 严格预算与编码

| 项目 | 上限与实际账 |
|---|---|
| selected path | 8×10=**80 commands**；每臂 cap10、全局 cap80；attempt/return 分开、不退款 |
| 原 warmup | 8×15=**120 steps**；每臂 cap15、全局 cap120；与 selected 分开 |
| root collection | 原 collector；step cap0，reset attempts/returns 独立实记，不混进 80 |
| endpoint visual encoder | 可选 HOST CUDA；fresh endpoints 各一次，cap8、B1 输入 `[1,3,224,224]` |
| existing projector | 独立 attempt/completed 账及 cap8，输入 `[1,192]`；不是隐藏在 encoder 数里 |
| predictor / action encoder | endpoint 阶段禁止；次数均应为0 |
| 新 Jacobian/SVD | consumer 无模型 AD；compiler exporter 的 saved-array 校验每调用96 SVD，另计 CPU provenance |

先完成并封存所有 physical branches 与 `physical_report.json`，才允许 CUDA model setup。冻结模型字节/配置必须等于原 planner，所有 modules eval、parameters frozen、buffers/parameters 同 CUDA device。用原 image transform 和 `model.encode` 得 native projector(CLS) 192 latent；**每个实际 endpoint 编码一次，同一个 latent 同时计算 old_Q / goal_patch_Q 两个 costs**。active Q 不改变另一臂的物理 metric，也不重新选动作。不同 Q 的 scalar score 不直接相减作同量纲效用收益。

encoder/projector pre-dispatch attempts/completions/blocked/failed 独立 durable journal。编码设置失败/坏 shape/坏 RGB hash/部分模块完成均保留，不自动重试。最终 hooks restore 每项独立捕获异常，不能覆盖首错或阻止后续恢复/report。模型未加载时 `model_unchanged=None`、`model_hash_verified=False`，不当作复核通过；加载后比较真实 tensor-state hash。最终 source/hash/model 异常另加入 failures，先前 physical/encoding error 保留在 row/encoding receipts。

历史 compiler 成本 528 predict+528 action-encoder、88 Jacobian、96 SVD 是 inherited，不再执行或称免费；Q 构造 80 encoder+80 projector 也保留历史成本。含 warmup/collector/历史 acquisition 的全 pipeline 不声称等成本。

## prepare、bind 与 main 命令

`--prepare` 只读取源输入与 AST/source closure，冻结 new source/tests/本文及静态可达 installed evaluator/model 包，不载入模型/环境。`--bind` 使用真正 compiler report SHA 与 `selected_actions`，冻结全部报告、arm/root JSON、journal 哈希和本 consumer 的独立 history 内容证据；source-only 和 bound 分别独占新目录，后者不修改前者。

交付消息给出 source/bound prepare.json 的确切 SHA。main 执行使用 **bound** receipt：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m experiments.spectral_reachability.p1_goal_consistent_conditional_physical_20261027_v1 \
  --verify --prepared results/p1_goal_consistent_conditional_physical_20261027_v1_bound \
  --prepare-sha256 BOUND_PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.p1_goal_consistent_conditional_physical_20261027_v1 \
  --execute-physics --prepared results/p1_goal_consistent_conditional_physical_20261027_v1_bound \
  --prepare-sha256 BOUND_PREPARE_SHA \
  --output results/p1_goal_consistent_conditional_physical_20261027_v1_main_v1 --encode-terminals
```

source prepare 目录使用 `_source_prepared` 后缀。若将来另 bind，只能原 source prepare 加 `--bind --compiler-output ... --report-sha256 ... --output NEW`；不接受伪旧 report 或未存在 v2。

## 本轮验证与边界

测试仅 fake env/NumPy/接口与实际 saved v1 arrays：完整/guard partial/failed-step/render/termination、H2/hash/动作越界预拒绝、双 Q 同 endpoint、module caps/失败不退款、80/120 分账、cleanup 首错保留、未载模型不标复核、全候选 action-history/layout 内容与非 winner 篡改拒绝。fake-host 只证明协议/账本，不构成神经预测或物理有效证据。

预计 source+bound prepare 合计约数十 MiB；physical PNG/journal/report 预留64 MiB，CUDA 外部缓存另预留200 MiB较稳妥。没有删除/下载/训练。仍需 main review+真实物理/端点编码；四根 calibration 不能独立准入，P3 dynamic/H3 范围仍是后续独立任务。
