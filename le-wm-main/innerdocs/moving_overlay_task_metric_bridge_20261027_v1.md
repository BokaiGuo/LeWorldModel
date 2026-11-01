# Moving overlay task metric bridge：纯 saved 接线 v1

独占文件为同 stem 的 source/tests/innerdoc。本实现不修改任何 producer、冻结 RGB 三文件、共享 progress/dispatch/closure。没有训练、模型/CPU 神经执行、物理、prepare、网络、删除或发布。只实现与 RAM 测试；真实报告由 MAIN 完成 RGB 审读/运行后再执行本 bridge。

## 输入和完整分母

CLI 必须显式提供两个**已经存在**报告的实际路径与独立 SHA：

- 当前 `p1_moving_overlay_recompile_comparison_20261027_v1.report`：来自 MAIN 已验证的物理 comparison。
- `p1_moving_overlay_recompile_rgb_cuda_20261027_v1.report` 所在 output directory：调用该冻结 evaluator 的 `consume(output, report_sha256)`，只使用保存的原生 latent/Q，不运行其 run/prepare，不加载模型做计算。

comparison source SHA 必须等于 comparison 自身 binding.source_sha256；RGB consumer source 固定 SHA `70c03508d775b8b488c12133da929bae607d08d6163d5e1c6578edd9a2422bcc`。不存在真实 RGB report/SHA 时不能用 future/default/示例 SHA 执行。合法64位字符串仍必须与实际文件 SHA 相等，绝不把字符串本身当报告存在证明。

只把外部 SHA 绑定的现有 comparison 当物理验证提供者；不再次执行 comparison.material、不重新实现完整物理 journal 审计。RGB 的 own consume 会重验其 saved prepare/source/config/weights 等闭包及原始 receipt、重算 saved Q；可能读取/哈希保存的 weights 文件，但绝不调用神经模型、环境或 renderer。

固定两根 × off/on × `line_search_GN`/`full10_conditional_D2P` 共8臂，每臂 step0/5/10 共24显式 checkpoint slots。按 `(root_slot,condition,arm)` 建索引而非按输入列表位置；重复、缺臂或错误 selection_digest 立即失败。各 step 区分真实可用与 missing，不把合法早停的最后观察冒充 step10；missing 的 Q/latent/errors 均 null，不补零、不补图、不筛选有利样本。

## 实际数据关联

comparison.binding.files.physical_report 的**绝对路径和 SHA 同时**等于 RGB report.input_binding。两端 physical report 必须是同一真实对象，不按文件名猜测。逐臂匹配 selection_digest/condition/arm；own consume 每条 checkpoint 必须与原 RGB report 一致。

比较器没有保存原 state7，bridge 从 RGB 实际 checkpoint receipt 的 `physical.state/snapshot` 取 state7，以 comparison 原 `goal` 为目标。逐项验证 step/PNG相对引用/绝对路径/fileSHA/pixelSHA，state=snapshot.state，image.row 正确，latent=consume receipt，Q 为有限非负 float；四个真实 goal 按 slot/condition 单独关联，257×192 full-hidden schema 必须保留。每个声明的 checkpoint image 必须恰好消费一次，不静默丢弃或重复。

目标/根在同一 slot 的 off/on 和两 arm 间必须一致；有 step0 时，从实际 float64 `[state0_7,goal7]` bytes 重算 root SHA。没有 step0 则不制造初始状态，其变化值保持 null。原 comparison 的 initial/prefix5/terminal10 指标，均与实际 state/goal 重算的全部 native metrics 精确相等，否则拒绝合并。所有 bound mismatch 是接线/数据错误，不能写成科学负结果。

## 分解与比较语义

原 native state 顺序为 `[agent_x,agent_y,object_x,object_y,angle,velocity_x,velocity_y]`；delta 始终为实际 state−原 goal，不假设目标 velocity 为0。输出：

- agent/object/all position：原单位的 delta、squared L2、L2，未归一化或乘除渲染比例。
- angle：raw delta/absolute/squared、`atan2(sin(delta),cos(delta))` 的 signed wrapped/absolute/squared，及原 native term。
- velocity：相对真实 goal 的两维 delta、squared L2、L2。
- 原 raw squared total、wrapped squared total、完整既有 native metrics；这些只解释原指标的分项，不构造新权重或新科学评分。
- actual_latent_Q：来自冻结 own consume 的真实 native192 saved Q，预测 Q/residual 保持 null；不以像素距离或 state error 替代 latent Q。

原 native angle term `min(abs(delta),2π−abs(delta))` 可能在多圈时为负，原 success 不含 velocity 条件；bridge 保留二者并同时给 wrapped angle，绝不静默修正环境成功定义。角度、位置、速度与 latent Q 单位不同，禁止相加成新的 fitted/composite task score。

每臂给 prefix5−initial0、terminal10−initial0、terminal10−prefix5 的各指标差；另给每 root/arm 的 on−off、每 root/condition 的 D2P−GN。只有差两端都存在才给数值，否则 null。不会对不完整两根做有利子集均值。off/on 是**重新优化**后的描述比较，latent Q 各自使用真实 condition-specific goal 图，并非一个共同视觉目标的纯因果效应。

latent 改善不等价 task 改善：专门 RAM 测试构造 Q 降低但 object 位置误差增大，输出两个方向，不产生 task_pass。完整旧 comparison 原样嵌入 `physical_comparison`，其旧 actual_RGB/missing 字段不回写；新的 Q 在本 bridge.rows.checkpoints.actual_latent_Q，避免把新增证据伪装成旧报告本来就有。

## 输出、CLI 与资源

输出 schema `moving_overlay_task_metric_bridge_20261027_v1.report`，status `saved_task_metric_bridge_not_admitted`。保留原 comparison 全文、物理 errors/counts 和 planner nested cost receipt；嵌套资源不相加，不声称 equal-total-FLOPs。新 neural/physics/selection=0。输出单 JSON≤2MiB、写入前 free≥200MiB+2MiB，独占 `xb`、文件/父目录 fsync，不覆盖旧文件；输入 metadata≤2MiB，拒绝 symlink 或路径别名。own RGB consume 的 saved tensor读取沿用其32MiB有界合同。

MAIN 实际 RGB 落盘、核终态与 SHA 后，在 repo root 运行。所有路径使用绝对路径；三个环境变量必须来自实际记录，不可预造：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python -B -m experiments.spectral_reachability.moving_overlay_task_metric_bridge_20261027_v1 \
  --comparison /home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_overlay_recompile_comparison_20261027_v1_MAIN01.json \
  --comparison-sha "${COMPARISON_SHA:?MAIN independently recomputed comparison SHA256}" \
  --rgb-output "${RGB_OUTPUT:?actual completed RGB output absolute directory}" \
  --rgb-report-sha "${RGB_REPORT_SHA:?MAIN independently recomputed actual RGB report SHA256}" \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/moving_overlay_task_metric_bridge_20261027_v1_MAIN01.json
```

MAIN 若选择了另一 actual comparison 路径，应显式替换该参数与对应 SHA；不能按目录发现自动选最新。输出若存在则选新后缀，不覆盖。bridge 会在 own consume 前验证外部 SHA，在 consume 后重读两报告检查漂移，再完整合并后写 NEW 输出。工程失败抛错，不保存为已通过/已否定科学结论。

## 测试、冻结和限制

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_moving_overlay_task_metric_bridge_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.moving_overlay_task_metric_bridge_20261027_v1 --help
sha256sum experiments/spectral_reachability/moving_overlay_task_metric_bridge_20261027_v1.py tests/test_moving_overlay_task_metric_bridge_20261027_v1.py innerdocs/moving_overlay_task_metric_bridge_20261027_v1.md
```

35 RAM tests：完整8臂、重排输入按身份join、digest/condition/PNG/state/metric冲突拒绝、missing保null、wrapped/native angle及非零goal velocity、latent/task反向变化、future SHA前置拒绝、own consume接线、exclusive/floor/cap/fsync及CLI mock。联合冻结 RGB47 tests 与 physical comparison35 tests，117 passed。注入的 consume/CUDA字段仅测试协议，绝非运行凭证或科学证据；没有执行真实 material/consume/prepare/CUDA/physics。

P1=False、model_repaired=False、independent_confirmation=False、latent_improvement_implies_task_improvement=False。两个复用 known roots 不是独立新样本，无统计推广；不认证新的 physical root equivalence/reset bound/global freshness，不清除16旧阻断、不恢复未知 prediction、不推定 training stats来源。遵循 experiment-code 的实现/记录边界；按用户权限仅三文件和RAM测试，不生成额外图或神经实验。
