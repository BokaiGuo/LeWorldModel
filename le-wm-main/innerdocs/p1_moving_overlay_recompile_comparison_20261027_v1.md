# Moving-overlay full-recompile：saved 物理比较 v1

本模块只读 NEW compiler / NEW physical 产物，固定分母为两根 × off/on × `line_search_GN`/`full10_conditional_D2P`，共八项。不是旧命令换渲染图的比较，也不是另一个 planner、freshness resolver 或科学准入器。MAIN 负责审读及实际执行。此交付仅实现与 RAM/mock 测试，不创建实际 prepare/result，不调用神经模型或环境。

## 已对齐的 producer 合同

直接依据当前两个 NEW 源码的 saved 接口，不导入或执行它们：

- `p1_moving_overlay_recompile_cuda_20261027_v1`：report 状态 `overlay_recompiled_plans_sealed_not_admitted`；八个 plan JSON 的 `selection_digest` 是去除自身字段后、`json.dumps(sort_keys=True, allow_nan=False)` 的 SHA256。physical prepare 的 `compiler_handoff_sha256` 绑定完整八 plan 加外部 `planner_report_sha256` 后的有序列表。
- `p1_moving_overlay_recompile_physical_20261027_v1`：plain JSON prepare；report schema 为该 stem 加 `.report`，状态为 `moving_overlay_recompile_physical_recorded` 或 `partial_or_failed`。固定八 rows；以 `(root_slot, condition, arm)` join，不靠 report 数组顺序。
- journal 的 `run_started.plans` 等于 prepare plans。`branch_closed` **不是整条 row**：只比较身份、selection、branch index、status、own_returned、first_error。`fresh_observation.receipt` 含 phase/condition/overlay_index，而 checkpoint 仅含 step/png/file_sha256/pixel_sha256/state/snapshot：比较共享字段，不要求整字典相等。
- 已接受 trajectory 必须有完整 off/on/off 三帧、pre/post 全 invariant 相同且两张 off 的 PNG/pixel SHA 相等；chosen 标签 off=0、on=1。未完成 capture 的 raw journal 证据可存在，但不能替代已接受 checkpoint。

当前工具没有跨 agent 发送消息接口；以上具体合同由 MAIN 转发 Bohr 即可，若 producer 改 schema，应新版本对接，不能用配置任意改 tag 伪装兼容。

## SHA 与目标来源

输入是一个由 MAIN 填写并独立 hash 的小 binding JSON，精确两键 `schema`/`files`；schema 为 `p1_moving_overlay_recompile_comparison_20261027_v1.binding`。`files` 必须恰好包含下面十键，每项为 `{"file":"绝对实际路径","sha256":"真实64位SHA"}`：

| 键 | 真实文件 |
|---|---|
| physical_report | NEW physical output/report.json |
| physical_prepare | physical 实际使用的 plain prepare JSON |
| physical_execution_binding | 同 output/execution_binding.json |
| physical_journal | 同 output/journal.jsonl |
| physical_source | NEW physical `.py` |
| planner_report | NEW compiler output/report.json |
| planner_source | NEW compiler `.py` |
| overlay_source_report | `results/p1_moving_target_overlay_source_20261027_v1_main01/report.json` |
| original_sourcepack | `results/p1_constructive_moving_history_source_20261027_v1_main01/sourcepack.json` |
| original_source_journal | original sourcepack 同目录 journal.jsonl |

没有实际 physical/planner SHA 时不得虚构 binding 或运行。sourcepack 还固定原 authority SHA `c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6`；overlay report 的已知真实 SHA 是 `85ca1df3de910628bfd6c5bd1a0a482f61880a2ee76bfd404732a77dd8c7666a`，仍必须在 binding 显式写入并验证，而非作为未来结果默认值。

目标来自 **original sourcepack 的 construction journal 最后 native_return.state**，初始来自同根 replay0 step10 fresh_observation。按原 float64 `[initial7, goal7]` bytes 重算 root SHA。overlay `blind_exports[].old_root_sha256` 再将 slot/condition blind NPZ 与原根绑定；NPZ 文件自身 SHA 实测，不从终点标签或新 latent 生成目标。

验证 external 文件 SHA、prepare 内上述依赖 SHA、execution binding 文件及 report 副本、journal 文件 SHA/tail/全链、run start prepare、八个 plan 的 semantic seal/manifest/handoff digest、raw commands 与 actual dtype/bits、所有 fresh PNG 和 decoded RGB224 pixel SHA。原始 warmup 固定前五 `[.1,0]` 后五 `[0,.1]`，float32；无第二次 `/100`、裁剪或动作补全。

边界：这里不再次运行 compiler.consume，也不重新遍历全部 compiler tensor/候选数组/模型权重；MAIN 应先按 compiler 的 own consume 核实其全输出。本模块核的是 source→frozen compiler commands→physical saved chain 和标签算术，不冒充第二次完整神经 trace 认证。producer helper 全闭包的既有执行验证由封存 prepare/report 承载；这里显式复核的源码为两个外部指定 `.py`，不执行其中代码。

## 指标与缺失语义

每分支给 initial0/prefix5/terminal10，未出现的 checkpoint 为 null，原 errors/first_error/secondary_errors 与 attempted/returned 留存。合法早停不补跑，不把最后可用图/状态冒充 step10。两个根不按好坏筛选；两根平均差仅在两根都有终点时给出，否则 null。

- 原 native distance/reward：七维欧氏距离及其负值，含速度；native success 仅 `norm(diff[:4]) < 20` 与原 angle 项 `< pi/9`，**不含速度条件**。
- 原 angle 项 `min(abs(diff[4]), 2*pi-abs(diff[4]))` 原样保留，多圈时可能为负。独立 wrapped angle 用 `abs(atan2(sin(delta), cos(delta)))`，同时报 wrapped success、object XY、agent XY、velocity L2、wrapped joint L2 和 native/wrapped 分歧。不能将 2π 表示差当成几何错位，也不能静默改 env 指标。
- on−off 每 arm/root 的真实终距及 native success 差，属于 **重新优化后策略的 paired 描述比较**；不是纯 observation 因果效应。prefix distance5−distance0 与同 condition D2P−GN prefix 差提供可观测的前缀行为。没有 raw-zero 同根前缀控制，严格“相对 raw-zero 非贪心”保持 null。
- v1 没有新 RGB encoder 产物输入：所有 actual latent Q 为 null / `missing_not_encoded`。像素 SHA 验证不等于 latent Q 评价，更不能把模型 Q 当成真实收益。

## 全账与执行

保存 compiler contract、逐分支 resources/AD_API_resources，不声称同 FLOPs。对 physical journal 的 operation_attempted/completed 独立计数，保留失败 attempt，并核每分支原始动作 bits、返回数及 PNG 分母。完整状态还验证 producer census 恒等式。

物理上限是 **80 selected + 80 warmup = 160 public steps**、16 reset、内部 physics 1632、528 显式 render；不是“总共80步”。合法早停可低于上限；不得靠 padding 凑账。此比较器 new neural/physics 为零，输出单个 JSON 上限2MiB、写入前保留200MiB+2MiB；独占 `xb`、fsync，不覆盖旧文件，不复制上游大产物。

MAIN 填写并 hash binding 后，在 `le-wm-main` 执行（`$BINDING_SHA` 是实际 binding 文件 SHA，不能照抄占位符）：

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_overlay_recompile_comparison_20261027_v1 \
  --binding /absolute/NEW_comparison_binding.json --binding-sha "$BINDING_SHA" \
  --output /absolute/NEW_comparison_report.json
```

CLI 在完整 read/verify/算术完成后才独占创建输出；绑定/工程失败抛错，不写成科学负结果。输入损坏与合法缺失不同：缺失 checkpoint 保 null；SHA/身份/动作/链不一致拒绝比较。

## RAM 测试与结论边界

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_p1_moving_overlay_recompile_comparison_20261027_v1.py
```

所有 PNG、链、NPZ 字节占位、report 和 CLI I/O mock 都只在 RAM；mock 不代表真实8分支成功或实际 journal 已通过。代码不导入 producer、torch、环境或神经 loader。不会生成 pytest 大临时 fixture，也不创建 prepare/results。

实际比较待 MAIN 真实 compiler/physical 报告及外部 SHA 后运行；known moving 两根复用而非独立新根，P1=false、模型修复=false、独立确认=false、equal-total-FLOPs=false。即便0/8 native success，也不能推出理论或组合机制否定。
