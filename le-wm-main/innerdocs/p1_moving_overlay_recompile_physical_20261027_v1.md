# Moving overlay 新编译八计划物理 consumer（冻结交接）

本文件仅说明实现与 RAM 回归，不是物理研究报告，也不授予 P1 admission。
仅新增同 stem source/tests/innerdoc；不修改旧执行器、共享进度或 MAIN 协议。
依 experiment-code 技能保留来源闭包、实际资源账、失败产物与可重现 CLI；本任务的 RAM-only 限制优先，不运行真实 prepare、神经或物理。

## 入口与封存

模块：`experiments.spectral_reachability.p1_moving_overlay_recompile_physical_20261027_v1`。
`material` 必须调用 Nash compiler 的 `consume(output, report_sha)`，完整八项通过才接受；随后还检查固定顺序、原始 float32[10,2]、selection/task/blind/source/planner 身份。
接受 producer 的额外字段，但仅投影明确的十个执行字段；完整 handoff 另存摘要。若提供 with_target、selection_uses_physics 或 P1_admission，必须符合本合同。

顺序为 slot0/off/GN、slot0/off/D2P、slot0/on/GN、slot0/on/D2P，再 slot1 同序。不存在部分计划释放、失败重选、动作修复或 outcome 筛选。

来源固定为 `results/p1_moving_target_overlay_source_20261027_v1_main01/report.json`，SHA256：
`85ca1df3de910628bfd6c5bd1a0a482f61880a2ee76bfd404732a77dd8c7666a`。
读取原 paired prepare/bindings、journalchain、真实 warmup 图和四个 blind NPZ；不调用原 execute/episode/witness，不猴补旧 globals。
`own_files` 绑定本三文件、实际复用 helper 源码及外层只读 `docs/D2P_MOVING_OVERLAY_RECOMPILE_PROTOCOL_20261027.md`；compiler 文件也进入 prepare bindings。
prepare/verify 均重建完整 material；run 在外部 prepare SHA 验证后再次验证全八 consume。依赖变化应停止，由 MAIN 审核后使用新的独占 prepare/output 路径，不能改封存物。

## 真实执行语义（待 MAIN 运行）

每项全新独立 metered env。调用原 placement 的两个 reset 与原 warmup10；每个 warmup0..10 都检查 off 图的文件/像素 SHA、原图精确相等、七维状态及完整 body snapshot，与 paired source 原身份闭合。
constructor 的 attempt/return 包围 factory；native hooks 在 factory 返回并验证环境后安装，不覆盖 constructor 内部。当前实际安装 PushT 的 __init__ 无 reset/space，按 MAIN 已读源码和固定依赖 SHA 约束；不得将此实现描述为 constructor 前安装 hooks。
0/5/10 三张 history 与对应 condition blind packet 精确相等；root/goal 身份哈希精确匹配。其后仅一次 goal bind，使用同 condition 真实 witness goal RGB，不修改原生 goal_pose。

控制始终 with_target=False。每个 warmup0..10、plan0..10 均采真实 off/on/off triplet，切换前后检查物理、goal、goal_pose、renderer 及 off 像素不变；选择当前 condition 的图作为本项标签。正常全程 66 图/项、528 图/八项。on 默认轮廓不是 task goal。
执行 sealed raw10，不归一化、不裁剪、不替换、不补步。保留原生 success/reward/termination 检查。native stop 可合法提前终止，工程失败则记原错误；均保持八项分母，不生成缺失 checkpoint。
每次返回先保存 transition/真实图，再作 action/状态守卫，尽量保留事故现场。

资源计量记录 attempted/completed、journalchain 和失败；完整正常路径上限：

| 资源 | 上限 |
| --- | ---: |
| constructor / close | 各 8 |
| reset / public_step | 16 / 160 |
| physics_step / set_state | 1632 / 32 |
| goal_setter / goal_render / goal_bind | 24 / 16 / 8 |
| explicit_render | 528 |

固定输出上限 64MiB、final reserve 2MiB、磁盘 floor 200MiB、墙钟 900s（包含 run 内 verify）。每次派发前检查预算；超时/磁盘耗尽后仅非物理 close 清理仍尝试一次，日志不可写时记录 secondary error，不继续物理。最终超时标记失败。不是抢占式硬超时：已进入的原生调用不能被安全中断，清理/收尾也可能越过 900s，但不能因此报告成功。
输出目录和 prepare 均独占创建，fsync；不覆盖/删除旧文件。磁盘或 final report 写失败可能导致无完整磁盘 report，此时返回 partial/error，不能按成功消费。

## 输出接口

`report.json` 顶层稳定字段：schema/status/rows/errors/prepare_sha256/planner_report_sha256/source_report_sha256/counts；附 journal_chain_tail、journal_sha256、execution_binding 及其文件 SHA、fixed_denominator=8、P1_admission=false。
每 row 至少有 root_slot/condition/arm/selection_digest/status/checkpoints/native_success/missing_checkpoints；完整保存 trajectory、attempt/return、首错与 secondary_errors。
checkpoint 只在实际 plan step0/5/10 成功采图后写入，字段为 step/png/file_sha256/pixel_sha256/state/snapshot，state 是真实七维。
未评 native_success 为 null，缺失 checkpoint 保留 missing_checkpoints，不填 0。`moving_overlay_recompile_physical_recorded` 只表示执行记录合同完成，允许合法 native stop/缺失 checkpoint，并不表示任务成功或 P1 通过。

供下游 comparison/RGB consumer 读取 report rows、相对输出目录的 png、checkpoint 哈希与真实七维 state/snapshot；本 consumer 不调用神经评分，不能将图像存在等同于 latent Q 已评。MAIN 后接新 CUDA RGB 评价，不等待该实现而延迟此交付。

## MAIN 准确 CLI

在 `/home/gbk/code/LeWorldModel/le-wm-main`，使用 MAIN 审核后的 HOST 环境。以下变量必须由 MAIN 填入实际绝对路径和实际外部 SHA；此处不预造未来 SHA。prepare 的父目录必须已存在；prepare 路径及 physical output 目录必须未存在。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_overlay_recompile_physical_20261027_v1 prepare --planner-output "$PLANNER_OUTPUT" --planner-report-sha "$PLANNER_REPORT_SHA" --output "$PHYSICAL_PREPARE"
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_overlay_recompile_physical_20261027_v1 verify --prepared "$PHYSICAL_PREPARE" --prepare-sha "$PHYSICAL_PREPARE_SHA"
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_overlay_recompile_physical_20261027_v1 run --prepared "$PHYSICAL_PREPARE" --prepare-sha "$PHYSICAL_PREPARE_SHA" --output "$PHYSICAL_OUTPUT" --authorize-physics
```

`PLANNER_OUTPUT` 是 Nash 实际八计划 output 目录；`PLANNER_REPORT_SHA` 是 MAIN 提供的实际 report.json SHA。
首条输出 `prepared/prepare_sha256/bytes`；MAIN 核对实际文件后将其 SHA 作为 `PHYSICAL_PREPARE_SHA`。run 无 `--authorize-physics` 不执行。若返回 partial_or_failed，CLI exit1；异常也失败，不应绕过 consume 或换动作。

## RAM 回归与限制

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_moving_overlay_recompile_physical_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_overlay_recompile_physical_20261027_v1 --help
```

测试使用纯 RAM 人工 Env、PNG/数组、metadata callbacks；不导入 torch/gymnasium/pymunk/stable_worldmodel，不实际 prepare。覆盖八计划、七维接口、528 图与资源账、精确原始动作、condition/来源错配、native stop、不补零、事故留存、hook 恢复、清理/输出失败、资源限制、外部 SHA/全八 consume/verify 重建及协议绑定。
本次最终回归：56 passed in 19.38s；CLI --help exit0。测试提速仅优化 RAM journal 缓存/哈希，不改变生产路径预算或阈值。
这些测试不是科学证据。真实 source closure、native API 运行等价、Nash 实际 handoff、HOST 物理、真实输出体积/耗时和下游评分仍待 MAIN 审核执行；无真实运行报告或新的 P1 通过证书。两根仅 known calibration roots，不是独立 fresh roots。
