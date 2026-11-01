# P2–P5 强基线 saved-only 资源账本与 preflight

日期：2026-11-01。该侧线只新增独占源码、测试和本文；未修改 P1、其他代理文件或 sealed artifacts，未运行 CUDA、物理、训练或神经 CPU fallback。

## 交付结论

新增统一入口：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m experiments.spectral_reachability.baseline_pstages_saved_ledger_20261101_parallel inventory
.venv/bin/python -m experiments.spectral_reachability.baseline_pstages_saved_ledger_20261101_parallel preflight --manifest MANIFEST.json
```

它只读取固定 SHA-256 的本地报告，按 `encoding / prediction / differentiation / search / physical / timing / memory` 分类保存资源轴。每个操作只能是：

- `measured`：报告直接计量；
- `derived`：从报告内固定行或事件计数推导；
- `declared`：报告合同明确声明；
- `absent_by_contract`：合同明确不存在，允许值为 0；
- `unknown`：未计量，不能转成 0。

preflight 对未知操作、未知资源、报告哈希漂移、协议 cohort 混用和跨协议排名 fail-closed。它不把嵌套 API、样本、VJP、wall 或显存相加成 FLOPs。

## 当前本地资产

|阶段|账本 ID|身份|当前可核资源|证据等级与边界|
|---|---|---|---|---|
|P2|`p2_zero_h3`|zero control|明确无神经/搜索；public reset/step、terminal RGB|四个已知根物理控制；`total_resource_matched=false`|
|P2|`p2_no_composition_sra`|本地无组合谱规划|action encoder、predict、context token、selection、J/VJP/SVD、peak allocated|四个校准根 HOST CUDA；upstream RGB、wall、FLOPs unknown|
|P2|`p2_gn_h15`|本地 GN|顶层 end-to-end wall、peak allocated/reserved|两已知根；顶层只绑定子报告哈希，不据此虚构调用账本|
|P2|`p2_cem_h15`|本地 CEM|顶层 end-to-end wall、peak allocated/reserved|与 GN 同 cohort；尚非全轴公平|
|P2|`p2_grasp_h1_local`|本地 proxy|action encoder、predictor、context token、scalar backward；物理明确未执行|`official_equivalence=false`；继承 RGB 编码成本 unknown|
|P3|`p3_fixed_schedule`|固定 horizon|predict/action encoder/context/search 事件、runtime backward/VJP 为 0、总 elapsed|八个已知 calibration rows；dense Jacobian 字段为 null，非 per-arm wall|
|P3|`p3_action_jacobian_schedule`|动态调度|同上|继承 Jacobian 获取成本明确 unknown，不能称总 pipeline 等价|
|P3|`p3_reacher_cem_feedback`|本地 CEM feedback|encoder/action encoder/predict samples、physical steps；导数明确不存在|单已知根 fixed/dynamic；`equal_actual_compute=false`|
|P4|`p4_feedback_pair`|本地 feedback pair|encoder/projector/action encoder/predict/grad/J、public reset/step、wall、peak CUDA|单已知根 saved audit；`P4_admission=false`|
|P5|`p5_architecture_readiness`|readiness only|无可比较运行资源|静态 readiness；`cuda_executed=false`，不是基线结果|

所有 entrypoint 文件当前存在；“可运行”表示已有本地入口和 pinned saved report，不表示已授权重跑，也不表示完整依赖、独立根或公平预算已满足。

## preflight manifest

严格 JSON：

```json
{
  "schema": "baseline_pstages_saved_ledger_20261101_parallel.preflight.v1",
  "asset_ids": ["p2_gn_h15", "p2_cem_h15"],
  "required_categories": [],
  "required_operations": ["end_to_end_wall", "peak_cuda_allocated"],
  "comparison": {
    "requested": true,
    "ranking": false,
    "protocol_cohort": "p2_h15_matched_host_two_known_roots"
  }
}
```

`required_categories` 要求该类全部操作有覆盖；若只需要少量轴，应放入 `required_operations`。同 cohort 只解除“协议身份明显不同”的阻断，不自动证明 equal compute。任何 `ranking=true` 都不会被该 saved-only 工具授权。

作者报告值没有进入 inventory 或 manifest。GRASP、Hi-LeWM、LeWM、PlanWAM 的论文数值继续只保留在 `paper_reported/context_only` 文档和 registry，不能与这里的本地值排行。

## 仍缺的依赖与证据

1. P2 GN/CEM 需要一个绑定全部子报告字节、分轴 API/样本/导数/search/物理及 per-arm wall 的统一 receipt；当前顶层报告不足。
2. P2 GRASP 的 inherited RGB/goal encoding、完整 wall/FLOPs 与官方协议等价性仍 unknown；当前只能称 local proxy。
3. P3 动态调度的继承 Jacobian 获取成本、per-arm wall、FLOPs、energy、model load 和 dynamic source closure 未闭合。
4. P4 只有单已知根 saved audit，且没有独立 CUDA/导数 replay；不能升级为 P4 机制或泛化证据。
5. P5 尚无同协议执行资源和性能结果；readiness 不能替代本地基线。
6. 所有后续公平比较仍需冻结共同资源合同、反馈机会、失败计费和独立根来源。该工具只认证 saved coverage，不认证公平、效果或 P2/P3/P4/P5 admission。

## 证据等级

- `saved_host_cuda_*`：已有主机 CUDA 报告，但本工具只复核保存字节和字段。
- `saved_physical_*`：已有物理保存结果，但不是本轮新执行。
- `saved_audit_*`：已有保存审计，未在本轮独立重放模型/物理。
- `static_readiness_*`：静态依赖/架构检查，无神经效果证据。

因此本交付的最高结论是：**后续 P 阶段已有统一、可执行、fail-closed 的本地 saved-only 资源覆盖入口；尚未证明任何跨方法公平性、性能排名或阶段通过。**
