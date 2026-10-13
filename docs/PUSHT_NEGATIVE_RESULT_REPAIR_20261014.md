# PushT 负结果修复记录（2026-10-14）

## 根因分解

本批次把“负结果”拆成三类，而不是继续统一调参：

| 层 | 证据 | 判定 |
|---|---|---|
| 数据层 | 原有 `pusht_expert_train.h5.zst` 解压时报 `Data corruption detected`；镜像重新获取后成功产生 46,300,921,856-byte HDF5。 | 原始数据 artifact 损坏，相关旧失败不能直接归因于模型。 |
| 接触接口层 | 旧 signed-torque audit 的 center-radius root 不是 T 形物体真实 boundary contact；旧 chart 给出 direction accuracy `0.870`，但闭环 contact/pose 失败。 | 属于协议伪正结果，不能保留为因果证据。 |
| 动力学/规划层 | 修复真实 boundary chart 后 action ranking 仍有局部收益，但 terminal pose success 仍为 0；contact-stratified intervention 显示 normal/tangential mode 内效果不稳定。 | 剩余瓶颈是 within-mode action identifiability、候选支持和终端姿态闭合，不是单纯 memory 长度。 |

## 已运行结果

### 接触分层 paired intervention

`pusht_contact_stratified_paired_intervention_20261014` 使用四个 root-disjoint
strata（free、approach、normal_contact、tangential_contact），每个 strata 64 个
train roots、32 个 held-out roots，且每个 root 使用同状态 `+a/-a/zero`。

- specialist geometry 在 free/approach 明显优于 state/action baseline；
- normal_contact 和 tangential_contact 没有稳定优势；
- 结论支持“mode observability 与 within-mode action identifiability 是正交瓶颈”。

### 真实 polygon boundary signed-torque audit

`pusht_signed_torque_audit_boundary_corrected_20261014` 使用 64/32 root-disjoint
roots、真实 polygon boundary contact frame、4-step pulse 和 24-step closed loop。

- direction accuracy：`0.609`，未达到预设 `0.70`；
- classifier regret：`0.0151`，仍低于 geometric regret `0.0265`；
- angle improvement：`+0.124 rad`，95% CI `[-0.078, +0.324]`；
- strict pose success：`0/32`，因为 radial-contact roots 不是 terminal-pose XY roots，不能用该数宣称姿态闭合。

这说明 action ranking 的局部可用性不足以保证闭环姿态闭合。

### RGB-only repaired predictor MPC

在同一 8 个 held-out episodes 上，frozen visual encoder + repaired predictor 的
结果为：

| protocol | contact rate | success | block XY | final distance | endpoint pose error |
|---|---:|---:|---:|---:|---:|
| base, action block 2 | 1.0 | 0/8 | 172.74 | 507.27 | 262.31 |
| repaired, action block 2 | 1.0 | 0/8 | 159.47 | 204.62 | 126.67 |
| repaired, action block 1 | 1.0 | 0/8 | 145.38 | 203.83 | 128.17 |

这是一条真实但有限的正向链：predictor repair → lower local endpoint/XY error；
它没有通过 predictor repair → terminal pose success。

## 当前方法论调整

正式控制接口必须按以下顺序释放动作：

```text
RGB/latent observation
→ actual-boundary contact chart
→ mode/support gate
→ signed action-effect prediction
→ short pulse
→ RGB re-observation
→ contact and transition consistency check
→ replan or fail-closed fallback
```

若 contact/mode gate 或 action-effect uncertainty 不通过，系统只能拒绝 learned
pose action，不能以 mode accuracy 或 latent MSE 代替 torque sign 证据。
