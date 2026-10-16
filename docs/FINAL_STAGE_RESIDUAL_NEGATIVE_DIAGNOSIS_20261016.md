# TwoRoom final-stage residual 负结果诊断与 action-support 修复（2026-10-16）

## 结论先行

本轮 CUDA 闭环复核没有证明 final-stage residual 能带来稳定规划收益。它确实改变了 CEM 的候选代价和最终动作，但在两个全新的 32-root panel 上，逐 episode success 向量与 frozen Gaussian baseline 完全一致。

真正得到的可复现修复是 action-support intervention：在同一 frozen LeWM 和同一 CEM 预算下加入预先声明的 candidate bank，并保持 Euclidean frozen cost。相对纯 Gaussian baseline，candidate bank 在两个独立 panel 上均有正向 paired 差异。

## 代码与协议修复

- `experiments/run_live_tworoom_planning.py` 修复了 final-stage residual 运行完成后写 `planning.json` 时引用局部变量导致的 `NameError`；artifact 的 `residual_start_iteration` 现在由顶层 driver 显式记录。
- `tests/test_capacity_readout_runtime.py` 新增 final-stage gate、起始 iteration 和缺失 iteration metadata 的回归测试；相关测试共 `15 passed`。
- runner 增加默认关闭的 `--record-solver-trace` 诊断开关。trace 仅用于验证 residual 是否改变动作，不参与正式策略选择。
- `audit_tworoom_cem_final_stage_residual.py` 增加可选的 train/calibration-only `success_penalty` 目标，用于检验连续 task cost 与二值 success 的目标错位；该分支没有被升级为正式策略。

所有正式运行均通过宿主 RTX 5060 Ti（PyTorch CUDA 可用），没有 CPU fallback。

## 独立闭环证据

### Gaussian CEM：final-stage residual

| 新 panel | frozen Gaussian | final-stage residual | paired success 差异 |
|---|---:|---:|---:|
| seed `20261400–20261431` | `8/32` | `8/32` | wins/losses/ties = `0/0/32` |
| seed `20261432–20261463` | `7/32` | `7/32` | wins/losses/ties = `0/0/32` |

在第二个 panel 上把 residual 放宽为全候选、全强度（`q=1.0, multiplier=1.0`）仍为 `7/32`，说明问题不是 trust-region 的单一阈值过窄。将 train/calibration-only success penalty 固定为 `20` 后第一 panel 仍为 `8/32`，说明 penalty 也没有修复闭环 transfer。

诊断 trace 表明 residual 并非没有生效：两次重规划的动作均值绝对差约为 `0.149` 和 `0.566`。因此失败机制是“动作改变但没有跨过 success basin”，不是 gate 没触发。

### Candidate-bank support intervention

在同样的两个新 panel 上，比较纯 Gaussian frozen baseline 与 candidate-bank frozen baseline：

| 新 panel | Gaussian frozen | candidate-bank frozen | bank 相对 Gaussian paired |
|---|---:|---:|---:|
| seed `20261400–20261431` | `8/32` | `12/32` | `5/1/26`（wins/losses/ties）|
| seed `20261432–20261463` | `7/32` | `9/32` | `4/2/26`（wins/losses/ties）|

合并为 `21/64` 对 `15/64`，paired 为 `9/3/52`。这只是 TwoRoom 的 support-aware 闭环结果，不是 universal world-model 证据。

反例同样重要：把旧 residual readout 接到 candidate bank，在第二 panel 上从 bank baseline 的 `9/32` 降至 `2/32`，paired 为 `0/7/25`。因此 candidate coverage 的收益不能归因于旧 residual；旧 residual 必须 fail-closed。

## 原因定位

当前负结果由两个正交问题造成：

1. **CEM proposal support 不足。** 纯 Gaussian proposals 在有限预算下不能稳定覆盖少量成功方向；candidate bank 直接增加了预声明的轴向/对角方向，因而改善了 success coverage。
2. **latent residual 的 transfer 不足。** final-stage residual 学到的是连续 `real_task_cost` 的局部修正。它能改变候选排序，却不能稳定识别部署 root 上的成功 basin；将其扩大到全候选或改成 success-aware penalty 也没有解决这一点。candidate-bank 上旧 residual 退化进一步说明，readout 的 action-support 与部署 bank 不匹配时会造成负 transfer。

final-stage gate 仍然是必要的 feedback-shift 修复，但它只是防止 residual 污染前期 proposal distribution，不等于恢复 action identifiability，也不等于获得闭环收益。

## 当前正式运行策略

- TwoRoom 正式控制协议：`candidate-bank + frozen Euclidean LeWM`。
- final-stage residual：保留为诊断 artifact，默认 fail-closed，不作为正式控制器。
- Gaussian baseline、candidate-bank baseline、residual ablation 必须分开报告，不能合并成 residual 的正结果。
- 当前结论限于 TwoRoom 的 action-support 机制；尚未证明跨环境、跨架构或通用 world-model 结论。

## Cube：从一步 action 到 macro transfer 的后续诊断

本轮在 `swm/OGBCube-v0` 上继续追踪接触动作负结果。原始 40-calibration /
32-held-out、root-disjoint macro ranking 的结果为：

- 原始 RGB geometry + 线性 macro readout：mean regret `0.02610 m`，
  oracle agreement `65.6%`；task 2 的 7 个 held-out roots 全部错选到
  world-y 方向。
- 加入预先声明的 RGB goal-direction × world-action bilinear interaction，
  并保持相同 split/action bank：mean regret `0.00648 m`，oracle agreement
  `84.4%`；paired improvement `0.01962 m`，bootstrap 95% CI
  `[0.00781, 0.03275] m`，10 wins / 2 losses / 20 ties。

这定位了第一层原因：RGB goal displacement 位于 image frame，而 action
位于 world frame；仅拼接两者的线性 readout 无法表达坐标变换。修复代码在
`experiments/audit_cube_macro_action_ranking.py`，artifact 在
`results/gdd_wm_cross_environment_20261016/cube_macro_rank_root32_bilinear_v1/`。

但该结果仍是 guided contact-support mechanism audit，不是在线规划收益。将
decision point 移到 episode initial state 后，原 20-step cardinal macro 的
8/8 roots 对所有候选产生完全相同的 block effect，说明第二层原因是
open-loop approach overshoot / action-support 缺失。预先冻结 8-direction、
60-step macro 仍不能稳定覆盖初始接触；加入 `{2,4,6}` approach-duration
bank 的 8-root smoke 仅有 2 个 roots 出现非零可改善候选，mean regret
`0.00120 m`、agreement `75.0%`，不足以进入正式闭环结论。

直接尝试 RGB-only visual-servo 接近也未闭合：

- `experiments/run_cube_rgb_visual_servo.py` 已实现 calibration/test root
  分离和 GPU camera-action fit；
- 4 calibration / 4 held-out smoke 的 paired gain 为 `0/4`；
- RGB effector component map condition number 为 `6766`，fit residual
  `0.0339`，runtime phase 全部停留在 approach。

因此当前 Cube 的真实失败边界更具体地是：

> RGB latent/hand-crafted chart 尚不能稳定恢复末端执行器在 action frame
> 下的可控视觉坐标；在此之前，macro action ranking 的离线改善不能传递为
> frozen visual online planning。

当前不升级以下主张：Cube 姿态闭合已解决、RGB-only online planning 已解决、
普适 world model 或通用 commutator certificate。后续若继续攻坚，应替换
effector observation/interface（例如学习式 RGB keypoint/flow 或 frozen
visual successor probe），而不是继续增加 Fourier 阶数或 readout 容量。

## 后续证据门槛

下一步应在未参与 bank 设计的 Reacher / Cube / PointMaze 上复用同一 support-aware 协议，并保持 action bank、CEM 预算、seed pairing 和 train/calibration/outcome 分离。如果 candidate bank 的收益不能跨环境复现，就把它定位为 TwoRoom-specific support repair，而不是普适控制理论。

## WA-JEPA-inspired action-conditioned future/task intervention — 2026-10-16

为检验 WA-JEPA 的可迁移启发，新增
`experiments/decision_benchmark/train_joint_rgb_future_task_adapter.py`。
它在同一候选动作序列上使用 causal source context，同时预测：

1. realized RGB successor residual；
2. physical terminal XY effect；
3. 可选的直接 terminal task cost。

共享 trunk 的拟合、归一化和固定 epoch 全部只使用 train roots；8 个
calibration roots 和 8 个 outcome roots 不参与拟合或选择。该实验的物理
terminal state 只作为 intervention-supervised label，runtime 输入没有
simulator state，也没有把 outcome 结果用于调权重。

### 正向机制结果：history-conditioned support panel

在 `jepa_history4_conditioned_source` 的同一 24/8/8 root split 上，
`task-only` 与 `joint future + task-effect` 的 outcome 对照为：

| context / objective | task-effect RMSE | candidate spread ratio | paired regret gain |
| --- | ---: | ---: | ---: |
| frozen history, task-only | 0.32546 | 0.93720 | 0.00674 |
| frozen history, joint RGB-future/task | 0.31021 | 0.95749 | 0.02286 |

joint 目标同时降低 effect RMSE、保持候选变化幅度接近真实幅度，并提高
held-out paired regret gain。这是一个有限的 action-conditioned
future/task 机制证据，不是闭环规划证据。完整 artifact 为：

`results/decision_benchmark_candidate_bank_20261016/joint_latent_future_task_history4_cuda.json`

### 失败复核：support 不足的 checkpoint panel

在另一个 frozen checkpoint/readout panel 上，RGB projection 与 latent
projection 实际来自同一 export，且 source context 无法稳定辨识 root-specific
action geometry。结果为：

| objective | task-effect RMSE | candidate spread ratio | paired regret gain |
| --- | ---: | ---: | ---: |
| task-only | 0.63241 | 1.68250 | -0.07519 |
| joint RGB-future/task | 0.49800 | 1.70414 | -0.11883 |
| joint + direct cost head | 0.43439 | 1.68377 | -0.06734 |
| cost-only | — | 0.22488 | -0.07588 |

因此“平均 prediction error 下降”仍不能推出 action ranking 改善；pairwise
ranking loss 也没有改变被选 candidate。该失败将原因进一步定位为
representation/action-support insufficiency，而不是单纯的 loss、memory
length 或 cost-readout 形式错误。

### 当前方法决策

正式方法不把 joint adapter 自动发布为 controller。只有同时通过
candidate variation、calibration contrast bound、action-inverse risk 和
real/predicted successor consistency 时，才允许使用 learned candidate；否则
回退到冻结的 support-aware baseline。该 intervention 当前状态是
`implemented + partial mechanism evidence`，仍不是 frozen-LeWM causal
dynamics、跨环境闭环规划或 universal world-model 证据。

## Cube RGB visual-servo follow-up — 2026-10-16

The next controlled intervention isolated two implementation/mechanism failures
in the previous RGB visual-servo negative result:

1. The current frame contains both the physical red cube and a smaller
   semi-transparent goal marker. The old goal-y component matching selected the
   marker as the current object. The selector now chooses the largest
   sufficiently visible red component and records this as the physical-cube
   feature. Nine RGB observability tests pass.
2. OGB-Cube uses relative gripper-opening actions: `+1` closes and `-1` opens.
   The old close phase sent `-1`, so an apparently aligned effector could not
   establish contact. The protocol now uses an explicit
   `GRIPPER_CLOSE_ACTION = +1.0` constant and records the semantics in the
   result metadata.

The fixed-seed CUDA smoke with 8 calibration and 8 held-out episodes then
changed from no-contact behavior to 4 wins / 1 tie / 3 losses, mean paired
final-distance delta `-0.01085 m`; six of eight episodes had nonzero evaluator
contact steps. This is an intervention-level improvement, not yet a stable
planning result.

A calibration-only world-chart approach reduced the mean delta to `-0.00707 m`
when calibration was expanded to 40 roots, so calibration support alone is not
the remaining explanation. A goal-conditioned 25-mm standoff was tested but
caused out-of-support approach failures on task 4 and was not adopted as the
default. A pixel-only approach with the corrected gripper semantics produced
only 1 win / 3 ties / 4 losses and mean delta `+0.01487 m`.

The remaining boundary is therefore not just object correspondence or gripper
semantics: world-chart extrapolation and contact-side selection still fail on
diagonal task layouts. A world-approach/pixel-re-observation push controller has
been implemented as the next preregistered comparison, but its CUDA smoke was
not run in this pass because the host-permission request was rejected. No result
from that arm is treated as evidence.

## Temporal closed-loop audit correction — 2026-10-16

An existing Cube temporal closed-loop artifact reported a favorable point
estimate on one 32-episode panel (`7/32` wins, no losses, mean delta
`-0.02646 m`). Re-auditing its paired records shows that all seven gains are
task 1; tasks 2–5 have exactly zero temporal-versus-static difference. A
second 16-episode panel has only 3 gains and 13 ties, while an independent
32-episode panel has 32 ties. This is task/initial-condition heterogeneity,
not a general temporal-memory gain.

The runner's fixed prelude also used the obsolete gripper sign (`-1` for
closing). It has now been corrected to `+1`, and the regression test covers the
action semantics. Therefore the old temporal result is retained as historical
diagnostic evidence but is not valid evidence for the corrected protocol. The
next rerun must use the corrected prelude, root-disjoint pairing, and per-task
readouts before any claim about RGB online planning or temporal memory is
advanced.

## RGB visual-servo contact-threshold repair — 2026-10-16

The first CUDA/EGL world-approach plus pixel-push run exposed a further
protocol error. The controller closed when the calibrated world-space error
norm was below `0.05 m`; task-4 tracing showed an actual approximately 2.6-cm
effector-to-cube lateral gap at closure. The gripper sign was correct, but the
controller was still closing before contact.

The repair has three parts:

- tighten the frozen approach threshold to `0.015 m`;
- increase the fixed episode budget to `90` steps so a long approach does not
  consume the entire push phase;
- use `pixel_adaptive` push, which treats the effector image Jacobian as a
  proposal and reverses its short-pulse direction only when the observed RGB
  cube-to-goal error increases.

The same 40-root calibration and 8-root test panel changed from the preceding
world-approach/adaptive-push result (`2 wins / 5 losses / 1 tie`, mean delta
`-0.00347 m`) to `7 wins / 1 loss / 0 ties`, mean delta `-0.10354 m`.
Increasing only the budget from 65 to 90 on this discovery panel gave
`7/1/0`, mean delta `-0.10891 m`, confirming that the lone loss was partly
budget truncation rather than a new controller effect.

The frozen protocol was then evaluated on an independent seed panel (new
calibration and held-out roots, same five tasks): `7 wins / 1 loss / 0 ties`,
mean delta `-0.11432 m`. Across the two 8-episode panels this is 14 wins and 2
losses with mean paired delta approximately `-0.109 m`. This is strong
evidence for a repaired RGB Cube contact protocol, but it remains a task- and
environment-specific closed-loop result; it does not validate frozen LeWM
visual planning, universal world modeling, or a general transfer certificate.

## Corrected temporal closed-loop rerun — 2026-10-16

The corrected `run_cube_temporal_closed_loop.py` was also rerun with CUDA/EGL,
40 calibration roots, 16 new root-disjoint test episodes, the fixed 20-step
prelude, and the frozen candidate bank. The prelude now uses `+1` for all five
close actions.

Temporal versus static produced `3 wins / 13 ties / 0 losses`, with mean final
distance delta `-0.00737 m`. All three gains occurred on task 3; tasks 1, 2, 4,
and 5 were ties. The support-gated temporal arm was identical on this panel
because its support fallback was never activated. The result therefore rules
out the old obsolete-gripper explanation, but does not support a general
temporal-memory gain: the remaining positive effect is task-specific and
sparse.

## Corrected temporal panel aggregation — 2026-10-16

The two corrected, root-disjoint panels were pooled only after checking the
frozen protocol signature and disjoint test seeds. The aggregate artifact is
`results/negative_cause_diagnosis_20261016/cube_temporal_closed_loop_aggregate_corrected_v1/aggregate.json`.

- temporal versus static: `4` wins / `0` losses / `44` ties across `48`
  held-out episodes; mean delta `-0.00277 m`; paired bootstrap 95% CI
  `[-0.00740, -0.00015] m`; exact two-sided sign-test `p=0.125`;
- support-gated temporal versus static: `5` wins / `0` losses / `43` ties;
  mean delta `-0.00327 m`; paired bootstrap 95% CI
  `[-0.00824, -0.00032] m`; exact two-sided sign-test `p=0.0625`;
- the support gate triggered on `0/48` episodes, so this pooled comparison does
  not test a nontrivial fallback intervention.

The aggregate confirms that the corrected protocol has no observed temporal
deterioration, but the effect is sparse, task-dependent, and not statistically
decisive under the frozen paired sign test. It is therefore retained as a
boundary result rather than evidence for a general temporal-memory gain.

## Frozen RGB visual-servo panel aggregation — 2026-10-16

The discovery and independent-seed threshold-0.015 panels were pooled in
`results/negative_cause_diagnosis_20261016/cube_rgb_visual_servo_aggregate_threshold015_v1/aggregate.json`.
Both panels use the same frozen world-approach, `0.015 m` closure threshold,
`pixel_adaptive` push protocol, CUDA/EGL runtime, RGB-only policy input, and
root-disjoint calibration/test separation; the test seeds are disjoint. The
input artifacts carry schema `v9`, but their checked protocol fields are
identical to the current corrected implementation.

- pooled result: `14` wins / `2` losses / `0` ties over `16` held-out episodes;
  mean paired final-distance delta `-0.11161 m`;
- paired bootstrap 95% CI: `[-0.16766, -0.06074] m`;
- exact two-sided sign-test `p=0.00418`;
- calibration support: `80` roots total, with `16` root-disjoint test episodes;
- task-stratified gains are strongest for tasks 1, 3, 4, and 5, while task 2
  contains both observed losses. The per-task sample sizes and intervals are
  preserved in the aggregate JSON rather than collapsed into the pooled claim.

This is the strongest current positive result for the repaired RGB contact
protocol. It identifies a reproducible protocol-level fix for Cube contact
closure, not a proof that frozen LeWM performs universal visual online
planning, that latent dynamics are causally physical, or that temporal memory
is generally useful.

## Temporal budget intervention and 64-episode pooled boundary — 2026-10-16

To test whether the temporal near-ties were caused by the 20-step post-prelude
budget, a third independent CUDA/EGL panel used the same frozen calibration,
candidate bank, prelude, and support gate but increased `max_steps` from `40`
to `90`. Its test seeds were new and disjoint
(`20262500`–`20262515`). The artifact is
`results/negative_cause_diagnosis_20261016/cube_temporal_closed_loop_corrected_budget90_seed20261500/closed_loop.json`.

The budget-90 panel produced `2 wins / 0 losses / 14 ties`, with mean temporal
minus static final-distance delta `-0.00149 m`. Thus extending the action
budget did not unlock a broad temporal-memory gain; the remaining ties are not
explained by simple episode truncation.

The three corrected panels were then pooled in
`results/negative_cause_diagnosis_20261016/cube_temporal_closed_loop_aggregate_corrected_64ep_v2/aggregate.json`:

- temporal versus static: `6 wins / 0 losses / 58 ties` over `64` episodes;
  mean delta `-0.00245 m`; paired bootstrap 95% CI
  `[-0.00615, -0.00026] m`; exact two-sided sign-test `p=0.03125`;
- support-gated temporal versus static: `7 wins / 0 losses / 57 ties`;
  mean delta `-0.00282 m`; paired bootstrap 95% CI
  `[-0.00664, -0.00046] m`; exact two-sided sign-test `p=0.015625`;
- support fallback triggered on `0/64` episodes, so the gate does not explain
  the pooled effect.

This is a reproducible but sparse, task/initial-condition-dependent benefit,
not a universal temporal-memory result: all observed non-tie changes favor
the temporal arm, but `58/64` paired episodes are exact ties. The mechanism
diagnosis is now sharper: the corrected temporal representation can help a
small subset of Cube roots, while the fixed candidate bank and current
RGB/action interface make it decision-equivalent to the static arm on most
roots. Further Fourier or selector tuning is therefore not justified; the
next repair must increase action-effect identifiability or use a richer
closed-loop action interface.

## Temporal action-divergence audit — 2026-10-16

To distinguish “the temporal representation is not used” from “it is used but
does not transfer to the task,” the three corrected panels were audited with
`experiments/audit_cube_temporal_action_divergence.py`. The artifact is
`results/negative_cause_diagnosis_20261016/cube_temporal_action_divergence_64ep_v1/action_divergence.json`.

Across `64` paired episodes, temporal and static selected different actions in
`37/64` episodes (`57.8%`), with a mean of `21.77` changed action steps per
episode. Nevertheless, only `6/64` episodes changed the final evaluator
outcome, all in task 3. Among episodes with any action change, the mean final
distance delta was `-0.00424 m`; episodes with equal actions had exactly zero
delta.

The task stratification makes the failure mechanism explicit:

- task 2: action changes in `10/13` episodes, but `0/13` outcome changes;
- task 4: action changes in `12/12` episodes, but `0/12` outcome changes;
- task 5: action changes in `6/12` episodes, but `0/12` outcome changes;
- task 3: `7/12` episodes changed actions and all six pooled outcome changes
  occurred there.

Thus the remaining negative result is not a missing-memory or inactive-gate
bug. The temporal feature changes the candidate policy, but the current RGB
action interface often maps those different candidates to the same block
trajectory. This directly supports the revised diagnosis of within-mode
action-effect non-identifiability and explains why more temporal capacity or
Fourier harmonics alone is unlikely to solve the pose/transfer failure.

## Latent-temporal plus RGB successor-feedback repair — 2026-10-16

The action-divergence audit suggested a targeted repair: retain the frozen
latent-temporal ranker for candidate selection, but use the current RGB
physical-cube/goal error only as a deployment feedback signal. After the fixed
prelude, if the observed RGB error increases after the previous XY pulse, the
next latent-temporal XY pulse is sign-reversed. No simulator contact, block
state, or evaluator outcome is exposed to the controller; the readout and
candidate bank are unchanged.

The implementation is in
`experiments/run_cube_temporal_closed_loop.py` as the
`temporal_pixel_feedback` arm. On an independent 16-episode, root-disjoint
CUDA/EGL panel (`seed=20261600`, `max_steps=90`), the unmodified temporal arm
was `0 wins / 0 losses / 16 ties` against static, while the feedback arm was:

- `6 wins / 0 losses / 10 ties`;
- mean final-distance delta `-0.01687 m`;
- paired bootstrap 95% CI `[-0.03449, -0.00361] m`;
- exact two-sided sign-test `p=0.03125`;
- feedback reversed the proposed direction `68` times across `9/16` episodes.

The machine-readable aggregate is
`results/negative_cause_diagnosis_20261016/cube_temporal_pixel_feedback_aggregate_budget90_16ep_v1/aggregate.json`.
The improvements are concentrated in tasks 2 and 3; tasks 1, 4, and 5 are
ties. This is the first intervention that directly repairs the identified
latent-action transfer bottleneck, but it is still one independent Cube panel.
It must be replicated on new roots and then on Reacher/TwoRoom/PointMaze before
being promoted to a cross-environment method claim.

## RGB successor-feedback replication and pooled adjudication — 2026-10-16

The same implementation and frozen protocol were rerun on a second,
root-disjoint 16-episode CUDA/EGL panel (`seed=20261700`, `max_steps=90`). The
raw artifact is
`results/negative_cause_diagnosis_20261016/cube_temporal_pixel_feedback_budget90_seed20261700/closed_loop.json`.
On this panel the unmodified temporal arm again had `0` wins / `0` losses /
`16` ties against static, but temporal plus RGB feedback had `1` win / `2`
losses / `13` ties, with mean delta `+0.000261 m`. Therefore the first-panel
gain does not replicate as a stable root-independent improvement.

The two feedback panels are pooled in
`results/negative_cause_diagnosis_20261016/cube_temporal_pixel_feedback_aggregate_32ep_v2/aggregate.json`:

- temporal versus static: `0` wins / `0` losses / `32` ties;
- RGB feedback versus static: `7` wins / `2` losses / `23` ties over `32`
  episodes;
- pooled mean delta: `-0.00831 m`, with paired bootstrap 95% CI
  `[-0.01800, -0.00132] m`;
- exact two-sided sign-test `p=0.1796875` because only `9/32` episodes are
  non-ties;
- feedback reversed the proposed direction on `9/32` episodes in the pooled
  panel (`68` reversals in the first panel plus the second-panel trace).

The correct conclusion is therefore **inconclusive, task/root-specific
feedback benefit**, not a stable solution. RGB successor feedback addresses
the diagnosed within-mode action-effect mismatch in some roots, but its
current pulse-sign rule is not yet a general repair. The next intervention
must condition the feedback on calibrated effect confidence and evaluate it
with new environments; no temporal-memory or universal visual-planning claim
is promoted by this pooled result.

## Third feedback panel and strategy comparison — 2026-10-16

To test whether the second-panel deterioration was a root-specific failure or a
failure of the feedback rule itself, a third root-disjoint CUDA/EGL panel was
run with the same frozen calibration and protocol. Its artifact is
`results/negative_cause_diagnosis_20261016/cube_temporal_feedback_strategy_panel_seed20261800/closed_loop.json`.
The original persistent-sign feedback arm produced `6` wins / `0` losses /
`10` ties. A new one-step variant, which reverses only the next XY pulse after
an RGB error increase and then restores the sign, produced `6` wins / `2`
losses / `8` ties with mean delta `+0.00398 m`; its single-panel bootstrap CI
was `[-0.01372, 0.02535] m` and exact sign-test `p=0.2891`. Thus replacing the
persistent rule with a one-step correction is not supported by this panel.

The three independent persistent-feedback panels are pooled in
`results/negative_cause_diagnosis_20261016/cube_temporal_pixel_feedback_aggregate_48ep_v3/aggregate.json`:

- persistent RGB feedback versus static: `13` wins / `2` losses / `33` ties
  over `48` episodes;
- mean final-distance delta: `-0.01209 m`;
- paired bootstrap 95% CI: `[-0.02097, -0.00459] m`;
- exact two-sided sign-test `p=0.007385` after removing ties;
- task-stratified results: task 1 `3/0/9`, task 2 `4/0/5`, task 3 `6/2/1`,
  task 4 `0/0/9`, task 5 `0/0/9` (wins/losses/ties).

This is now a meaningful positive intervention for the diagnosed action-effect
transfer bottleneck, but its scope is explicit: the gain is concentrated in
Cube tasks 1–3 and absent on tasks 4–5. The persistent RGB feedback rule is
retained as the current Cube protocol; one-step feedback is retained as a
negative strategy comparison. Cross-environment validation on Reacher,
TwoRoom, and PointMaze is still required before claiming a general solution.

## Independent TwoRoom consequence — 2026-10-16

To check that the action-support diagnosis was not specific to Cube contact
geometry, a new 32-episode root-disjoint TwoRoom panel was run with the frozen
LeWM checkpoint, RGB-only current/goal observations, a predeclared candidate
bank, and same-seed paired random control. The artifact is
`le-wm-main/results/gdd_wm_cross_environment_20261016/live_tworoom_candidate_bank_independent_panel_seed20261900_32/planning.json/planning.json`.
The new panel produced `2` wins / `1` loss / `29` ties, with success-rate
difference `+0.03125`.

Together with the two previously frozen, same-protocol 32-episode panels, the
predeclared pooled artifact is
`le-wm-main/results/gdd_wm_cross_environment_20261016/live_tworoom_candidate_bank_baseline_aggregate_96ep_v1/aggregate.json`:

- frozen LeWM versus random: `20` wins / `2` losses / `74` ties over `96`
  paired episodes;
- success-rate difference: `+0.1875`;
- paired bootstrap 95% CI: `[0.10417, 0.28125]`;
- one-sided exact sign-test `p=6.06e-05`.

This is independent evidence that a frozen RGB planner with fixed action
support can yield a positive scoped consequence outside Cube/Pusht. It does
not validate the residual geometry selector, prove causal latent physics, or
establish a universal world model: the control is random rather than a strong
policy baseline, and the result remains environment/protocol-specific.

## Independent Reacher consequence — 2026-10-16

A third environment was tested with a root-disjoint 16-episode CUDA/EGL panel
using the frozen RGB-only Reacher LeWM, same-seed random control, and a
predeclared four-candidate pulse-recovery bank. The raw artifact is
`le-wm-main/results/gdd_wm_cross_environment_20261016/live_reacher_support4_independent_panel_seed20261950_16/planning.json`;
the schema-compatible aggregate is
`le-wm-main/results/gdd_wm_cross_environment_20261016/live_reacher_support4_independent_aggregate_16ep_v1/aggregate.json`.

The frozen planner achieved `6` wins / `0` losses / `10` ties against random,
with success rate `0.375` versus `0.0`, paired difference `+0.375`, bootstrap
95% CI `[0.125, 0.625]`, and one-sided exact sign-test `p=0.015625`. This is an
independent RGB planning consequence outside Cube, PushT, and TwoRoom. It is
still a scoped support-bank result against a random baseline; it does not
establish universal transfer or validate the failed residual geometry selector.

## PointMaze representation/action-interface diagnosis — 2026-10-16

The corrected PointMaze replay already fixed the earlier five-physics-step
clock mismatch, but the support-matched learned-adapter panel still reports
high action inverse risk. To separate a split artifact from a representation
condition, `experiments/empirical_spectral_audit/diagnose_pointmaze_representation_condition.py`
was repaired to support package execution and an explicit train/calibration
split. The diagnostic uses raw `action_blocks_raw`, never reads outcome rows,
and requires at least eight calibration rows. Its artifact is
`le-wm-main/results/negative_cause_diagnosis_20261016/pointmaze_representation_condition_train_cal_v2/diagnostic.json`.

On the available 32-row matched export (`14` train / `8` calibration /
`10` outcome-reserved rows), the effective action rank is `13` for both
representations. Action-only calibration R² is negative for JEPA at h=1/2/4
(`-0.374/-1.417/-0.633`) and for DINO (`-0.830/-0.868/-0.969`). Adding source
latent features does not repair it: JEPA gives `-0.297/-1.276/-0.552`, while
DINO gives `-0.104/-0.154/+0.066`. The result points to insufficient
within-root action-to-latent identifiability under the current representation
and sample support, rather than a missing Fourier or temporal feature.

The diagnostic remains deliberately non-promotional: the DINO export has only
32 rows, so a larger matched export is required for a high-power estimate.
Until that data condition is met, PointMaze learned-adapter transfer must
remain fail-closed; goal-frame action-coordinate repairs can be reported as
offline action-effect diagnostics, not as proof of online visual planning.

## Indexed canonical readout: calibration-gated repair — 2026-10-16

The earlier indexed canonical intervention exposed a sharper failure than a
missing candidate identity. On the same 32-root CUDA panel and the same
candidate bank, replacing frozen Euclidean CEM costs with the train-only
indexed task readout reduced success from `14/32` to `8/32`. The readout was
therefore changing the online ranking in a way that did not transfer to the
held-out RGB control consequence.

The runtime protocol now applies a predeclared calibration gate before an
indexed readout can become active:

\[
\text{enable indexed readout}
\Longleftrightarrow
\operatorname{upperCI}_{95}\!\left(
R_{\mathrm{indexed}}-R_{\mathrm{action-only}}
\mid \text{calibration roots}\right)<0.
\]

The gate is fit on train roots, evaluated on root-disjoint calibration roots,
and never reads outcome labels for selection. If it fails, the runtime returns
the frozen base cost for every candidate, including supported bank entries.
Older artifacts without gate metadata are also treated as disabled.

For the raw indexed artifact, the calibration delta was `-18.7344` with
bootstrap 95% CI `[-45.6880, 9.7637]`; for the latent-relative goal artifact it
was `-14.9146` with CI `[-38.7106, 10.9242]`. Both gates therefore failed
closed. On a fresh 32-root CUDA panel (`20261640–20261671`), both safe-gated
artifacts produced exactly `14/32`, with identical episode-success vectors to
the frozen candidate-bank baseline; the paired result against random was
`13/0/19` wins/losses/ties (`0.4375` versus `0.03125`). This repairs the
negative transfer, but does not establish an indexed-readout gain.

The implementation is in
`experiments/gdd_wm/build_tworoom_indexed_canonical_readout.py`,
`experiments/capacity_readout_runtime.py`, and
`experiments/run_live_tworoom_planning.py`; regression coverage is in
`tests/test_capacity_readout_runtime.py` and
`tests/test_gdd_wm_indexed_readout.py` (`24 passed`). The correct current
scientific interpretation is: train-only task readouts are not admissible as
online controllers without an independent calibration gate; candidate-bank
support remains the scoped TwoRoom positive mechanism, while indexed readout
benefit remains unproven.

## Reacher calibration-safe readout: runtime support mismatch and fail-closed repair — 2026-10-16

The first attempt to validate the Reacher calibration-safe readout used the
standard CEM runner. Its frozen `CapacityReadoutCostModel` correctly returned
the base cost because standard CEM does not provide `_candidate_bank_mask`; the
result was exactly identical to Euclidean CEM and therefore was not evidence
that the readout had been evaluated.

The corrected protocol uses `run_live_reacher_sequence_candidate_planning.py`
with the matching `capacity17` candidate bank, which supplies the support mask.
On the same 8 seeds (`20262000–20262007`):

- Euclidean capacity-bank CEM: `2/8` success;
- raw capacity readout replacing supported-bank costs: `0/8` success;
- additive conservative readout (`base_rank_quantile=0.25`,
  `correction_multiplier=0.5`): `2/8` success.

The fresh 32-seed panel (`20262000–20262031`) confirms the boundary:

- Euclidean capacity-bank CEM: `10/32`;
- additive conservative readout: `10/32`;
- paired result: `0/0/32` wins/losses/ties, with identical success vectors.

This isolates two distinct issues. First, the original standard-CEM smoke was
a no-op caused by missing action-support metadata. Second, once support is
provided, replacing the frozen cost with an offline task readout produces
negative transfer (`0/8`); keeping the frozen cost as the primary term and
restricting the readout to the base top quartile removes the deterioration but
does not create a planning gain. The implementation now makes standard CEM
reject a gated readout instead of silently reporting an inactive intervention;
the sequence runner exposes and records the conservative overrides.

Artifacts:

- `results/gdd_wm_cross_environment_20261016/live_reacher_capacity17_smoke8/`;
- `results/gdd_wm_cross_environment_20261016/live_reacher_capacity17_panel32/`;
- `experiments/run_live_reacher_planning.py`;
- `experiments/run_live_reacher_sequence_candidate_planning.py`.

The current Reacher conclusion is therefore fail-closed/non-degradation, not a
positive latent-transfer result: offline capacity regret does not yet transfer
to online planning, and the readout remains diagnostic-only.

## PointMaze RGB online planning: dynamics-gated frozen WM repair — 2026-10-16

The previous PointMaze RGB closed-loop negative result was reproduced under the
same frozen JEPA-WM checkpoint, RGB-only runtime contract, calibration/test
split, and route-conditioned candidate bank:

- original `wm_rgb`: `5/80` success, mean final distance `0.9850`;
- visual-greedy baseline: `80/80`, mean final distance `0.1464`;
- visual-dynamics baseline: `80/80`, mean final distance `0.1550`.

The failure was localized by adding a predeclared calibration-only block
dynamics gate. The gate compares the frozen WM's selected action with the
action preferred by an RGB-velocity plus calibration-only dynamics readout and
falls back when the WM action's dynamics-score regret exceeds the frozen
tolerance `0.10`. It does not use held-out simulator state for policy input or
selection.

The first 10-root mechanism panel gave:

| method | success | mean final distance |
| --- | ---: | ---: |
| original `wm_rgb` | `0/10` | `1.0391` |
| `wm_rgb_guarded` | `10/10` | `0.1476` |

The independent 80-root panel then gave:

| method | success | mean final distance | mean blocks |
| --- | ---: | ---: | ---: |
| original `wm_rgb` | `5/80` | `0.9850` | `56.44` |
| `wm_rgb_guarded` | `80/80` | `0.1483` | `19.41` |

Against the original WM on the same 80 roots, the guarded controller improved
final distance by `-0.8367` on average (paired bootstrap 95% CI
`[-0.8854,-0.7802]`) and improved success by `75/80` roots (paired bootstrap
95% CI for the success-rate difference `[0.875,0.988]`). The gate made `1553`
decisions and fell back on `1290` (`83.1%`), direct evidence that the original
WM action ranking was the dominant failure mechanism under this deployment
support. This is a positive result for a support-aware safety gate, not
evidence that the unguarded frozen WM plans successfully.

Implementation and regression coverage:

- `experiments/run_live_pointmaze_planning.py` now wires
  `wm_dynamics_tolerance`, exposes `--methods`, and records the gate in the
  protocol and per-step trace;
- `tests/test_live_pointmaze_planning.py` contains a deterministic regression
  test proving that an unsafe WM ranking falls back to calibration dynamics;
- CUDA reports:
  `results/pointmaze_rgb_online_planning_v6_cuda_guarded_smoke10/report.json`
  and
  `results/pointmaze_rgb_online_planning_v7_cuda_guarded_panel80/report.json`.

The scientific boundary remains explicit: the gate uses a calibration-only
proxy and is currently validated on legacy PointMaze dynamics only. It does
not establish universal world modeling, physical causality of the latent, or a
general commutator certificate. The negative result is repaired as a
deployment-risk diagnosis: frozen WM ranking must be rejected when it is
inconsistent with an independently calibrated action-effect interface.

## Cube post-contact temporal memory: two support failures and current boundary — 2026-10-16

The first branch-calibrated Cube panel did not produce a valid result: one
calibration root silently returned `None` after failing to reach the
post-contact phase. The runner now fails closed with the root, task, phase, and
final approach error instead of writing a partial artifact. The failing RGB
approach was `seed=20261226`, task 2, at `0.01846 m` after 120 steps; a
180-step check also failed on task 4 (`seed=20261223`, `0.04139 m`). This
identifies a visual approach-geometry boundary, not a latent-memory result.

For a mechanism-isolation experiment, the runner now has an explicit
`--branch-controller privileged_world` mode. It uses world geometry only to
construct a shared post-contact state; all four controller arms receive the
same frozen RGB frames, previous action, goal frame, and state reset, and no
privileged state is passed at runtime. This mode is not RGB visual-approach
evidence; RGB visual-servo results remain separate.

The original post-contact temporal readout was then shown to have a second
support mismatch: it was fit mostly with the close action as `previous_action`,
while after the first deployment step the previous action is a push action. The
calibration protocol was changed to include, for every calibration root, the
initial close frame plus one transition for every fixed candidate-bank previous
action, followed by the same candidate bank as the next action. This is
calibration-only sequence support expansion.

Development panel (`40` calibration roots, `32` test roots,
`20262220–20262251`, CUDA) changed the raw temporal-minus-static consequence
from a clear negative to:

- mean final-distance difference `-0.01219 m`;
- paired bootstrap 95% CI `[-0.03156, 0.00108]`;
- `5/4/23` wins/losses/ties;
- task 1 and task 2 positive, task 3 near zero, tasks 4 and 5 identical.

This is a development result, not a confirmed universal gain. On an independent
root-disjoint panel (`20263020–20263051`, same frozen checkpoint and protocol),
raw temporal-minus-static was `+0.00275 m` with `3/8/21` wins/losses/ties.
Thus sequence support repair removes the diagnosed failure mode but does not
establish a stable cross-root temporal benefit.

The code also includes two explicitly diagnostic alternatives:

- `latent_temporal_compact`: fixed 24-bin latent pooling;
- `latent_gated_residual`: static carrier plus a calibration-only event/support
  gate, otherwise fail-closed to static.

The gated smoke correctly produced no degradation (`0/0/4` against static),
but also no gain; compact pooling was not consistently better. These results
support the narrower conclusion that temporal history is useful only under
matched previous-action support and task/mode conditions. They do not support
the claims of universal latent causal dynamics or universal planning gain.

Artifacts:

- `results/gdd_wm_cross_environment_20261016/cube_hybrid_contact_latent_sequence_calibrated_panel40cal32test_cuda/`;
- `results/gdd_wm_cross_environment_20261016/cube_hybrid_contact_latent_sequence_calibrated_independent_seed20262020_cuda/`;
- `results/gdd_wm_cross_environment_20261016/cube_hybrid_contact_latent_gated_residual_smoke_cuda/`;
- `experiments/run_cube_hybrid_contact_latent.py`;
- `experiments/audit_cube_temporal_action_effect.py`.

## WA-JEPA-inspired RGB contact coverage audit — 2026-10-16 continuation

The WA-JEPA reading suggests an action-conditioned future interface, but the
first prerequisite remains observable contact coverage.  The next intervention
therefore added a deployment-visible local RGB Jacobian to
`run_cube_hybrid_contact_latent.py`: after descent, the controller may issue
short `+x/-x/+y/-y` RGB probes, fit a ridge-regularized local action-to-effector
map from successive RGB centroids, and then run the approach controller.  No
simulator state, contact label, or held-out outcome is used by the update.

The smoke results locate another boundary rather than closing the task:

- RGB world approach with `max_steps=90` failed before push on test seed
  `20262221`, task 2, at `0.04431 m` approach error.
- Increasing the fixed budget to `180` did not solve it; the same protocol
  failed on seed `20262221`, task 2, at `0.02689 m`.  Thus this is not only a
  step-budget problem.
- A global RGB pixel approach reached push on task 2 but remained in approach
  on tasks 3 and 4 under the fixed gain-0.20 smoke.  Gain `0.50` reached push
  on tasks 1, 2, and 4 but still failed task 3, so gain tuning alone is not a
  stable repair.
- Passive online Jacobian updates were insufficient because early actions did
  not excite both action directions.  Explicit four-direction probes reduced
  the task-3 initial pixel error from about `0.288` to `0.153`, but the
  calibration root still failed to reach push after `180` steps.

The complementary world-frame control smoke reached push on all four held-out
test roots (`20262220–20262223`), while a separate calibration-seed panel
reached push on only `3/4` roots.  This root dependence is why the hybrid
runner remains fail-closed: a post-contact comparison cannot be released as a
visual-planning result when the contact branch itself is not complete.

The local-Jacobian implementation has five focused tests and the full Cube
regression suite passes (`28 passed`).  The current scientific conclusion is:

> WA-JEPA's action-conditioned future idea is relevant, but LeWM first needs a
> root-stable RGB action interface.  Until contact coverage is independently
> passed, a positive post-contact temporal or pixel-guard result remains a
> conditional mechanism result, not frozen visual online planning evidence.

## World-to-pixel fallback sensitivity and independent-panel boundary

To test whether the complementary world/pixel coverage could be repaired
without privileged state, the branch constructor added a fixed
`world_pixel_fallback` law: start with the calibration-fitted world chart and
switch to RGB pixel feedback after five consecutive RGB error non-improvements.
With a predeclared pixel close threshold of `0.06`, the 8-calibration/4-test
smoke reached the complete comparison protocol.  Its
`latent_temporal_pixel_guarded` arm improved final distance relative to
`latent_temporal` on average by `-0.08750 m` (`1/1/2` wins/losses/ties) and
relative to `latent_static` by `-0.06584 m` (`2/0/2`); it remained slightly
worse than pure RGB pixel-adaptive feedback (`+0.00910 m`, `0/2/2`).

This is only a mechanism smoke.  The independent 40-calibration/32-test panel
did not reach the comparison stage: calibration seed `20261238`, task 4,
stalled after 180 steps at RGB pixel error `0.26020` even after the fallback.
Consequently the smoke gain is not released as a cross-root planning result.
The correct interpretation is that the fallback repairs a subset of local
world/pixel chart failures, while the current RGB/action interface still has
out-of-support roots.  A complete visual-planning claim therefore requires a
separate contact-coverage gate or a new observation/action interface; it
cannot be obtained by tuning latent memory or guard thresholds.

The implementation remains covered by the Cube regression suite (`28 passed`)
and records the controller, fallback trace, probe settings, and threshold in
the artifact protocol.  Runtime inputs remain RGB-only; privileged IK is not
used in this branch.

## WA-JEPA-inspired future/action intervention on PointMaze — 2026-10-16

The first implementation exposed an evaluation confound before adding a new
model. The joint adapter returned a `cost_prediction` tensor for every
objective, and the per-seed report used that tensor for ranking even for
`task_only` and ordinary `joint` objectives that never trained the cost head.
The ensemble path already disabled that head for those objectives. This
created inconsistent seed-level evidence. `_split_report` now has an explicit
`allow_cost_head` flag, and only `joint_cost`, `cost_only`, and the explicitly
declared preference objective can use it. The regression test
`test_non_cost_objective_does_not_use_cost_head_for_ranking` guards this
invariant.

The code then added two asymmetric future/action objectives to
`train_joint_rgb_future_task_adapter.py`:

- `asym_joint`: future loss sees a detached action representation, while the
  task branch sees the live action representation and the future hidden state;
- `asym_rank`: the same architecture with the existing terminal-effect
  ranking loss;
- `asym_preference`: a direct action-preference score trained on within-root
  pairwise cost orderings rather than deriving ranking only from predicted XY.

These are mechanism interventions inspired by WA-JEPA's asymmetric
future/action coupling, not a claim that the small PointMaze adapter is a flow
matching world-action model.

On the history-4 support-sufficient panel (24 train / 8 calibration / 8
outcome roots, five seeds, CUDA), the corrected v3 artifact gave:

| objective | task-effect RMSE | paired regret gain | selected success | candidate spread ratio |
| --- | ---: | ---: | ---: | ---: |
| task-only | 0.32545 | 0.00674 | 0.125 | 0.93721 |
| shared joint | 0.31023 | 0.01720 | 0.125 | 0.95757 |
| asymmetric joint | 0.28842 | 0.02761 | 0.250 | 0.96775 |
| asymmetric rank | 0.28682 | 0.02438 | 0.250 | 0.96655 |

All five `asym_joint` seed-level paired gains were positive on this panel.
However, the root-level improvement was concentrated mainly in one outcome
root; this is mechanism evidence, not stable cross-root closure.

The same code was then run on the earlier single-frame negative panel. The
asymmetric architecture reduced the negative ranking consequence but did not
turn it positive:

- shared joint: paired regret gain `-0.11883`, selected regret `0.13569`;
- asymmetric joint: paired regret gain `-0.00471`, selected regret `0.02157`;
- asymmetric rank: paired regret gain `-0.00406`, selected regret `0.02092`;
- direct preference: paired regret gain `-0.07336`.

This separates two causes. Symmetric multi-task interference is real and can
be repaired in a history-sufficient regime, but the single-frame panel still
lacks enough root-specific action-effect information. A direct preference
head does not solve missing observability by itself. The earlier standalone
adapter gain (`+0.07477`) is not promoted: its random baseline varied with
seed, whereas the corrected paired protocol uses the same root-level random
draw for every method.

Artifacts:

- `results/decision_benchmark_candidate_bank_20261016/joint_rgb_future_task_history4_v3_cuda.json`;
- `results/decision_benchmark_candidate_bank_20261016/joint_rgb_future_task_singleframe_v3_cuda.json`;
- `results/decision_benchmark_candidate_bank_20261016/joint_rgb_future_task_singleframe_v4_preference_cuda.json`;
- `experiments/decision_benchmark/train_joint_rgb_future_task_adapter.py`;
- `tests/test_train_joint_rgb_future_task_adapter.py` (`4 passed`).

Current conclusion: WA-JEPA's asymmetric coupling is a valid partial repair,
but it does not remove the representation/support boundary. The next required
intervention is causal intermediate-future supervision plus a calibration-only
support/uncertainty gate; no universal planning claim is released from these
offline terminal panels.

## Temporal-amplitude and action-effect residual follow-up — 2026-10-16

The next CUDA rerun tested two distinct causes rather than treating every
negative result as a memory-length problem. First, the sequence objective was
augmented with an explicit first-difference loss (`asym_sequence_temporal`).
Second, an action-effect residual decomposition was added: a context-only
root-shared baseline plus a context/action-conditioned residual
(`asym_relative`), with a within-root vector difference loss in
`asym_relative_pair`.

The final comparable artifact is
`results/decision_benchmark_candidate_bank_20261016/joint_rgb_future_task_history4_final_v7_cuda.json`.
It uses the same 24/8/8 root split, five seeds, fixed 250 epochs, and CUDA.
The non-relative model's module registration order was restored before this
run so seed comparisons remain comparable to the original architecture.

| objective | mean seed paired-regret gain | mean task-effect RMSE | future change ratio |
| --- | ---: | ---: | ---: |
| `asym_joint` | `+0.01843` | `0.298` | n/a |
| `asym_relative` | `+0.01362` | `0.306` | n/a |
| `asym_relative_pair` | `+0.01465` | `0.306` | n/a |
| `asym_sequence` | `+0.00736` | `0.300` | `0.624` |
| `asym_sequence_temporal` | `+0.00839` | `0.297` | `0.736` |

The temporal derivative loss therefore repairs temporal amplitude: the
predicted/realized change-magnitude ratio rises from about `0.62` to `0.74`.
It does not improve the downstream ranking consequence over `asym_joint`, and
one of five seed-level gains remains negative. The residual decomposition and
the vector difference loss likewise do not exceed `asym_joint`. They prevent
candidate-spread collapse after the implementation fix, but do not recover
the missing goal-directed action sign.

An additional data-level audit explains this boundary. On outcome roots, the
realized RGB sequence change correlates strongly with displacement magnitude
(Spearman about `0.78`) but is nearly unrelated to goal cost (about `-0.14`).
Thus the current RGB successor preserves that an action caused motion, but not
which action moves toward the goal. Temporal memory is not sufficient when the
future observation chart lacks directional action identifiability.

During this follow-up two implementation issues were caught and repaired
before interpreting the final artifact: the first residual prototype omitted
the task loss for `asym_relative`, and adding the new branch changed module
registration order and therefore seed initialization. The invalid intermediate
artifacts are not promoted as scientific evidence. The regression suite now
passes `13` tests.

The updated claim is consequently narrower and more useful: asymmetric
future/action coupling is a reproducible partial repair; temporal derivative
supervision reduces future-amplitude shrinkage; neither resolves RGB goal-
directed action identifiability. The next valid intervention must change the
observation/action interface or add an independently calibrated directional
effect probe, followed by held-out planning—not another generic memory loss.

## Realized-future oracle upper bound — 2026-10-16

To distinguish successor-prediction error from representation insufficiency,
the same adapter was rerun with the *realized* future RGB projection sequence
as its context (`context_family=realized_future_projection`). This is an
offline oracle diagnostic only: future observations are unavailable at runtime
and no planning claim is attached to this arm.

Artifact:
`results/decision_benchmark_candidate_bank_20261016/joint_rgb_future_task_history4_realized_future_oracle_v8_cuda.json`.

The `asym_joint` seed-level paired regret gains were
`[-0.0311, -0.0535, -0.0120, -0.0707, -0.0190]`; all five were negative.
The ensemble gain was `-0.0560`, despite RGB successor RMSE improving to about
`0.122`. The direct preference head had a positive ensemble point estimate
(`+0.0303`) but mixed seed-level signs and is not promoted as a repair.

This rules out the narrow explanation that the remaining failure is only
caused by the learned successor predictor. The current projected RGB future
interface does not expose a stable directional action-effect coordinate under
the available root/action support; feeding the high-dimensional oracle also
raises variance. The next intervention must add a low-dimensional directional
effect probe or a richer action/observation chart, and must be validated on an
independent held-out planning consequence.

## Direct future-direction readout ablation — 2026-10-16

As a final architecture-side check, `asym_sequence_direct` explicitly feeds
the predicted sequence endpoint and endpoint-minus-start displacement into the
task head. This tests whether the failure was caused by the task head
discarding directional information already present in the predicted sequence.

Artifact:
`results/decision_benchmark_candidate_bank_20261016/joint_rgb_future_task_history4_direct_sequence_v9_cuda.json`.

The direct readout produced seed-level paired gains
`[+0.0310, -0.0047, +0.0052, +0.0170, -0.0070]`, while the original
`asym_joint` remained positive on all five seeds. Its future change ratios were
only `0.58–0.68`. Thus explicitly exposing the predicted temporal direction
does not turn the negative cases into a stable gain. The remaining failure is
not merely hidden inside the task-head pooling operation.

The negative-result diagnosis is therefore closed for this family of
interventions: asymmetric coupling repairs shared-objective interference;
derivative supervision repairs part of temporal amplitude shrinkage; residual
and direct directional readouts do not restore goal-directed action
identifiability. Further progress requires changing the measured interface
(for example, an independently calibrated short-pulse directional effect
probe or a richer RGB chart), followed by a fresh held-out planning panel.

## Root-bootstrap and uncertainty-gated follow-up — 2026-10-16

The active reference-probe result was audited with the root, rather than the
seed-root row, as the resampling unit. The five seed gains were averaged within
each root before bootstrap. For the `v17` `asym_joint` arm, the root mean gain
was `+0.01495`, but the root-bootstrap 95% interval was
`[-0.00920,+0.05593]`; removing one root produced a minimum mean of
`-0.00366`. The apparent positive point estimate is therefore dominated by
root heterogeneity and is retained as mechanism pilot evidence only.

Artifact:
`results/decision_benchmark_candidate_bank_20261016/adapter_root_bootstrap_audit_v17_vs_v15.json`.
The reusable implementation is
`experiments/decision_benchmark/audit_adapter_bootstrap.py`.

To test whether the remaining failure was a missing risk gate, a new
calibration-only ensemble gate was implemented. It fits five seeds on train
roots, estimates context support and seed disagreement, chooses a fixed gate
from calibration roots, and evaluates outcome roots once. On the passive RGB
history-4 hybrid panel:

| arm | outcome paired-regret gain | root-bootstrap 95% CI |
| --- | ---: | ---: |
| asymmetric joint, no gate | `-0.00153` | — |
| support gate | `+0.00153` | `[-0.01597,+0.01797]` |
| support + rank-instability gate | `-0.00005` | `[-0.01773,+0.01684]` |
| latent-history support/rank gate | `-0.00092` | `[-0.01862,+0.01603]` |

The calibration-selected uncertainty penalty was zero in both RGB runs. Thus
the gate can identify and slightly suppress risky roots, but it does not
create a stable action effect. Replacing RGB history with LeWM latent history
also does not recover the ranking consequence. The diagnosis is now narrower:

> The missing object is a root-stable, goal-directed action-effect coordinate;
> neither lower successor RMSE, longer history, latent substitution, nor
> generic uncertainty penalization supplies it.

The gate implementation is
`experiments/decision_benchmark/run_uncertainty_gated_selection.py`, with
regression coverage in `tests/test_uncertainty_gated_selection.py`. All runs
used the host CUDA interpreter and `device=cuda`; no CPU fallback was used.
These results remain diagnostic/negative and do not open claims of frozen RGB
online planning, universal world-model transfer, or a universal commutator
certificate.

## Goal-relative task-effect chart: in-support repair and external boundary — 2026-10-16

The previous hybrid action interface exposed both raw and goal-relative action
coordinates, but it still trained the task-effect head in raw XY. This can
create a chart mismatch: the input action semantics are goal-relative while
the supervised effect is root-oriented. The new intervention trains the effect
head in the goal-relative direct/perpendicular basis and maps its prediction
back to physical XY before candidate ranking.

On the original 16-root near-goal outcome, the goal-relative `asym_joint` arm
had positive paired-regret gains for all five seeds:

```text
+0.01873, +0.01040, +0.01837, +0.01556, +0.01727
```

Its ensemble gain was `+0.01642`, compared with `-0.00153` for the same
architecture trained in raw XY. The root-bootstrap CI was still
`[-0.00952,+0.05746]`; therefore this is a mechanism pilot, not stable
cross-root planning evidence.

To test the boundary without reusing the original outcome roots, a new
canonical 100-root bank was generated with 48 train roots, 16 calibration
roots, 20 previously unused outcome roots, and 16 old outcome roots excluded.
The corrected explicit split contains 84 roots total. All 84 roots passed the
dataset-matched RGB renderer/source gate and realized PCA successor extraction.

On the 20 previously unused outcome roots, the same frozen protocol reversed:

| task chart | ensemble paired-regret gain | root-bootstrap 95% CI | success support |
| --- | ---: | ---: | ---: |
| raw XY | `-0.01000` | `[-0.01649,-0.00440]` | `0/20` |
| goal-relative | `-0.01034` | `[-0.01622,-0.00503]` | `0/20` |

The new outcome roots had source-goal distance `2.19–2.79`, whereas the
original near-goal outcome averaged `1.36` and contained `3/16` roots with at
least one successful candidate. Thus the positive chart result is not a
universal adapter repair. It identifies a more precise law:

> Goal-relative task supervision repairs a chart mismatch only inside the
> candidate/action support where goal-directed effects are observable; under
> distant roots with no successful candidate support, it cannot recover
> planning and may amplify the wrong geometric bias.

The operational consequence is to make task-chart selection
horizon/support-conditioned and fail-closed. A model must not be permitted to
claim a planning gain on roots whose action support has no independently
calibrated directional effect. The external artifact is
`results/decision_benchmark_candidate_bank_20261016/task_frame_external_outcome20_hybrid_v1_cuda.json`;
the RGB bridge is
`results/decision_benchmark_candidate_bank_20261016/rgb_successor_history4_canonical_external_outcome20_v2_cuda/`.
This closes the current positive result as a scoped mechanism plus a sharp
failure boundary; it does not open frozen RGB online planning, universal
world-model, or commutator-certificate claims.

## Goal-directed action-support intervention — 2026-10-16

To test whether the external failure was caused by the paired-order action
bank rather than by the future/task adapter, a separate
`goal_directed_mixture_v1` bank was generated. The source export, checkpoint,
history window, 48/16/20 root split, 16 excluded roots, RGB renderer, frozen
JEPA encoder, five adapter seeds, and task-frame protocol were held fixed. The
new bank was not selected using terminal outcomes and does not overwrite the
canonical bank.

The bank passed the visual bridge on all 84 retained roots:

- renderer/source consistency: `84/84`;
- realized RGB latent extraction: complete;
- PCA successor comparison: complete;
- runtime device: CUDA on the host RTX 5060 Ti.

The candidate support improved, but did not close the task:

| outcome support statistic | canonical bank | goal-directed v1 |
| --- | ---: | ---: |
| mean best terminal distance | `2.1490` | `1.9513` |
| roots with a candidate within `0.25` of goal | `0/20` | `0/20` |
| raw/goal-relative `asym_joint` selected success | `0/20` | `0/20` |

The task-frame adapter selected the best available candidate on all 20 outcome
roots for the new bank, but this still yielded zero terminal successes. A
calibration-only uncertainty/context-support gate selected no fallback roots;
its outcome bootstrap interval was `[0.38665, 0.45595]` for the bank's paired
regret statistic, but this does not constitute a success gain because the
success support remained `0/20`. The gate's context support signal therefore
does not measure action reachability.

The remaining cause is now more specific than generic representation failure:
the 20-step action horizon is physically insufficient for the distant roots.
For example, a far root with source-goal distance about `2.79` moved only about
`0.5` under a saturated 20-step pulse in the declared PointMaze dynamics.
No latent readout can select a successful candidate when the fixed bank and
horizon contain none. The positive v1 changes are therefore an
action-support/reachable-horizon effect, not evidence that the frozen latent
has acquired physical causal dynamics.

Artifacts:

- `results/decision_benchmark_candidate_bank_20261016/jepa_history4_goal_directed_external_outcome20_cuda/`;
- `results/decision_benchmark_candidate_bank_20261016/rgb_successor_history4_goal_directed_external_outcome20_cuda/`;
- `results/decision_benchmark_candidate_bank_20261016/task_frame_goal_directed_external_outcome20_hybrid_v1_cuda.json`;
- `results/decision_benchmark_candidate_bank_20261016/uncertainty_gated_history4_goal_directed_external_outcome20_cuda.json`.

The next repair must therefore be a predeclared reachable-horizon/receding-
horizon protocol (or a separate longer-horizon adapter), evaluated on the
same root split. It must report permission/rejection separately from planning
success. Expanding the action bank alone must not be presented as a latent or
world-model repair.

## RGB-only receding-horizon repair — 2026-10-16

The repository already contained an RGB-only PointMaze receding-horizon
runner, but it originally accepted only contiguous episode ranges. That made
it impossible to reuse the explicit 16-calibration/20-outcome root split
without accidentally placing four outcome roots in the calibration range. The
runner now accepts explicit `--calibration-episodes` and `--test-episodes`
lists, rejects overlap and duplicate IDs, and records the split contract.

On the exact 16/20 split, the original 12-block execution budget was still too
short:

| method | success | mean final distance |
| --- | ---: | ---: |
| `wm_rgb` | `2/20` | `1.6961` |
| `wm_rgb_guarded` | `2/20` | `1.4424` |
| `visual_dynamics` | `2/20` | `1.3961` |

The guarded arm improved distance over unguarded WM (`-0.2537`, paired
bootstrap CI `[-0.4348,-0.0581]`) but still did not improve success and was
worse than the calibration-only visual-dynamics fallback. This confirms that
the earlier positive 80-root result cannot be generalized to the harder
external split at the shorter horizon.

Keeping every other setting fixed and increasing the predeclared receding
horizon to 24 blocks repaired the reachable-horizon failure:

| panel | method | success | mean final distance |
| --- | --- | ---: | ---: |
| external outcome-20 | `wm_rgb` | `2/20` | `1.0773` |
| external outcome-20 | `wm_rgb_guarded` | `13/20` | `0.3243` |
| external outcome-20 | `visual_dynamics` | `12/20` | `0.3247` |
| excluded-16 | `wm_rgb` | `2/16` | `0.9330` |
| excluded-16 | `wm_rgb_guarded` | `13/16` | `0.2191` |
| excluded-16 | `visual_dynamics` | `12/16` | `0.2678` |

Across the two root-disjoint panels (`n=36`), guarded versus unguarded WM had
`26/36` versus `4/36` successes and paired final-distance delta `-0.7356`
with bootstrap CI `[-0.9027,-0.5782]`. Guarded versus visual dynamics had
`26/36` versus `24/36` successes and delta `-0.0219`, CI
`[-0.0780,+0.0275]`; the learned WM is therefore non-inferior to its
calibration-only fallback on this protocol, but not decisively superior. The
gate fell back on `564/703` decisions (`80.2%`).

This is the first stable positive closed-loop consequence in the current
PointMaze branch, but its causal interpretation is deliberately narrow:

> the repair is a support-aware, fail-closed, long-horizon RGB controller;
> most decisions are delegated to a calibration-only visual-dynamics policy,
> so the result does not prove that the frozen learned dynamics alone caused
> the gain.

Artifacts:

- `experiments/run_live_pointmaze_planning.py`;
- `results/pointmaze_rgb_online_planning_v8_cuda_external20_split/report.json`;
- `results/pointmaze_rgb_online_planning_v9_cuda_external20_longer24/report.json`;
- `results/pointmaze_rgb_online_planning_v10_cuda_excluded16_longer24/report.json`.

The protocol now has a real positive planning result for PointMaze, while the
pure `wm_rgb` arm remains negative. The next required test is to reduce the
fallback rate without losing the held-out non-degradation guarantee; any such
change must be selected on calibration roots only.

## WM action projection intervention — 2026-10-16

To test whether the high fallback rate was caused by a binary gate rather than
by an unusable WM action direction, `run_live_pointmaze_planning.py` now adds
`wm_rgb_projected`. It first selects the frozen WM candidate, then searches the
line segment from the calibration visual-dynamics candidate to that WM action.
The executed action is the largest point whose calibration-dynamics score is
within the fixed `0.10` tolerance. The projection uses no outcome labels and
does not alter the frozen WM ranking, action bank, calibration roots, or
receding-horizon budget.

The implementation is covered by two endpoint/support tests in
`tests/test_live_pointmaze_planning.py`; the full live-planning regression
file reports `12 passed`.

The exact 20-root external panel gave:

| method | success | mean final distance |
| --- | ---: | ---: |
| `wm_rgb` | `2/20` | `1.0773` |
| `wm_rgb_projected` | `5/20` | `0.5971` |
| `wm_rgb_guarded` | `13/20` | `0.3243` |

The independent excluded-16 panel gave:

| method | success | mean final distance |
| --- | ---: | ---: |
| `wm_rgb` | `2/16` | `0.9330` |
| `wm_rgb_projected` | `10/16` | `0.4421` |
| `wm_rgb_guarded` | `13/16` | `0.2191` |

Across both panels (`n=36`), projection versus pure WM reduced final distance
by `0.4850` with root-level bootstrap CI `[-0.6151,-0.3583]` and improved
success from `4/36` to `15/36`. However, projection versus guarded planning
was worse by `+0.2507` distance, CI `[+0.1508,+0.3559]`, with success
`15/36` versus `26/36`. Projection was active on `80.3%` of 714 decisions,
with mean interpolation coefficient `0.484`; it therefore reduced binary
fallback by changing actions, not by recovering the missing transfer signal.

The updated diagnosis is:

> action-support mismatch is a genuine component of the negative result, but
> continuous action projection cannot replace action-identifiability. The
> frozen WM ranking remains unreliable in a substantial fraction of roots, and
> the strongest held-out controller still requires fail-closed fallback.

Artifacts:

- `results/pointmaze_rgb_online_planning_v14_cuda_external20_projected/report.json`;
- `results/pointmaze_rgb_online_planning_v15_cuda_excluded16_projected/report.json`;
- `experiments/run_live_pointmaze_planning.py`;
- `tests/test_live_pointmaze_planning.py`.

This is a successful causal diagnosis and a partial repair, not evidence for
pure frozen-world-model planning. The next controller experiment must either
calibrate an alpha-dependent fallback on disjoint calibration roots or add a
future/action preference head that improves the WM ranking itself.

## Alpha-gated projection follow-up — 2026-10-16

The next direct test asked whether the high fallback rate could be reduced by
executing a projected WM action whenever at least `alpha=0.25` of the WM
action remained inside the calibration visual-dynamics support region. The
alpha threshold was fixed before the excluded-16 outcome panel and was not
selected from outcome performance. All other settings matched the 24-block
receding-horizon protocol.

| panel | method | success | mean final distance | fallback rate |
| --- | --- | ---: | ---: | ---: |
| external outcome-20 | `wm_rgb_projected_guarded` | `5/20` | `0.5507` | `13.0%` |
| excluded-16 | `wm_rgb_projected_guarded` | `9/16` | `0.4310` | `17.1%` |

Using the exact 24-block baselines on the same 36 roots gives:

| method | success | mean final distance |
| --- | ---: | ---: |
| `wm_rgb` | `4/36` | `1.0132` |
| `wm_rgb_projected` | `15/36` | `0.5282` |
| `wm_rgb_guarded` | `26/36` | `0.2775` |
| `wm_rgb_projected_guarded` | `14/36` | `0.4975` |

The alpha-gated controller reduced distance relative to pure WM by `-0.5156`
(root bootstrap 95% CI `[-0.6590,-0.3767]`), but it was worse than the
predeclared guarded controller by `+0.2200` (CI `[+0.1400,+0.3074]`) and
lost success on 12 roots while winning on none. Thus reducing binary fallback
does not recover action identifiability: the projected WM ranking remains
unsafe on a substantial subset of held-out roots. This is a failed repair of
the transfer bottleneck, not a failed implementation.

The result sharpens the causal diagnosis:

> support mismatch is real, but the dominant residual error is an unreliable
> within-support action-effect ranking; fail-closed fallback is carrying the
> positive closed-loop result.

Artifact:

- `results/pointmaze_rgb_online_planning_v16_cuda_external20_projected_guarded_alpha025/report.json`;
- `results/pointmaze_rgb_online_planning_v17_cuda_excluded16_projected_guarded_alpha025/report.json`;
- `experiments/run_live_pointmaze_planning.py`;
- `tests/test_live_pointmaze_planning.py` (`12 passed`).

The alpha-gated projection branch is therefore closed as a negative boundary.
The next intervention must change the ranking signal itself (for example a
future/action preference head or an independently calibrated task-transfer
readout), and must preserve the guarded controller as the fail-closed arm.

## HTSC finite-sample calibration correction — 2026-10-16

The review identified a real implementation/theory mismatch in the task
contrast certificate. The previous calibration used the order statistic
`ceil(n(1-alpha))`, which can be below the standard split-conformal finite-
sample rank. The implementation now uses
`ceil((n+1)(1-alpha))`. When the requested rank is larger than the number of
independent calibration roots, the report returns
`status=insufficient_calibration` and `bound=null`; `certify_pair`, offline
candidate auditing, and RGB selective release then fail closed.

This correction is covered by the task-contrast tests and preserves the
existing claim boundary: the construction concerns fixed declared candidate
banks and exchangeable calibration roots. It does not provide conditional
coverage for an adaptive CEM-generated candidate set, and it does not turn a
certificate into a guarantee that the planner can reach the task goal.

## Privileged fixed-bank candidate ceiling — 2026-10-16

To separate learned-transfer failure from planner reachability, the runner now
includes a deliberately privileged `ground_truth_candidate_oracle`. At each
decision it evaluates every candidate in the same fixed action bank with one
exact simulator block, selects the true best immediate successor, and restores
the simulator state before execution. No RGB policy may use this method; it is
only a candidate-bank ceiling.

| panel | method | success | mean final distance |
| --- | --- | ---: | ---: |
| external outcome-20 | `ground_truth_candidate_oracle` | `11/20` | `0.3097` |
| excluded-16 | `ground_truth_candidate_oracle` | `13/16` | `0.2416` |
| combined | `ground_truth_candidate_oracle` | `24/36` | `0.2794` |

The oracle is not perfect on the external panel despite exact one-step
dynamics, while the guarded RGB controller reached `13/20` there. Thus a
portion of the negative result is planner-side: fixed-bank local greedy
selection cannot always reach the goal. The result also prevents a stronger
but incorrect conclusion that all remaining failures are representation
failures. The next control must be a true multi-step perfect-dynamics or
subgoal planner on the same roots; only the residual between that ceiling and
the learned controller should be attributed to transfer.

Artifacts:

- `experiments/run_live_pointmaze_planning.py`;
- `results/pointmaze_rgb_online_planning_v18_cuda_external20_oracle/report.json`;
- `results/pointmaze_rgb_online_planning_v19_cuda_excluded16_oracle/report.json`.

## Exact-planner protocol audit and root-bank correction — 2026-10-16

The initial finite-horizon implementation exposed two experimental confounds.
First, the root action bank was regenerated from the privileged simulator state,
while the established fixed-bank oracle used the RGB-calibrated observed
position. In this contact-rich environment, a small angular change can select a
different wall-contact outcome. Second, the initial finite-horizon runs used a
12-block execution budget, whereas the established oracle reports used 24
blocks. Comparing their final success rates would therefore confound action
selection with episode budget.

The code now takes an explicit `root_current` position. The root candidate bank
is identical to the observed fixed bank; exact simulator positions are used
only for successor expansion. The depth-1/beam-32, 24-block run then exactly
reproduced the established external-20 oracle:

| method | budget | success | mean final distance | mean blocks |
| --- | ---: | ---: | ---: | ---: |
| fixed-bank exact oracle | 24 | `11/20` | `0.3097` | `21.05` |
| corrected exact MPC, depth 1 / beam 32 | 24 | `11/20` | `0.3097` | `21.05` |

This closes an implementation/protocol negative result rather than improving
the learned model. The preliminary depth-4/beam-8 runs with 12 blocks remain
diagnostic smoke artifacts only (`2/20`, distance `1.4335` on external-20;
`5/16`, distance `0.9847` on excluded-16). A later matched 24-block run is
reported below; the smoke artifacts are not used in that comparison.

The correction is covered by the live-planning regression, which now has
`21 passed` together with the task-contrast tests. Artifacts:

- `experiments/run_live_pointmaze_planning.py`;
- `tests/test_live_pointmaze_planning.py`;
- `results/pointmaze_rgb_online_planning_v24_cuda_external20_perfect_mpc_d1_b32_rootfixed_max24/report.json`.

## World-fixed bank root cause and decision-repair result — 2026-10-16

The previous indexed canonical-action gain was subjected to Bank-only,
No-dynamics, within-root Shuffled-prediction, and unseen-candidate-ID controls.
The original `+0.1090` regret reduction was reproduced against the old
`source_plus_delta` baseline, but Full did not beat Bank-only or No-dynamics;
its consistency was significantly worse than Bank-only. The result was a
goal-relative candidate-bank shortcut, not learned-dynamics evidence.

The repaired benchmark reuses one world-coordinate action bank on every root,
so candidate identity no longer encodes the goal direction. On this benchmark,
the previous linear task readout collapsed: Full, Bank-only, No-dynamics and
Shuffled-prediction all had outcome regret `0.2062` in the initial 40-root
pilot. The missing function was then localized to a state-action interaction:
the earlier interaction family contained `source x delta` and `delta x action`
but omitted `source x action`, which is necessary for root-conditioned ranking
of a shared world-frame bank.

A fixed train-only bilinear ridge readout with terms
`[action, source x action]` was evaluated under a new source-order-independent
hash split over 100 episode-disjoint roots and two checkpoint contexts. Its
release gate used only five-fold root-disjoint train OOF regret and the sign of
calibration regret; outcome was never used for selection. The gate selected
the No-dynamics bilinear adapter and rejected the predicted-dynamics extension.

| outcome-20 method | mean regret | pairwise consistency |
| --- | ---: | ---: |
| action-only | `0.1103` | `0.5594` |
| gated No-dynamics bilinear | `0.0476` | `0.6735` |
| Full predicted-dynamics bilinear | `0.0416` | `0.6873` |
| Shuffled-prediction bilinear | `0.0397` | `0.6885` |

For the released No-dynamics adapter, held-out-panel regret changed by `-0.0628` with paired
95% CI `[-0.1196,-0.0130]`, while consistency changed by `+0.1141` with CI
`[+0.0315,+0.2080]`. Train OOF regret changed by `-0.0215`, CI
`[-0.0419,-0.0019]`, and calibration regret had the same sign (`-0.0136`).
Because Full did not beat No-dynamics and Shuffled-prediction was at least as
good, this is a positive frozen-RGB-latent **decision repair** result, not a
positive learned-dynamics result. The same gate and outcome values were
reproduced when jointly fitting four checkpoint contexts from two predictor
architectures (`jepa_final_mlp` and `jepa_final_standardized_mlp`); the Full
versus No-dynamics train-OOF CI still crossed zero. This is useful
architecture-repeated evidence, but all contexts share the same PointMaze
roots and simulator labels. Because these roots were already visible during
method development, the hash-split outcome is exploratory rather than a final
untouched test. The method is simulator-oracle-supervised and still requires a
new environment/root collection and online consequence before becoming a
planning claim.

Artifacts:

- `experiments/decision_benchmark/generate_candidate_bank.py`;
- `experiments/gdd_wm/audit_train_only_teacher_capacity.py`;
- `results/decision_benchmark_candidate_bank_20261016/worldfixed100_hashsplit_decision_repair_v9_s20261016_cuda/capacity_audit.json`;
- `results/decision_benchmark_candidate_bank_20261016/worldfixed100_hashsplit_decision_repair_v9_s20261016_cuda/capacity_readout.npz`.
- `results/decision_benchmark_candidate_bank_20261016/worldfixed100_hashsplit_2arch4context_decision_repair_s20261016_cuda/capacity_audit.json`.

## Matched depth-4 reachability ceiling — 2026-10-16

The previously blocked depth-4/beam-8 exact-dynamics run is now complete on the
same external 20 roots and the same 24-block budget as depth-1/beam-32. The
runner also now reports both exact-dynamics methods as privileged and sets
`runtime_state_access=true`; the earlier depth-1 report incorrectly omitted
that metadata even though successor expansion used simulator state.

| exact-dynamics ceiling | success | mean final distance | mean blocks |
| --- | ---: | ---: | ---: |
| depth 1 / beam 32 | `11/20` | `0.3097` | `21.05` |
| depth 4 / beam 8 | `12/20` | `0.2967` | `21.85` |

The episode-paired success difference was `+0.05`, bootstrap 95% CI
`[-0.20,+0.30]`; the final-distance improvement was `+0.0130`, CI
`[-0.0886,+0.1094]`. Thus increasing privileged search depth did not produce a
stable gain. Planner reachability is now measured rather than blocked, but it
does not explain or repair the frozen predicted-dynamics failure.

Artifacts:

- `results/pointmaze_rgb_online_planning_v25_cuda_external20_perfect_mpc_d4_b8_rootfixed_max24/report.json`;
- `results/pointmaze_rgb_online_planning_v26_cuda_external20_perfect_mpc_d1_b32_rootfixed_max24_privfixed/report.json`;
- `results/pointmaze_rgb_online_planning_v25_cuda_external20_perfect_mpc_d4_b8_rootfixed_max24/paired_vs_d1_privfixed.json`;
- `experiments/compare_pointmaze_planner_depths.py`.

## H-JEPA insight integration and Reacher 128-root replication — 2026-10-16

The H-JEPA research note suggests a sharper interpretation of the current
failure pattern: information value is conditional on prediction horizon and
planning phase. Fine action-effect state can be necessary locally while a
different, slower task geometry is useful for distant goals. Slow
predictability alone remains insufficient because an action-irrelevant scene
identity can also be stable. The resulting proposed object is therefore
phase- and horizon-conditioned transfer sufficiency, not an unconditional
hierarchical or universal world-model claim. The complete protocol is in
`docs/PHASE_HORIZON_CONDITIONED_TRANSFER_SUFFICIENCY_20261016.md`.

The frozen capacity protocol was then rerun on the deployment-aligned
128-root Reacher panel on CUDA. It used the unchanged `[64,32]` source-action
MLP, 400 epochs, seed `20262250`, second-replan roots, final-CEM candidates,
and the original raw-control/absolute-cost diagnostic chart. The split was
64 train, 32 calibration and 32 untouched outcome roots.

On the outcome roots, action-only regret was `0.5319`; source-action MLP regret
was `0.2226`, a paired delta of `-0.3093` with 95% CI
`[-0.4311,-0.1930]`. The full predicted-dynamics MLP had regret `0.3116`, and
shuffled-prediction had regret `0.1919`. Thus the source-action decision signal
replicates at larger sample size, but the current predicted dynamics do not
add independent transfer information and can degrade the decision readout.

The exploratory bilinear calibration-only rule selected full dynamics, but
the stricter five-fold train-OOF plus calibration gate rejected both bilinear
repairs because the train-OOF intervals crossed zero. This is direct evidence
that one phase-specific calibration result must not be promoted into a general
transfer certificate. No online Reacher claim is opened by this audit.

Artifact:

- `results/gdd_wm_cross_environment_20261016/reacher_deployment_aligned_panel128_capacity_frozen/capacity_audit.json`.

### Fresh online consequence and proposal-support repair

The 128-root source-action artifact was then evaluated against Euclidean
top-1 execution on 64 fresh roots (`20263000`--`20263063`). Success changed
from `31/64` to `33/64`, but the paired success CI crossed zero
(`[-0.0625,+0.1250]`) and final qpos error worsened by `+0.0514`, 95% CI
`[+0.0116,+0.0959]`. There were six readout-only wins, four baseline-only
wins and 54 ties. This is a failed independent online consequence despite the
strong offline regret result.

The initially reported `4.30%` proposal-support rate was subsequently found to
be an invalid denominator: the numerator counted candidates only on active
final-CEM/second-replan rows, while the denominator included inactive planner
calls and iterations. It must not be used as evidence that deployment support
was only four percent. The old `98.44%` empirical-quantile gate was also too
strict for finite candidate clouds and forced all rows to abstain. Those two
artifacts remain useful fail-closed smoke tests, but not a causal diagnosis.

The repaired action chart is `proposal_relative`: each 12-dimensional action
summary is standardized by the mean and standard deviation of its own CEM
candidate cloud. This removes cloud-specific translation and scale while
preserving local action position, and applies the identical transformation at
fit and deployment time. The finite-sample support gate now takes a one-sided
90% Wilson lower bound for each calibration cell and freezes the fifth
percentile of those lower bounds (`0.927105`). It reads calibration action
support only and never reads online outcomes.

On the corresponding CUDA offline audit, action-only regret was `0.47152` and
the proposal-relative source-action MLP reached `0.21495`, paired delta
`-0.25657`, 95% CI `[-0.34151,-0.16521]`. However, full predicted dynamics
reached only `0.25498`, while shuffled prediction reached `0.19389`. Thus the
repair recovers a phase-specific source-action decision signal but still
rejects independent value from the learned predicted dynamics.

Two root-disjoint fresh online batches then used the frozen readout and Wilson
gate without outcome tuning:

| fresh roots | baseline | readout | paired success delta | 95% CI | readout-only / baseline-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| `20263300`--`20263363` | `32/64` | `38/64` | `+0.09375` | `[+0.015625,+0.171875]` | `7/1` |
| `20263400`--`20263463` | `36/64` | `39/64` | `+0.046875` | `[-0.03125,+0.125]` | `5/2` |

The prespecified batch-stratified root bootstrap gives `68/128 -> 77/128`,
paired delta `+0.0703125`, 95% CI `[+0.015625,+0.1328125]`, with `12/3/113`
readout-only wins / baseline-only wins / ties. Both batch point estimates are
positive; the second-minus-first batch contrast is `-0.046875`, CI
`[-0.15625,+0.0625]`. Active-row conditional support is `0.93933` on average
(`0.93359`--`0.94922`), all `64/64` active rows were evaluated, none abstained,
and all changed argmin.

Continuous trajectory quality did not improve reliably: pooled final-qpos
delta is `+0.00921`, CI `[-0.04270,+0.04936]`, and minimum-qpos delta is
`-0.00188`, CI `[-0.00750,+0.00352]`. The valid conclusion is therefore a
fresh Reacher binary-success gain from proposal-relative source-action
decision repair, not a predicted-dynamics repair, universal certificate, or
general continuous-error improvement.

Artifacts and adjudication code:

- `results/gdd_wm_cross_environment_20261016/reacher_deployment_aligned_panel128_capacity_proposal_relative_wilson/capacity_audit.json`;
- `results/gdd_wm_cross_environment_20261016/live_reacher_top1_proposal_relative_wilson_seed20263300_64_paired.json`;
- `results/gdd_wm_cross_environment_20261016/live_reacher_top1_proposal_relative_wilson_seed20263400_64_paired.json`;
- `results/gdd_wm_cross_environment_20261016/live_reacher_top1_proposal_relative_wilson_two_batch_128_stratified.json`;
- `experiments/compare_reacher_capacity_planning.py`;
- `experiments/aggregate_reacher_capacity_batches.py`.

Artifacts and implementation:

- `results/gdd_wm_cross_environment_20261016/live_reacher_top1_panel128_seed20263000_64_paired.json`;
- `results/gdd_wm_cross_environment_20261016/live_reacher_top1_support_gate_seed20263100_64_paired.json`;
- `results/gdd_wm_cross_environment_20261016/reacher_deployment_aligned_panel128_capacity_support_fraction_gate/capacity_readout.npz`;
- `experiments/capacity_readout_runtime.py`;
- `experiments/compare_reacher_capacity_planning.py`.

### Predicted-cost repair attempts: smoothing gain, causal-pairing no-go

The remaining no-dynamics boundary was attacked without changing the frozen
checkpoint. The first intervention blended the proposal-relative source-action
score with the frozen LeWM predicted-to-goal cost at fixed weight `0.5`. A
within-cloud rolled model cost was run as the paired placebo. On 64 fresh roots
(`20263600`--`20263663`), no-dynamics / true fusion / shuffled fusion reached
`39/64`, `34/64`, and `38/64` success. True fusion improved final qpos distance
by `-0.03426`, CI `[-0.06971,-0.00152]`, but reduced success by `-0.078125`, CI
`[-0.171875,+0.015625]`, and did not beat shuffled fusion. The distance gain is
therefore a smoothing effect, not accepted candidate-paired dynamics evidence.

A second conservative intervention allowed model cost to choose only within a
small source-action elite. The formal calibration-only audit tests
`k={2,4,8,16,32,64}` and requires the true tie-break to beat both no-dynamics
and within-cell shuffled cost with a root-bootstrap CI below zero. Every
true-versus-shuffled interval crossed zero, so the legal selection is `k=1`,
fail-closed to no-dynamics. An exploratory fresh 64-root `k=2` control was also
run: no-dynamics / true / shuffled reached `38/64`, `39/64`, and `39/64`;
true-minus-shuffled success was exactly zero with CI `[-0.0625,+0.0625]`, and
all continuous intervals crossed zero.

This localizes the unresolved representation failure. Frozen LeWM cost can
regularize trajectory scale, but its candidate identity does not add reliable
online information after conditioning on proposal-relative source-action
repair. The formal controller therefore remains no-dynamics. A predicted-
dynamics claim now requires retraining or distillation that changes the
candidate-paired action-effect representation, not another runtime weight.

Artifacts:

- `results/gdd_wm_cross_environment_20261016/live_reacher_cost_fusion_seed20263600_64_paired.json`;
- `results/gdd_wm_cross_environment_20261016/reacher_model_cost_elite_selection/calibration_and_outcome.json`;
- `results/gdd_wm_cross_environment_20261016/live_reacher_cost_elite_seed20263700_64_paired.json`;
- `experiments/audit_reacher_cost_elite_selection.py`;
- `experiments/compare_reacher_cost_fusion.py`.

## HTSC finite-sample and candidate-shortcut closure — 2026-10-16

The old eight-root calibration silently used the sample maximum at
`alpha=0.1`, although the valid split-conformal rank is nine. The implementation
now returns `insufficient_calibration` whenever
`ceil((n+1)(1-alpha)) > n`; exact enumeration covers `n=8,10,20,40`.
Candidate certificates also reject adaptive candidate generation explicitly.

A deeper provenance audit found that the candidate bank had been repartitioned
after the predictor was trained. Several nominal calibration/outcome episodes
were predictor-fit episodes. The new `checkpoint_protocol_v1` partition uses
the checkpoint's immutable episode lists and yields 54 train, 14 calibration
and 32 outcome roots. On 1,600 RGB-verified candidate transitions, the legal
frozen fixed-bank bounds cover `31/32` and `32/32` outcome roots in the
root-conditioned and shared-world variants, respectively, while releasing no
action. Thus coverage is empirically supported but useful coverage is zero.

The negative ranking result was then decomposed with mandatory shortcut
controls. In the root-conditioned bank, a candidate ID nearly determines the
goal-directed action; Bank-only regret is `0.0132`, better than the repaired
model's `0.0832`. The apparent shuffled-prediction gain is therefore invalid as
dynamics evidence. A shared world-coordinate bank removes this leakage and
balances temporal-order preferences. A five-seed train-root-only intervention
effect ensemble then obtains outcome regret `0.0328`, versus Bank-only `0.0851`,
No-dynamics `0.2034`, and shuffled prediction `0.1602`. Paired gains and 95%
root-bootstrap intervals are respectively `+0.0523 [0.0139,0.0923]`,
`+0.1706 [0.1189,0.2180]`, and `+0.1275 [0.0945,0.1610]`. The predeclared unseen
candidate-ID gain over shuffled is `+0.0310 [0.0110,0.0535]`; matched-order
gain over Bank-only is `+0.0394 [0.0190,0.0602]`.

This is a valid mechanism positive for intervention-supervised decision
repair, not a frozen-world-model dynamics result. The indirect RGB-successor
adapter overfits train roots, worsens held-out successor RMSE, and still
releases zero roots. The 90% exact-winner and epsilon-regret certificates also
remain too wide because of calibration outliers. The operational verdict is
therefore:

> Know marginal fixed-bank coverage; repair shared-world action ranking;
> abstain from certified online execution.

Primary artifact:
`le-wm-main/results/htsc_checkpoint_provenance_20261016/summary.json`.

## PointMaze offline/online preprocessing repair — 2026-10-16

The PointMaze temporal exporter applied the visual transform in the dataset and
then passed the transformed tensor through `model.encode`, which owns the same
normalization/transform step for online RGB planning. Offline source latents
were therefore generated by a double visual transform, while runtime latents
used a single transform. This invalidates the old offline-to-online interface,
independently of planner quality. The exporter now keeps observations in raw
0--255 form, assigns preprocessing exclusively to `model.encode`, records that
ownership in the protocol, and has a regression test that rejects a transformed
observation dataset.

The repaired 100-root CUDA export preserves the checkpoint-derived
`54/14/32` train/calibration/outcome split. A direct equivalence audit gives
zero L1 and max error between exporter and runtime single-pass encoding. The
old double-pass chart is retained only as a historical diagnostic; its PCA was
fit in the invalid chart and cannot support a deployment claim.

An exploratory rank grid exposes a bias--variance trade-off rather than a legal
winner:

| chart | outcome regret | Bank-only gain, 95% CI | unseen-candidate gain, 95% CI | matched-order gain, 95% CI | full-bank calibration bound | epsilon=0.25 release / violation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| old PCA on repaired latents | `0.04137` | `+0.04371 [-0.00168,+0.08505]` | `+0.03308 [-0.00040,+0.06683]` | `+0.03254 [+0.01423,+0.05216]` | `0.11052` | `24/32`, `0/24` |
| deployment-aligned rank 8 | `0.04046` | `+0.04463 [+0.00141,+0.08795]` | `+0.02713 [+0.00565,+0.05192]` | `+0.02369 [+0.00244,+0.04508]` | `0.30093` | `0/32`, not estimable |
| deployment-aligned rank 16 | `0.04231` | `+0.04277 [-0.00787,+0.08637]` | `-0.02029 [-0.04687,+0.00336]` | `+0.03228 [+0.01121,+0.05473]` | `0.08918` | `28/32`, `2/28` |

Rank 8 passes the Bank-only, No-dynamics, shuffled-prediction, unseen-candidate
and matched-order incremental controls, but its tail bound is too wide for the
predeclared `epsilon=0.25`. Rank 16 restores useful bounded-regret release but
fails Bank-only and unseen-candidate incremental controls. Because outcomes
were inspected while comparing ranks, this grid is exploratory and no rank is
selected as confirmatory evidence.

The corrected verdict remains `Repair but Abstain`: the preprocessing bug is
closed and useful coverage is technically achievable, but no deployment-aligned
chart currently satisfies both the complete counterfactual-control conjunction
and the useful-release gate. This is not yet frozen-LeWM dynamics evidence or
an independent RGB closed-loop consequence.
