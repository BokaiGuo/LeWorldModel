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
