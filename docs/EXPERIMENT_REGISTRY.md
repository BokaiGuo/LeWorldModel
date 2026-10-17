# 实验入口注册表

本表是迁移期的入口索引。分类描述维护责任，不代表科学主张成立。

## Active

| 入口/目录 | 责任 | 状态 |
| --- | --- | --- |
| `le-wm-main/experiments/run_unified_research_protocol.py` | geometry、memory、contact-control 和显式 claim workflow | 正在接入统一 runtime |
| `le-wm-main/experiments/run_full_spectral_transfer.sh` | full-data spectral transfer 兼容入口 | 保留 CLI，逐步变薄 |
| `le-wm-main/experiments/empirical_spectral_audit/` | learned latent 与 transfer diagnostics | active |
| `le-wm-main/experiments/empirical_spectral_audit/run_pointmaze_support_matched_panel.py` | corrected root-disjoint PointMaze support-matched external panel；five-step action clock and repaired labels | 50-checkpoint CUDA panel completed；replay closed，planning/identifiability gate failed |
| `le-wm-main/experiments/decision_benchmark/` | action-effect、ATM/IAEI、task contrast、PDTC、candidate bank、RGB successor、RGB-to-task selective audit | active / gated |
| `le-wm-main/experiments/decision_benchmark/audit_rgb_task_contrast.py` | candidate-collapse diagnostics、train-only joint action support、fail-closed release | active / gated |
| `le-wm-main/experiments/decision_benchmark/train_rgb_candidate_successor_adapter.py` | train-root-only counterfactual RGB successor adapter control；--history-windows supports causal RGB context | intervention-only / not planning evidence |
| `le-wm-main/experiments/decision_benchmark/train_joint_rgb_future_task_adapter.py` | WA-JEPA-inspired causal future/task-effect adapter；支持 asymmetric joint/ranking/preference、sequence temporal-derivative 与 root-relative pair-effect ablations | implemented / partial mechanism evidence / not planning evidence |
| `le-wm-main/experiments/decision_benchmark/verify_rgb_successor.py` | dataset-matched realized RGB successor verification；legacy two-frame and history-conditioned temporal contracts | active / gated |
| `le-wm-main/experiments/run_pusht_mode_aware_shared_candidate.py` | privileged PushT mode/action-effect controller；local observed-effect neighborhood and paired moment-sign probes | active / mechanism-only / pose gate failed |
| `le-wm-main/experiments/run_live_tworoom_planning.py` | frozen RGB TwoRoom CEM versus paired random control | 4-panel, 32-episode CUDA aggregate completed; scoped positive consequence |
| `le-wm-main/experiments/run_live_tworoom_planning.py --candidate-bank` | predeclared action-support bank versus pure Gaussian CEM on frozen RGB TwoRoom | two fresh 32-root CUDA panels: `21/64` bank versus `15/64` Gaussian; support repair scoped to TwoRoom |
| `le-wm-main/experiments/run_live_reacher_planning.py` | frozen RGB Reacher CEM versus paired random control | 4-panel, 32-episode CUDA/EGL aggregate completed; target-geometry scope retained |
| `le-wm-main/experiments/audit_live_reacher_shared_candidate_ranking.py` | root-disjoint Reacher RGB shared-bank action ranking, terminal qpos error and regret | 32-root CUDA/EGL negative ranking boundary completed; contact metric not applicable |
| `le-wm-main/experiments/summarize_reacher_shared_candidate_ranking.py` | pooled predeclared Reacher target-geometry ranking audit | 64-root two-condition aggregate; geometry-conditioned transfer boundary |
| `le-wm-main/experiments/summarize_cross_environment_planning.py` | non-PushT evidence index with native estimands and explicit failure boundaries | TwoRoom/Reacher positive; Cube negative; PointMaze inconclusive; no pooling |
| `le-wm-main/experiments/summarize_prediction_to_planning_gap.py` | joins CUDA frozen prediction/action-effect audits with independent planning consequences | action-effect propagation supported in TwoRoom/Reacher; Cube decision boundary isolated |
| `le-wm-main/experiments/run_live_cube_phase_conditioned_planning.py` | frozen RGB Cube phase-conditioned candidate-support intervention | strict paired CUDA/EGL `0/8`; negative boundary retained |
| `le-wm-main/experiments/audit_cube_rgb_observability.py` | root-disjoint RGB versus frozen-LeWM geometry and one-step block-effect readouts; generic, guided-intervention, and color-geometry controls | 80-root CUDA/EGL mechanism panels completed; sparse contact support and goal-conditioned effect boundary retained; not planning evidence |
| `le-wm-main/experiments/audit_pusht_orientation_response.py` | same-edge zero/+moment/-moment orientation-response identifiability audit；`edge_boundary` edge/fraction protocol；CUDA aggregate diagnostics | active / partial positive / cross-edge transfer gate not passed |
| `le-wm-main/experiments/calibrate_pusht_orientation_gate.py` | calibration-only edge/fraction support gate with root-disjoint transfer audit | active / partial positive / 6–8 strata released per seed |
| `le-wm-main/experiments/gdd_wm/` | GDD-WM D0-D2 proposed workstream | proposed / not yet validated |
| `le-wm-main/experiments/gdd_wm/join_rgb_realized_refs.py` | provenance-safe join of dataset-matched realized RGB references into a copied candidate bank | audit utility / D1 still gated |
| `le-wm-main/experiments/gdd_wm/audit_nonleaking_teacher_control.py` | train-only predicted-latent task-readout control for separating label leakage from transfer failure | diagnostic only / not D-JEPA or planning evidence |
| `le-wm-main/experiments/gdd_wm/audit_transfer_sufficient_feature_families.py` | fixed train-only source/latent/action feature-family comparison | diagnostic only / no stable family yet |
| `le-wm-main/experiments/gdd_wm/audit_feature_family_panel.py` | cross-checkpoint wrapper for the fixed feature-family audit | diagnostic only / no family selection |
| `le-wm-main/experiments/gdd_wm/audit_relative_action_effect_geometry.py` | train-root-centered interaction, low-rank, and pairwise action-effect controls | diagnostic only / no cross-bank positive interval |
| `le-wm-main/experiments/gdd_wm/audit_relative_action_effect_panel.py` | five-bank paired root-bootstrap wrapper for relative-effect controls | diagnostic only / support/identifiability boundary |
| `le-wm-main/experiments/gdd_wm/audit_cross_root_action_effect_consistency.py` | detects raw-action bank mismatch and repeated candidate-pair rank flips across roots | diagnostic only / protocol repair input |
| `le-wm-main/experiments/decision_benchmark/generate_candidate_bank.py` | optional canonical goal-relative action bank generation via `--canonical-action-bank` | protocol repair implemented / fresh CUDA replication completed |
| `le-wm-main/experiments/gdd_wm/aggregate_relative_action_effect_panel.py` | separates root uncertainty from architecture/seed uncertainty for canonical relative-effect panels | diagnostic only / 10-case, five-seed replication completed |
| `le-wm-main/experiments/gdd_wm/aggregate_rgb_successor_panel.py` | aggregates outcome-root frozen RGB successor verification without checkpoint selection | diagnostic only / 10-case replication completed |
| `le-wm-main/experiments/gdd_wm/aggregate_feature_family_panel.py` | aggregates train-only RGB-compatible candidate-selection consequences with paired case/root uncertainty | diagnostic only / estimated-frame repair partial positive |
| `le-wm-main/experiments/gdd_wm/audit_rgb_calibrated_candidate_selection.py` | tests train-only realized-RGB successor calibration before task candidate selection | negative control / prediction-decision gap persists |
| `le-wm-main/experiments/gdd_wm/audit_indexed_canonical_action_geometry.py` | train-derived candidate-id lookup for fixed goal-relative action geometry; removes test-root latent-to-XY reconstruction | 10-case offline positive / RGB-planning gate remains closed |
| `le-wm-main/experiments/gdd_wm/aggregate_indexed_canonical_action_panel.py` | aggregates indexed canonical-action consequences with case/root uncertainty | 2-architecture, 5-seed panel completed; RGB-compatible task consequence completed |
| `le-wm-main/experiments/gdd_wm/audit_rgb_indexed_canonical_task_selection.py` | train-only RGB successor task readout with indexed canonical action bank and root-disjoint outcome evaluation | 10-case offline RGB-compatible positive; closed-loop RGB planning remains gated |
| `le-wm-main/experiments/gdd_wm/aggregate_rgb_indexed_canonical_task_panel.py` | aggregates RGB-compatible indexed-action task consequence with case/root bootstrap | 2-architecture, 5-seed panel completed; simulator-labeled offline evidence |
| `le-wm-main/experiments/run_causal_visual_pusht_world_model.py` | intervention-supervised RGB-only visual world model with contact-frame/torque, contact-angle support, invariant angle readout, and online planning audits | goal-pose label repair + formal 1600/240 CUDA continuation completed; offline translation/effect partial positive, within-contact orientation and stable online-planning gates remain failed |
| `le-wm-main/experiments/run_causal_visual_pusht_world_model.py` (`--contact-interface-dynamics`, `--angle-readout-current-only`) | RGB contact-interface chart and current-only explicit angle-head controls | implemented and CUDA-tested; 1600/240 interface run and interaction pilot remain negative; no planning claim |

## Baseline

| 入口/目录 | 用途 |
| --- | --- |
| `le-wm-main/experiments/theory_spectral_transfer/` | 受控理论、counterexample 和合成诊断 |
| `le-wm-main/experiments/latent_eigenspace_audit.py` | latent export 与历史 baseline |
| `le-wm-main/experiments/planning_alignment_audit.py` | planning-alignment 对照 |

## Legacy / Superseded

PushT contact、Fourier/history memory、pose adapter、pure learned closure 及早期
checkpoint audit 脚本必须保留。它们对应负结果和边界证据，不能因为新协议出现而
删除或静默重分类。新 runner 若替代旧 runner，必须记录 protocol version、输入
映射和语义差异；不能声称结果完全等价，除非有行为回归。

## Utility / Test

`summarize_*.py`、`build_*_evidence_matrix.py`、schema validator 和 `tests/` 属于
utility/test。summary 只汇总明确输入，不拥有新的科学数据选择权。

## 新增入口规则

新增 runner 必须回答：现有 runner 加配置为何不能完成同一实验？如果只是环境、seed、
horizon、checkpoint 或 batch size 改变，应优先新增 protocol/config，而不是新增
`run_*_v2/final/fixed.py`。

## 迁移状态

- P0 文档与注册表：本文件和 `RESEARCH_ENGINEERING_GOVERNANCE_20261014.md`；
- P1 统一 fingerprint/artifact/stage runtime：`experiments/research_runtime/`，已实现并有回归测试；
- Unified ATM→HTSC P0/P1 rerun：`experiments/decision_benchmark/build_unified_p0p1_manifest.py` 汇总 PointMaze task contrast、ATM inverse diagnostics、learned matched panel 和 cross-environment gate，并记录 GPU fail-closed 状态；artifact 位于 `results/unified_decision_transfer_p0p1_20261014/`；
- GDD-WM D1 preference audit：`experiments/gdd_wm/audit_preference_stability.py` 检测 candidate-set 重复上下文、execution consistency、realized-ref support 和 teacher/execution label leakage；v4 为历史 fail-closed artifact，v5 已补齐 realized-ref support 但仍因单一 context 和 teacher label leakage fail-closed；
- P1 unified runner manifest/status：已加入 input lock、code identity、manifest hash、顶层 `run_status.json` 和 stage 状态；
- P1 claim matrix 显式 evidence manifest：已禁止 protocol run 从 visual/counterfactual 历史目录隐式回退，并校验 record checksum；
- P1 full Bash compatibility runner：已接入 run contract、GPU preflight、JSON cache validation 和 dry-run 无副作用；仍保留旧 CLI 兼容面；
- P2 研究域 package 化：尚未开始，保持现有路径以保护复现。

## 2026-10-15 continuation entries

- `le-wm-main/experiments/audit_rgb_analytic_pose.py`: root-disjoint frozen
  RGB mask-moment pose audit with separate current/goal calibration artifacts;
  current pose is partial positive, goal-domain mismatch diagnosed; no planning
  claim.
- `le-wm-main/experiments/audit_rgb_reobserved_action_effect.py`: RGB
  short-pulse successor re-observation and held-out observed-effect adapter;
  local effect is measurable, task ranking/adapter remains unstable; verification
  only.
- `le-wm-main/experiments/aggregate_rgb_paired_reobserve.py`: fixed-protocol
  multi-seed aggregation for same-root zero-action finite-difference effect
  audits; no policy or checkpoint selection authority.
- `le-wm-main/experiments/run_visual_geometry_servo.py` with analytic pose
  calibration flags: frozen RGB pose readout and support-aware geometry servo;
  coordinate bug repaired, but closed-loop result remains `0/8`.
- `le-wm-main/experiments/run_pusht_mode_aware_shared_candidate.py`: CUDA
  root-disjoint mode-aware verification-gate continuation completed at
  `results/negative_cause_diagnosis_20261015/mode_aware_verification_gate_64x32_seed20261015_final/`;
  mode accuracy is high but pose closure and ranking transfer remain failed.
  Batched candidate inference is an execution-only repair. The 8-root pilot is
  retained separately and is not pooled with the 32-root estimate.
- Distance-aware online support is implemented in the same runner and audited
  at `results/negative_cause_diagnosis_20261015/mode_aware_distance_support_64x8_seed20261015_pilot/`;
  the pilot remains negative, so cache-distance restriction is not promoted as
  the causal repair.

- `le-wm-main/experiments/aggregate_pusht_orientation_gate.py`: cross-seed,
  root-disjoint aggregation of calibration-only orientation permissions;
  three-seed intersection is 4/10 strata and is not a universal certificate.
- `le-wm-main/experiments/run_pusht_mode_aware_shared_candidate.py` with
  `--orientation-gate-artifact`: fail-closed orientation-permission control
  ablation on the shared candidate bank. CUDA smoke artifact is
  `results/negative_cause_diagnosis_20261015/orientation_gated_control_smoke_20261018/`;
  fallback coverage improves support safety but pose closure remains failed.

- `le-wm-main/experiments/run_pusht_mode_aware_shared_candidate.py` with
  `--online-effect-correction --paired-effect-probes`: paired local-effect
  excitation smoke at
  `results/negative_cause_diagnosis_20261015/paired_effect_probe_smoke_20261018/`;
  effect error improves locally, but contact retention and stable pose closure
  remain gated.

- `le-wm-main/experiments/run_pusht_mode_aware_shared_candidate.py`: repaired
  the boundary-contact retention definition. For `boundary_contact` roots,
  `contact_loss_rate` now uses polygon boundary gap (`>70 px`); the old
  pusher-to-block-centre predicate is retained as
  `center_distance_loss_rate` and is not a physical contact claim.
- `le-wm-main/experiments/aggregate_pusht_paired_effect_panel.py`: applies the
  same protocol-correct metric and reports the legacy centre-distance
  diagnostic separately. The three-seed CUDA replication is aggregated at
  `results/negative_cause_diagnosis_20261015/contact_metric_repair_panel_20261024/`.
  Physical contact loss is `0/48` for all variants; pose closure remains
  unresolved. This is a measurement repair plus a mechanism result, not a
  stable RGB planning claim.
- `run_pusht_mode_aware_shared_candidate.py --terminal-handoff`: the first
  artifact exposed an execution-order bug because adaptive reranking
  overwrote terminal selection. Terminal priority and
  `--terminal-max-moment` are now implemented; the corrected same-seed
  artifact is
  `results/negative_cause_diagnosis_20261015/terminal_max_moment_metric_repair_seed20261020_v2/`.
  It preserves physical contact but remains `2/16` pose success, so the
  stronger terminal pulse is not promoted.

- `run_pusht_mode_aware_shared_candidate.py` with the frozen contact-frame
  interface, paired probes, online local-effect correction, and boundary
  contact protocol has a three-seed CUDA replication at
  `results/negative_cause_diagnosis_20261015/contact_frame_boundary_panel_20261015/`.
  The 48-root aggregation reports adaptive effect-error delta `-0.04497`
  (95% CI `[-0.05848,-0.03224]`) and wrong-sign delta `-3.0625`
  (95% CI `[-5.4167,-0.9375]`) versus global one-step, with physical contact
  loss `0/48`. Pose success remains only `8.33%` and final-angle CI crosses
  zero; this is a stable local mechanism positive, not stable pose closure or
  visual planning evidence.

- `experiments/run_pusht_repaired_predictor_mpc.py` with the frozen
  horizon-conditioned short-pulse protocol (`h=4`, `action_block=1`, visual
  state source) completed a three-seed CUDA panel at
  `results/pusht/lewm/h4_shortpulse_visual_mpc_panel_20261015/`.
  Base and repaired arms both have `0/24` task success; repaired-minus-base
  block XY delta is `+16.26 px` and final-distance delta is `+383.68`, with
  no downstream positive. This closes the horizon-only repair hypothesis and
  keeps the active work on action-support/contact-interface transfer.

- `run_pusht_mode_aware_shared_candidate.py` now has a strict active paired
  probe path: stale zero-angle cache entries are rejected, opposite-moment
  pulses are matched in the local action chart, and insufficient support
  forces fail-closed probing. The three-seed CUDA panel is
  `results/negative_cause_diagnosis_20261015/active_paired_probe_panel_20261015/`.
  Adaptive effect error improves, but angle closure is worse and pose success
  is unchanged; retain as a mechanism positive, not a planning claim.
- `run_pusht_mode_aware_shared_candidate.py` now exposes the declared
  `--max-initial-angle-error` support-domain gate and records root-angle/action
  metadata. The CUDA `32/16` gate panel and phase-only follow-up show zero
  physical contact loss and lower angle error in some controllers, but no
  stable strict-pose gain; retain as a diagnosis of the
  within-contact translation/rotation conflict, not as adapter closure.
- The same runner now reports a state-only `staged` arm that switches from
  goal-conditioned translation ranking to mode-aware orientation ranking at a
  frozen `28 px` XY gate. The CUDA panel gives a point-estimate `5/16` pose
  success but overlapping uncertainty; adding explicit goal-translation
  candidates gives `2/16`. Retain both as mechanism/candidate-support audits,
  not stable pose closure.
- The same runner now includes a budget-matched fixed two-pulse macro:
  translation pulse, fresh observation, then orientation pulse. The CUDA
  `32/16` re-observation panel is
  `results/negative_cause_diagnosis_20261015/two_pulse_macro_reobserve_cuda_panel_32x16_seed20261029/`.
  Staged macro reduces predicted/observed effect error to `0.1611` and final
  XY to `29.69 px`, but remains `4/16` strict pose success. A second-pulse
  moment-sign feedback arm remains `4/16`; the frozen `28 px` phase-gated arm
  remains `4/16` while reaching effect error `0.1525`. These are mechanism
  positives and failure-localization evidence, not stable pose closure.
- A training-only joint two-pulse final-pose ranker was added to test whether
  one-step composition, rather than latent action transfer, caused the
  failure. The first run exposed a real pulse-length mismatch; training and
  deployment are now matched at `macro_pulse_steps`. The corrected CUDA
  artifact is
  `results/negative_cause_diagnosis_20261015/joint_macro_ranker_matched_cuda_panel_32x16_seed20261029/`.
  It reaches `5/16` pose success, final XY `30.16 px`, final angle `0.528 rad`,
  and effect error `0.157`. The implementation mismatch is repaired, but no
  independent planning gain appears; retain this as a protocol diagnosis and
  multi-step mechanism result.
- `run_pusht_mode_aware_shared_candidate.py` now includes a fixed successor-free
  `xy_shield_mask` for the orientation half of the two-pulse macro. It rejects
  learned orientation candidates whose predicted XY error increases by more
  than `2 px`, and falls back to the least damaging predicted-XY candidate
  when the support gate is empty. The three-seed CUDA panel is aggregated at
  `results/negative_cause_diagnosis_20261015/xy_shield_panel_3seed_20261015/`.
  It reduces pooled final XY from `37.29` to `18.44 px` with paired 95% CI
  `[-27.88,-10.09]`, while increasing final angle by `0.152 rad` and leaving
  strict pose at `15/48` (paired delta CI `[-0.104,+0.104]`). This is a
  reproducible translation-preservation mechanism result, not stable pose
  closure; the remaining bottleneck is within-contact angular action support
  and identifiability.
- The runner now includes a training-root-only signed orientation-delta ranker,
  a shielded control, and a fixed equal-weight absolute-plus-delta dual ranker.
  CUDA smoke artifacts are
  `results/negative_cause_diagnosis_20261015/orientation_delta_ranker_smoke_cuda_20261015/`,
  `results/negative_cause_diagnosis_20261015/orientation_delta_ranker_shielded_smoke_cuda_20261015/`,
  and `results/negative_cause_diagnosis_20261015/orientation_dual_ranker_smoke_cuda_20261015/`.
  Signed deltas improve angle error locally but worsen XY; shielding partially
  restores XY, while the dual arm reproduces the absolute ranker. None gives
  an independent strict-pose gain, so no formal multi-seed claim is authorized.

- Non-PushT continuation: the fixed epoch-50 JEPA-WM PointMaze checkpoint was
  evaluated on the tail holdout (episodes 1800--1899; 100 episodes, two
  windows per episode, horizons 1/2/4) on host CUDA. Artifacts are
  `results/pointmaze_cuda_temporal_followup_20261015/{observed,zero,reverse}.json`.
  Observed-action latent prediction error is `4.974/4.900/4.711` and the
  covariance-predictor commutator is `29.38/32.84/59.42`; zero-action gives
  `4.999/4.856/4.620` and `18.15/21.63/45.39`; reverse-action gives
  `4.996/4.902/4.732` and `37.60/50.37/70.65`. This confirms an
  action-conditioned geometry signal outside PushT, but is not a planning or
  universal-certificate claim.
- Expanded PushT normal-bank smoke was completed on CUDA at
  `results/negative_cause_diagnosis_20261015/expanded_normal_bank_smoke_cuda_20261015/`.
  Normal magnitudes `0.04/0.08` improve final XY (`45.38 -> 14.48 px`) but
  do not improve strict pose (`1/8 -> 0/8`) and worsen final angle
  (`1.629 -> 1.781 rad`), with physical contact loss `0/8`. The branch is
  closed as a translation-only mechanism result; no formal multi-seed rerun
  is authorized.
- Live non-PushT frozen-checkpoint audits completed on TwoRoom and Reacher at
  `results/live_cross_environment_prediction_20261015/{tworoom_cuda,reacher_cuda}/`.
  In-domain zero-minus-observed MSE is positive at h1/h2/h4 for TwoRoom
  (`+0.285/+0.469/+0.466`) and Reacher (`+0.494/+0.903/+1.155`), with
  calibration/test episode separation and CUDA inference. These are
  action-conditioned prediction diagnostics, not planning evidence.
- Same-checkpoint cross-environment controls are at
  `results/live_cross_environment_prediction_20261015/{tworoom_checkpoint_cross_env_cuda,reacher_checkpoint_cross_env_cuda}/`.
  TwoRoom checkpoint -> Reacher has observed h1/h4 MSE `6.97/8.22` and
  zero-minus-observed deltas `-0.245/-0.517`; Reacher checkpoint -> TwoRoom
  has observed h1/h4 MSE `0.46/1.28` and deltas `-0.230/-0.615`. The
  cross-environment failure is registered as an interface/geometry boundary,
  not hidden as a negative result.
- Calibration-only diagonal affine interface tests are at
  `results/live_cross_environment_prediction_20261015/{tworoom_checkpoint_affine_cross_env_cuda,reacher_checkpoint_affine_cross_env_cuda}/`.
  They reduce some absolute MSE but collapse the cross-environment action
  effect toward zero rather than restoring it; this is an under-supported
  192-dimensional calibration diagnostic, not a transfer repair.
- OGB-Cube provides a third non-PushT in-domain control at
  `results/live_cross_environment_prediction_20261015/cube_cuda/`. Its
  zero-minus-observed h1/h2/h4 effects are `-0.009/+0.016/+0.023` and all
  bootstrap intervals cross zero. The evidence supports environment-
  conditional action geometry, not a stable all-environment claim; action
  support and action semantics are the next repair target.
- A calibration-only, identity-regularized linear action-interface adapter is
  implemented in `experiments/audit_frozen_cross_environment_prediction.py`.
  Its independent test artifacts are
  `results/live_cross_environment_prediction_20261015/{tworoom_checkpoint_action_adapter_cross_env_cuda,reacher_checkpoint_action_adapter_cross_env_cuda}/`.
  It moves the two cross-environment negative effects close to zero but not
  reliably positive in either direction; source-domain action effects remain
  positive. This separates partial action-coordinate mismatch from the
  remaining visual/dynamics interface mismatch and is not a planning claim.
- `experiments/dmcontrol_compat.py` adds an explicit runtime compatibility
  shim for the Reacher dm-control/MuJoCo named-field mismatch. The Reacher
  artifact records `device=cuda`, `render_backend=egl`, and the shim's
  missing-field list; no simulator state is synthesized by the shim.
- The same-domain OGB-Cube action-adapter control is at
  `results/live_cross_environment_prediction_20261015/cube_action_adapter_cuda/`.
  It improves the zero-minus-observed effect from raw
  `+0.018/+0.050/+0.039` to adapted `+0.039/+0.082/+0.161` at h1/h2/h4,
  with all adapted bootstrap intervals excluding zero. This is a promising
  non-PushT prediction result, not yet a stable or planning-level claim.
- Two independent replications are additionally stored at
  `results/live_cross_environment_prediction_20261015/{cube_action_adapter_seed2_cuda,cube_action_adapter_seed3_cuda}/`.
  The adapted h1/h2 effects are positive with bootstrap intervals excluding
  zero in all three Cube seeds; h4 remains positive in point estimate but one
  of three intervals crosses zero. The current stable claim is therefore
  h1/h2 prediction improvement only, with h4 still exploratory.
- Independent live planning is now executed outside PushT. The OGB-Cube
  goal-alias runner is `experiments/run_live_cube_action_adapter_planning.py`;
  its corrected CUDA/EGL paired artifact is
  `results/live_cube_action_adapter_planning_20261015/paired_8ep_canonical/`.
  Baseline and calibration-only action adapter both obtain `0/8`, so the
  Cube prediction gain has no held-out planning consequence under the frozen
  CEM protocol. The TwoRoom live RGB runner is
  `experiments/run_live_tworoom_planning.py`, with canonical artifact
  `results/live_nonpusht_planning_20261015/tworoom_paired_8ep_canonical/`.
  Frozen LeWM is `2/8` versus random `1/8`; paired delta is `+0.125` with
  one win, zero losses, and seven ties. This is non-PushT evidence but not a
  stable general planning claim; the report records the bootstrap interval
  and keeps all universal claims closed.
- TwoRoom was then replicated on three non-overlapping 8-episode CUDA
  panels with fixed CEM and same-seed random controls. The aggregate is
  `results/live_nonpusht_planning_20261015/tworoom_aggregate_24ep/planning_aggregate.json`:
  random `2/24`, frozen LeWM `7/24`, paired `5/0/19` wins/losses/ties,
  delta `+0.2083`, bootstrap CI `[+0.0417,+0.3750]`, exact sign-test
  `p=0.03125`. This is the first replicated non-PushT planning consequence,
  scoped to TwoRoom; Cube remains `0/8` versus `0/8`, and no universal claim
  is opened.
- Reacher live RGB planning was added outside PushT at
  `experiments/run_live_reacher_planning.py`, with evaluator-side rendered
  qpos-match goal images and no qpos/qvel/target state in policy inputs. Three
  fixed 8-episode CUDA panels are stored at
  `results/live_nonpusht_planning_20261015/reacher_panel_seed20261015/`,
  `reacher_panel_seed20261023/`, and `reacher_panel_seed20261031/`; the
  aggregate is
  `results/live_nonpusht_planning_20261015/reacher_aggregate_24ep/planning_aggregate.json`.
  Random is `0/24`, frozen LeWM `16/24`, with paired `16/0/8` wins/losses/ties,
  delta `+0.6667`, bootstrap CI `[+0.4583,+0.8333]`, and exact sign-test
  `p=1.53e-05`. This is a Reacher-scoped live planning consequence. The
  official HDF5 dataset remains unavailable because the host network returned
  `Network is unreachable`; no dataset-replay claim is made.
- Reacher target-geometry robustness was tested with the same runner and two
  fresh unseen target offsets, plus one fresh panel for the original offset.
  The condition aggregate is
  `results/live_nonpusht_planning_20261015/reacher_target_geometry_aggregate/target_geometry_aggregate.json`:
  pooled random `1/24`, frozen LeWM `6/24`, paired `6/1/17` wins/losses/ties,
  delta `+0.2083`, bootstrap CI `[0.0000,+0.4167]`, exact sign-test
  `p=0.0625`. The point estimate stays positive across all three directions,
  but condition-level intervals cross zero; retain this as a robustness
  boundary, not as a universal planning result.
- Cube's former `receding_horizon=1` concern was closed with an official
  protocol-matched CUDA/EGL panel at
  `results/live_cube_action_adapter_planning_20261015/receding5_official_8ep/`:
  horizon `5`, receding horizon `5`, action block `5`, CEM `64/8/8`,
  `var_scale=1.0`, and 75 steps. Baseline and calibration-only action
  adapter both obtain `0/8`, with paired `0/0/8` wins/losses/ties. Final
  block-goal distance means are `0.1460 m` and `0.1500 m`, respectively.
  Combined with the prior lower-variance and 150-step traces, this rules out
  protocol mismatch, initial CEM variance alone, and episode budget alone as
  sufficient causes. The remaining boundary is goal-conditioned
  action-to-task transfer: prediction-level adapter gains do not translate to
  closed-loop Cube planning.
- The Cube action interface was audited and repaired as an explicit
  intervention. OGBench expects normalized environment actions in `[-1,1]`,
  but the prior z-score inverse transform could emit unbounded CEM outputs.
  `experiments/run_live_cube_action_adapter_planning.py` and
  `experiments/audit_live_cube_action_ranking.py` now support
  `--clip-env-actions` and record pre-clip fraction and maxima. The smoke
  showed 13.7%/11.7% out-of-range values for baseline/adapter, so this is a
  real interface defect rather than a hypothetical explanation.
- Equal Cube integer seeds were also insufficient for pairing because the
  environment samples its variation space without the episode seed. The live
  runners now snapshot baseline `qpos/qvel` plus variation values and restore
  them for the adapter arm. The strict CUDA/EGL 8-episode panel is
  `results/live_cube_action_adapter_planning_20261015/bounded_matched_8ep_v2/`:
  both arms are `0/8`, all eight final-distance deltas are exactly zero, and
  bounded output still clips 13.43%/13.03% of baseline/adapter values. The
  strict 25-step ranking audit is
  `results/live_cube_action_ranking_20261015/bounded_matched_audit25_8ep/`;
  both arms show approximately `5.2e-9 m` distance change and zero paired
  difference. This closes the interface and pairing confounds but does not
  solve Cube planning; retain Cube as a non-PushT prediction-to-planning
  failure boundary.
- A strict raw-physics Cube action-support intervention is stored at
  `results/cube_action_interventions_20261015_full.json` and implemented in
  `experiments/audit_cube_action_interventions.py`. It covers all five Cube
  tasks and two seeds per task with restored initial state/variation pairing.
  Zero, single-axis, and vertical-only controls produce only gravity settling;
  an explicit lateral-align + descend + close + push sequence produces
  nontrivial cube motion in `3/10` cases and demonstrates that the environment
  can express a task-relevant effect. This is a mechanism diagnostic, not a
  learned-planning success, and localizes the remaining Cube negative to
  action-support/approach-phase coverage plus goal-conditioned action ranking.
- The phase-conditioned candidate-bank intervention is implemented in
  `experiments/run_live_cube_phase_conditioned_planning.py` and recorded at
  `results/live_cube_phase_conditioned_planning_20261015/paired_8ep/planning.json`.
  It uses four generic x/y align-descend-close-push candidates, frozen-model
  cost ranking, CUDA/EGL, exact initial-state pairing, and the official Cube
  CEM budget. The bank was selected 24 times, but both baseline and
  intervention remain `0/8`; paired success delta is zero and mean final
  distance is `+0.00114 m` for the intervention. A short-pulse smoke also
  ties the baseline. This closes the candidate-support-only explanation and
  leaves goal-conditioned contact action ranking as the primary Cube boundary.

- Cube goal-conditioned macro-ranking follow-up (2026-10-17): a root-disjoint
  RGB macro audit tested a 25-candidate bank with 8 directions, three approach
  durations, and 40-step macros. It underperformed the predeclared short
  contact protocol (`mean regret=0.0202 m`, oracle agreement `53.1%`). A fresh
  40-calibration/32-held-out replication of the short 20-step, 4-direction
  bank reduced regret to `0.00258 m` and raised oracle agreement to `93.75%`.
  This makes contact phase and macro duration an identifiable design condition,
  not evidence that larger candidate banks are better. The result is an
  offline RGB mechanism audit and has not been promoted to online Cube
  planning.

- Full TwoRoom CUDA latent/planning alignment is complete. The phase-1
  artifact is `results/tworoom_full_spectral_cuda_20261015/full_latents/`
  (`1,592,619` collected samples; `200,000` rich rows retained), and phase 2
  contains `3,716,111` endpoint rows. Vanilla latent cost has Spearman
  `0.6653` with state distance, `0.2034` with goal distance, and `-0.6621`
  with the success proxy. Spectral top-32 gives `0.6560`, `0.2393`, and
  `-0.6490`, respectively. The paired empirical audit reports dynamics lift
  `24.33/17.38/10.41` and planning lift `15.97/11.12/6.77` at `k=4/8/16`,
  against nuisance lift `9.50/7.31/4.86`; predictor spectral mean cosine is
  `0.9605`. These are offline diagnostics, not a universal planning result.
  The result closes the “latent has no dynamics signal” explanation but leaves
  goal-conditioned action ranking as the next required intervention.

- The new root-disjoint TwoRoom goal/action transfer audit is implemented in
  `experiments/audit_tworoom_goal_action_transfer.py` and stored at
  `results/tworoom_full_spectral_cuda_20261015/full_latents/goal_action_transfer_audit.json`.
  On the held-out split, latent-action improves next-state RMSE (`0.0211`)
  over state-action (`0.0251`) and endpoint-goal progress RMSE (`0.0197`)
  over (`0.0228`), but adding the window endpoint goal gives no material gain
  and predictor-effect-goal-action is worse (`0.0222`). The action-vs-zero
  contrast is positive on `62.3%` of rows, but the protocol has no shared
  counterfactual candidate bank and is not a causal ranking or online-planning
  result. This closes the simple “concatenate goal” repair and points to a
  shared-candidate or fresh live visual intervention.

- A fresh CUDA TwoRoom receding-horizon panel using the compatible
  `horizon=5/action_block=5/receding_horizon=1` protocol is stored at
  `results/live_nonpusht_planning_20261015/tworoom_receding1_seed20261047/`.
  LeWM is `3/8` versus random `0/8` (`3/0/5` paired). The two-panel aggregate
  is `4/16` wins, no losses, delta `+0.25`, bootstrap CI `[+0.0625,+0.50]`,
  but exact sign-test `p=0.0625`; retain it as a positive but not yet
  significant scoped result. The matched `receding_horizon=5` run on the same
  fresh seeds has the identical LeWM success vector, so short feedback is not
  isolated as the cause. The live runner now rejects incompatible
  `action_block` values before model inference.

- The matched TwoRoom candidate-bank intervention is implemented in
  `experiments/run_live_tworoom_planning.py` and summarized by
  `experiments/summarize_tworoom_candidate_bank.py` at
  `results/live_nonpusht_planning_20261015/tworoom_candidate_bank_matched/pairwise.json`.
  The fixed bank uses zero, axis-aligned magnitudes `0.25/0.50/0.75`, and
  four diagonals within the same 64-candidate CEM budget. Across 16 paired
  episodes it yields `0/16` wins, `1/16` losses, `15/16` ties versus standard
  CEM (delta `-0.0625`, CI `[-0.1875,0]`). Generic candidate coverage is
  therefore not a demonstrated repair; the next target is the latent
  ranking/utility interface.

- A frozen covariance-spectral utility intervention is implemented in
  `experiments/run_live_tworoom_planning.py` and summarized by
  `experiments/summarize_tworoom_utility_intervention.py` at
  `results/live_nonpusht_planning_20261015/tworoom_spectral32_matched/pairwise.json`.
  It replaces terminal full-dimensional latent MSE with a top-32 covariance
  projection while keeping the checkpoint, CEM budget, RGB inputs, and seeds
  fixed. Across 16 paired episodes it yields `0/16` wins, `0/16` losses, and
  `16/16` ties. A simple spectral utility projection is therefore not a
  demonstrated repair; task/action-conditioned utility remains open.

- A train-split latent-to-physical readout was fitted by
  `experiments/fit_tworoom_physical_readout.py`. Its episode-disjoint test
  RMSE is `0.0864` against target std `1.0194`, but using it as a frozen live
  terminal utility loses two of 16 matched episodes relative to Euclidean
  utility (`0/16` wins, `2/16` losses, delta `-0.125`). Artifacts:
  `results/tworoom_full_spectral_cuda_20261015/full_latents/physical_readout/readout.json`
  and
  `results/live_nonpusht_planning_20261015/tworoom_physical_readout_matched/pairwise.json`.
  This closes the simple “decode physical state then plan in it” repair and
  reinforces that action-conditioned effect modeling is required.

- The action-conditioned physical successor readout is implemented in
  `experiments/fit_tworoom_action_conditioned_readout.py`. Its held-out RMSE
  improves from `0.6700` to `0.6474` after adding the action block, but the
  frozen live utility loses two of 16 matched episodes versus default
  Euclidean cost (`0/16` wins, `2/16` losses, delta `-0.125`). Artifacts:
  `results/tworoom_full_spectral_cuda_20261015/full_latents/action_conditioned_readout/readout.json`
  and
  `results/live_nonpusht_planning_20261015/tworoom_action_conditioned_matched/pairwise.json`.
  This separates prediction/readout improvement from candidate-selection
  improvement and makes shared-candidate ranking regret the next gate.

- The shared-candidate ranking audit is implemented in
  `experiments/audit_tworoom_shared_candidate_ranking.py`. A first pass used
  an invalid repeated three-frame input; live `WorldModelPolicy` actually
  supplies one current frame/action. The audit now defaults to `history_len=1`,
  records this contract, and was rerun on 32 fresh roots. The corrected
  constant-bank result is top-1 `34.4%`, top-4 `53.1%`, mean rank `7.41/16`,
  mean Spearman `-0.004`, and mean regret `42.20`.

- The multi-stage support control is implemented in
  `experiments/audit_tworoom_sequence_candidate_ranking.py`. With 32 shared
  sequences and 32 fresh roots, the frozen RGB ranker obtains top-1 `18.8%`,
  top-4 `31.3%`, mean Spearman `0.050`, and mean regret `31.88`. This changes
  candidate support but does not demonstrate reliable action ranking.

- The calibration-only diagnostic is implemented in
  `experiments/audit_tworoom_calibrated_action_ranking.py`. On an independent
  fresh 16-root held-out split, calibration worsens mean regret from `49.39`
  to `56.23` and top-1 from `31.25%` to `12.5%`; it is not used by the live
  controller. Together these audits localize the unresolved negative to the
  frozen goal-conditioned action-effect utility rather than generic candidate
  coverage or latent physical decodability. These are offline diagnostics
  with simulator labels, not universal transfer or online-planning evidence.

- `experiments/audit_tworoom_latent_action_ranker.py` implements a frozen
  latent/action task-readout diagnostic and saves a calibration artifact;
  `experiments/run_live_tworoom_planning.py` can load it as
  `--cost-metric latent_action_ranker`. Two fresh 16-root offline panels show
  lower mean replay regret but unstable exact top-1 selection. The independent
  8-episode RGB-only matched consequence is `0/8` wins, `1/8` loss, `7/8`
  ties versus Euclidean utility (delta `-0.125`), stored at
  `results/live_nonpusht_planning_20261015/tworoom_latent_action_ranker_candidatebank_matched`.
  The artifact is therefore diagnostic only: offline task-ranking gains do
  not establish a closed-loop planning gain.

- GPU rerun of the non-leaking GDD-WM control is stored at
  `results/gdd_wm_candidate_effect_cuda_20261015/`. On the complete-history
  `jepa_history_near_goal_v5_source` bank, train-root-only
  `history_latent_action` gives held-out effect RMSE `0.3358`, mean paired
  regret gain `+0.1243`, and paired success gain `+0.125`; the same-split
  `frozen_latent_action` control gives `0.3518`, `+0.0998`, and `+0.025`.
  This is an offline intervention-supervised decision-transfer control only:
  the source manifest has no realized visual latent for the new sequences,
  and the D0/D1 GDD-WM teacher gate remains blocked.

- A fresh non-PushT CUDA replication with seed `20261100` was completed using
  the unchanged frozen RGB protocols. The new panels are
  `results/live_nonpusht_planning_20261015/{tworoom_panel_seed20261100,reacher_panel_seed20261100}`;
  the five-panel aggregates are
  `tworoom_aggregate_40ep/planning_aggregate.json` and
  `reacher_aggregate_40ep/planning_aggregate.json`. Across 40 paired
  episodes, TwoRoom gives `6/40` wins, no losses, delta `+0.15`, bootstrap CI
  `[+0.05,+0.275]`; Reacher gives `24/40` wins, no losses, delta `+0.60`, CI
  `[+0.45,+0.75]`. These are scoped live RGB consequences with frozen
  checkpoints, not universal transfer evidence. The refreshed four-environment
  index is `results/cross_environment_planning_20261015/nonpusht_evidence_40ep.json`;
  Cube remains a negative boundary and PointMaze remains inconclusive.

- The predeclared Reacher target-geometry robustness panel was expanded with
  three fresh CUDA/EGL runs (`reacher_target_delta_pp_20261201`,
  `reacher_target_delta_np_20261209`, `reacher_target_delta_pn_20261217`).
  The six-panel, 48-episode aggregate is
  `results/live_nonpusht_planning_20261015/reacher_target_geometry_aggregate_48ep/target_geometry_aggregate.json`:
  `11` wins, `1` loss, delta `+0.2083`, bootstrap CI `[+0.0833,+0.3333]`,
  exact sign-test `p=0.00317`. Per-condition intervals remain wide and touch
  zero, so this is scoped target-family robustness rather than a universal
  planning claim.

- The realized-RGB successor verifier was aggregated over 2 architectures × 5
  seeds × 8 outcome roots in
  `results/gdd_wm_rgb_successor_panel_20261015/aggregate.json`. All 10 cases
  pass renderer, realized-RGB, and projected-successor gates. The candidate
  better-than-source fraction is `0.6656` with two-level case/root CI
  `[0.5977,0.7305]`; relative L2 improvement is `0.1421` with CI
  `[0.0849,0.1965]`. This closes only the realized-successor subgate; D1
  remains fail-closed because independent teacher preference contexts and
  leakage-free teacher provenance are still missing.

- A CUDA PointMaze temporal/action-intervention panel is stored at
  `results/nonpusht_pointmaze_temporal_20261015/`. It uses the frozen JEPA-WM
  checkpoint, tail holdout episodes `1800--1839`, and matched observed,
  zero-action, and cross-episode-shuffled action conditions. At h=1,
  observed actions improve latent MSE over zero by `0.0305` and over shuffled
  actions by `0.0082`; at h=2 and h=4 the observed condition is worse than
  both controls. The structured summary is
  `action_intervention_summary.json`. This is a non-PushT horizon-dependent
  action-conditioning boundary, not evidence for stable action
  identifiability, closed-loop planning, or a commutator certificate.

- The remaining disjoint PointMaze tail (`1840--1899`, 60 episodes) was run
  under the same CUDA protocol and pooled with the first 40-episode panel.
  The aggregate is
  `results/nonpusht_pointmaze_temporal_20261015/action_intervention_panel_aggregate_100ep.json`.
  Observed actions improve over zero at h=1 (`-0.0309` MSE) but are worse at
  h=2 (`+0.0265`) and h=4 (`+0.0534`); the corresponding observed-minus-
  shuffled deltas are `-0.0103/+0.0260/+0.0263`. The two holdout panels have
  opposite h=2/4 signs, so the formal conclusion is horizon-conditioned
  action-signal heterogeneity, not stable multi-step action identifiability.

- GDD-WM D1 repeated-context construction is implemented in
  `experiments/gdd_wm/build_repeated_candidate_contexts.py`. It matches
  independently generated candidate banks by exact root/action hash and
  preserves context provenance. The resulting artifact has 2 contexts, 40
  roots, 410 matched root/action pairs, and 820 records. D1 now reports
  estimable stability (`1,930` repeated pair-context comparisons, flip rate
  `0.0`), but remains `rejected_fail_closed` because the teacher is
  `offline_environment_cost_v1`, its score is exactly negative real task
  cost, and 164 outcome/test records lack complete runtime references. This
  repairs the experimental structure only; it does not unlock D2 or validate
  a D-JEPA teacher.

- The D0 builder now performs an early exact-leakage preflight in addition to
  schema checks. On the v5 RGB-reference table it detects 640 numeric
  teacher/cost pairs with `teacher_score = -real_task_cost`, records
  `teacher_execution_validation=rejected_exact_negative_cost_leakage`, and
  sets `d1_ready=false` while retaining `d0_schema_complete=true`. This is a
  fail-closed guard against promoting an execution-derived oracle as a
  teacher.

- Fresh non-PushT CUDA replication used two disjoint 32-episode panels per
  environment (`seed20261083/20261120` for TwoRoom and
  `seed20261084/20261120` for Reacher). The fresh 64-episode aggregates are
  `results/cross_environment_replication_20261015/{tworoom_aggregate_fresh64,reacher_aggregate_fresh64}`:
  TwoRoom `11/0/53` paired wins/losses/ties, delta `+0.171875`, CI
  `[+0.078125,+0.265625]`; Reacher `28/2/34`, delta `+0.40625`, CI
  `[+0.265625,+0.53125]`. The cross-environment index is
  `results/cross_environment_replication_20261015/nonpusht_evidence_fresh64.json`.
  These are scoped live RGB consequences; Cube remains a negative boundary and
  PointMaze remains inconclusive, so no universal claim is released.

- The strict external learned matched-quality benchmark was rerun on the
  immutable 90-record PushT external panel and 50-record PointMaze external
  panel. It identifies two environments and three matched strata, but the
  primary `Delta R` is only `+0.0030757` (bootstrap CI `[0,0.0092272]`,
  direct-benchmark sign-flip `p=0.5048`; wrapper `p=0.4935`), so the positive gate fails. Artifacts are
  `results/matched_learned_transfer_benchmark_external_crossenv_20261015/`,
  `results/cross_environment_transfer_diagnostic_external_20261015/`, and
  `results/best_paper_readiness_audit_external_20261015/`. This closes the
  missing-external-field audit path without converting a null learned selector
  result into a positive claim.

- `experiments/audit_reacher_latent_action_ranker.py` now includes nested
  calibration and an exact `alpha=0` raw-cost fallback. On two 32-root
  CUDA/EGL target-geometry panels, direct latent/action readout increases
  held-out regret (`0.3854 -> 0.5588` and `0.0905 -> 0.4219`); the nested
  safe blend selects `alpha=0` on both. The added effect-geometry feature
  family does not produce a positive held-out gain. These artifacts are
  `results/reacher_latent_action_ranker_v2_cuda_root32_target_{neg,pos}_20261015.json`
  and `results/reacher_latent_action_ranker_effect_geometry_cuda_root32_target_{neg,pos}_20261015.json`;
  status: action-identifiability repair failed, fail-closed fallback retained.

- `experiments/summarize_tworoom_method_comparison.py` compares live methods
  on an identical new 32-episode RGB seed panel. Raw frozen candidate-bank
  LeWM obtains `14/32` successes versus random `3/32`; the previously fitted
  latent/action ranker obtains `10/32` and action-conditioned physical readout
  `8/32`. Paired method-minus-raw deltas are `-0.125` and `-0.1875`,
  respectively. Artifact: `results/live_tworoom_ranker_validation_20261015/`.
  This closes the offline-to-closed-loop transfer check as a negative result;
  raw candidate-bank LeWM remains the scoped TwoRoom controller.

- `experiments/audit_reacher_sequence_candidate_ranking.py` and
  `experiments/summarize_reacher_candidate_support.py` implement a fixed
  sequence-bank intervention. On identical 32-root panels, adding pulse and
  recovery support reduces selected Reacher error, but this is an action-bank
  effect rather than a latent-dynamics repair. `compact` mode (49 candidates)
  is available to preserve CEM exploration.

- `experiments/run_live_reacher_sequence_candidate_planning.py` and
  `experiments/summarize_reacher_live_method_comparison.py` test the same bank
  in RGB-only closed-loop CEM. With a matched 256/8/16 budget, raw continuous
  CEM obtains `21/32` and `7/32` on the two target geometries; full bank gets
  `14/32` and `1/32`, compact bank `20/32` and `1/32`. Status: action-support
  mechanism localized, fixed-bank online repair rejected, raw continuous CEM
  retained as the scoped controller.

- `experiments/audit_cube_temporal_action_effect.py` and
  `experiments/summarize_cube_temporal_action_effect.py` implement the
  non-PushT temporal action-effect gate audit. Across two independent
  80-root CUDA/EGL guided panels, additive temporal history improves held-out
  next-block effect RMSE for the frozen latent relative to its static baseline
  by `-1.006e-4` (root bootstrap CI `[-1.906e-4,-0.168e-4]`), while RGB
  temporal history worsens its static baseline by `+1.769e-4`
  (`[+0.889e-4,+2.692e-4]`). The low-dimensional interaction lift is null or
  unstable, and the event/support gate worsens RGB with about 20.6% activation
  coverage. Registry status: representation-conditioned offline mechanism
  signal; not an online controller or causal-dynamics result. Aggregate:
  `results/cube_temporal_action_effect_20261015/aggregate_seed20261018_20261020.json`.

- `experiments/run_cube_temporal_closed_loop.py` now includes a predeclared
  fail-closed support-exploration cycle for low candidate-score spread, and
  `experiments/summarize_cube_temporal_closed_loop.py` performs paired,
  cross-panel bootstrap aggregation. On two disjoint CUDA/EGL RGB-only Cube
  panels (16 + 32 episodes), temporal minus static has `7/0/41` wins/losses/
  ties with mean final-distance delta `-0.01764 m` and CI
  `[-0.03380,-0.00489] m`; the support-gate arm has `10/0/38` and
  `-0.02010 m` with CI `[-0.03670,-0.00712] m`. The fallback triggers on only
  `4/48` episodes (`3` wins, `1` tie), so the gain is sparse and the temporal
  arm is panel-sensitive (`0/16` then `7/32`). Registry status: scoped positive
  action-support consequence, not universal visual planning or causal-latent
  evidence. Aggregate:
  `results/cube_temporal_closed_loop_20261015/aggregate_support_gate_panels_48ep.json`.

- `experiments/run_live_reacher_sequence_candidate_planning.py` now supports
  a four-candidate `support4` probe and `candidate_bank_iterations=1` for a
  one-shot probe. On fresh matched 32-seed target panels, repeated `support4`
  changes raw CEM by `-0.0625` and `-0.1875` on the two target geometries;
  one-shot support changes it by `-0.0625` and `+0.03125`, with the latter CI
  crossing zero. Registry status: fixed support injection rejected as a
  general controller repair; raw continuous CEM remains the accepted scoped
  Reacher controller. Comparison artifacts:
  `results/live_reacher_sequence_candidate_planning_20261015/method_comparison_support4_{neg,pos}.json`
  and
  `results/live_reacher_sequence_candidate_planning_20261015/method_comparison_support4_once_{neg,pos}.json`.

- The finite-horizon theory and control contract were executed by
  `experiments/theory_spectral_transfer/run_finite_horizon_transfer_risk_decomposition.py`
  and `run_covariant_transfer_risk_control_audit.py`. The artifacts verify a
  tight scalar bound, an explicit matched-diagnostic `G_h` counterexample,
  non-normal transient amplification, local nonlinear remainder conditions,
  and permitted versus fail-closed controller behavior. Registry status:
  formal conditional-control theory validated synthetically; learned-model
  calibrated certificate and universal commutator claim remain unvalidated.
  Artifacts:
  `results/finite_horizon_transfer_risk_decomposition_20261015/` and
  `results/covariant_transfer_risk_control_audit_20261015/`.

- `experiments/gdd_wm/train_projection_distillation.py` implements the D2
  projection-only geometry interface with a hard D1 gate and separate
  prediction/planning gates. Running it on the repeated-context PointMaze
  artifact correctly returns `blocked_d1_gate` and performs no training,
  because D1 remains `rejected_fail_closed` under the leaked offline-cost
  teacher. Registry status: D2 implementation complete, D2 evidence not
  authorized. Artifact:
  `results/gdd_wm_projection_geometry_distillation_20261015/d1_blocked/`.

- D2 provenance hardening now requires feature-table schema
  `gdd_wm_student_feature_table.v0.1`, exact candidate-dataset SHA-256,
  canonical record-key metadata, explicit no-context/no-teacher-rank/no-heldout
  outcome inputs, and frozen dynamics. Projection rank is also bounded by the
  train-record count. Legacy tables without these fields fail closed before
  fitting. Targeted tests: `4 passed`.

- `experiments/gdd_wm/audit_teacher_candidate_packages.py` audited all 24
  existing candidate packages and found `0/24` eligible. The blockers are
  `offline_environment_cost_v1`, exact negative-cost teacher leakage, missing
  repeated contexts in the original packages, and incomplete outcome/test
  runtime references. Artifact:
  `results/gdd_wm_teacher_candidate_audit_20261015/teacher_candidate_audit.json`.

- `experiments/summarize_cross_environment_planning.py` generated a native-
  estimand index across TwoRoom, Reacher, Cube, and PointMaze at
  `results/cross_environment_planning_evidence_20261015/`. TwoRoom and Reacher
  are positive live RGB paired consequences; Cube is a strict zero-gain failure
  boundary; PointMaze is statistically inconclusive. The rows are not pooled,
  and the artifact does not support universal world-model or commutator claims.

- `experiments/gdd_wm/build_train_only_readout_teacher_package.py` constructs a
  declared non-D-JEPA infrastructure control from two independent checkpoint
  contexts. Its package audit passes (`1/1 eligible`) with 2 contexts, 40 roots,
  640 matched root/action pairs, and complete runtime refs. D1 remains only
  partially useful: flip rate `0.3081`, overall execution consistency `0.4719`,
  and held-out outcome consistency `0.5800`. Registry status: D1 structure
  positive control, teacher validity rejected; D2 remains blocked.

- The same control now runs a predeclared five-family intervention through
  `experiments/gdd_wm/build_train_only_readout_teacher_package.py` and
  `experiments/gdd_wm/summarize_train_only_teacher_feature_panel.py`. The
  action-only confound has outcome consistency `0.6179` and mean regret
  `0.1060`, while predicted-terminal is `0.5625` and `0.2116`; the
  source/delta/action latent family does not improve consistency (paired delta
  `-0.0620`, CI crossing zero). This is a decision-signal diagnosis, not
  latent geometry or D-JEPA evidence. Artifact:
  `results/gdd_wm_nonleaking_teacher_control_20261015/feature_family_panel/summary.json`.

- The `--residualize-action` mode in
  `experiments/gdd_wm/build_train_only_readout_teacher_package.py` tests
  latent incremental value after a train-only action-summary readout. On the
  current held-out PointMaze roots, residualized predicted-terminal is below
  action-only by `-0.0247` consistency (CI crossing zero) and has `+0.0250`
  regret delta; residualized source/delta/action is `0.5006` consistency.
  Status: no independent latent transfer gain. Artifact:
  `results/gdd_wm_nonleaking_teacher_control_20261015/residualized_feature_panel/summary_vs_action.json`.

- `experiments/gdd_wm/audit_train_only_teacher_capacity.py` is the fixed
  CUDA readout-capacity control for the preceding residualized negative result.
  It compares one train-only standardized linear residual and one fixed
  `[64,32]` SiLU MLP on the same `source_delta_action` features and outcome
  roots. The linear residual has regret delta `-0.1568`
  (95% CI `[-0.2602,-0.0678]`) versus action-only; the MLP delta is `-0.0631`
  (CI crosses zero). This does not authorize D2 or planning claims: it rejects
  readout capacity as the sole explanation and leaves action support,
  representation validity, and cross-environment transfer unresolved.
  Artifact:
  `results/gdd_wm_nonleaking_teacher_control_20261015/capacity_audit/capacity_audit.json`.

- The capacity control was rerun on two canonical goal-frame seed contexts.
  Raw-action linear regret delta shrank to `-0.0305` (CI crosses zero); exact
  goal-relative action summaries gave linear delta `0.0000`, and the
  root-centered goal-relative protocol also gave `0.0000`. Fixed MLP deltas
  were `-0.0271`, `-0.0265`, and `-0.0360`, all with CIs crossing zero. The
  runner now exposes `--action-frame {raw,goal_relative}` and
  `--target-mode {absolute,root_centered}`. This identifies the first positive
  raw-action point estimate as a coordinate/root-cost confound and keeps the
  status diagnostic-only. Artifacts:
  `results/gdd_wm_nonleaking_teacher_control_20261015/canonical_goalframe_capacity_audit/`,
  `.../canonical_goalframe_capacity_audit_goal_relative/`, and
  `.../canonical_goalframe_capacity_audit_root_centered/`.

- The capacity audit was repaired to keep every context-root ranking cell
  separate; the previous mixed-context result is not final evidence. Corrected
  CUDA replications are in
  `results/gdd_wm_cross_environment_20261016/tworoom_capacity_root_centered_context_safe/`
  and
  `results/gdd_wm_cross_environment_20261016/reacher_capacity_root_centered_context_safe/`.
  Under the root-centered estimand, fixed MLP residual regret deltas are
  `-16.8480` for TwoRoom (95% CI `[-32.0209,-0.4949]`) and `-1.9058` for
  Reacher (95% CI `[-2.3356,-1.4063]`); linear deltas are `-7.1922`
  (TwoRoom, CI crossing zero) and `-1.6113` (Reacher, CI
  `[-2.0892,-1.1483]`). This is a context-safe cross-environment capacity
  diagnostic, not an independent architecture/seed matched benchmark or a
  planning/teacher/D2 claim. Action frame remains raw and must be replaced by
  the canonical goal-relative protocol for confirmatory learned-model claims.

- The TwoRoom/Reacher panel exporters now emit evaluator-only
  `source_state_raw` and `goal_raw` fields under panel schema `v0.2`, enabling
  the canonical action-frame audit without changing RGB runtime inputs. New
  CUDA/EGL panels are in
  `results/gdd_wm_cross_environment_20261016/{tworoom_history1_goalframe,tworoom_history3_goalframe,reacher_history1_goalframe,reacher_history3_goalframe}/`.
  With `--action-frame goal_relative --target-mode root_centered`, regret deltas
  are TwoRoom `-3.0036` (linear, CI crosses zero) and `-3.3582` (MLP, CI
  crosses zero), while Reacher is `+0.1087` (linear, CI crosses zero) and
  `+0.2677` (MLP, CI crosses zero). Thus the earlier raw-action capacity gains
  are not canonical latent-transfer evidence; the protocol now fails closed
  until independent architecture/seed strata are available.

- A frozen RGB goal-latent conditioning arm was added to the capacity audit
  (`--goal-conditioning latent_relative`) and evaluated on new CUDA/EGL panels
  under `results/gdd_wm_cross_environment_20261016/*_goal_latent/`. Canonical
  root-centered regret deltas are TwoRoom `-4.3478` (linear, CI crosses zero)
  and `-1.2867` (MLP, CI crosses zero), while Reacher is `+0.2943` (linear,
  CI `[+0.0060,+0.6155]`) and `-0.2280` (MLP, CI crosses zero). This is a
  runtime-safe diagnostic intervention, not a stable latent-transfer gain;
  it rules out missing goal embedding as the sole explanation and keeps the
  action-effect/support failure boundary open.

- A consequential direction bug in `gdd_wm/audit_train_only_teacher_capacity.py`
  was repaired on 2026-10-16: the target is terminal cost, so root-level
  selection must use `argmin`, not `argmax`. A regression test now guards this
  invariant; pre-repair capacity artifacts remain provenance-only. Re-running
  the canonical CUDA panels after the repair gives TwoRoom linear/MLP regret
  deltas `+15.5624`/`+6.6561`, versus Reacher `-0.8184`/`-1.0049`, exposing an
  environment-conditional action-effect interface rather than a uniform
  capacity explanation.

- The same audit now performs a train-only ridge grid selection on calibration
  roots and exposes a fail-closed `calibration_safe_gate`: a latent residual is
  enabled only when its paired calibration regret CI95 upper endpoint is below
  zero; otherwise the action-only baseline is retained. On the frozen
  goal-relative/root-centered panels, TwoRoom correctly falls back (calibration
  delta `+0.50`, CI `[-6.57,+8.07]`) and its held-out delta is exactly
  `0.0000 [0.0000,0.0000]`; Reacher enables the residual (calibration delta
  `-0.84`, CI `[-1.40,-0.33]`) and retains held-out regret reduction
  `-0.4114`, CI `[-0.7901,-0.0288]`. This is a non-degradation/fail-closed
  diagnostic result, not universal transfer or online-planning evidence.
  Artifacts: `results/gdd_wm_cross_environment_20261016/*_capacity_safe_gate/`.

- The indexed canonical TwoRoom readout now has the same calibration-safe
  contract. Train-only readout fitting is followed by a root-macro paired
  bootstrap gate against an action-only train ridge; the indexed intervention
  is enabled only when the calibration CI95 upper endpoint is below zero.
  Both raw and frozen-goal-latent artifacts fail the gate (deltas
  `-18.7344`, CI `[-45.6880,+9.7637]`, and `-14.9146`, CI
  `[-38.7106,+10.9242]`). On the fresh CUDA panel
  `20261640–20261671`, both gated artifacts exactly match the frozen
  candidate-bank baseline at `14/32`; the original ungated indexed arm was
  `8/32`. This repairs negative transfer by fail-closed fallback, but does not
  establish an indexed-readout gain. Implementation:
  `experiments/gdd_wm/build_tworoom_indexed_canonical_readout.py`,
  `experiments/capacity_readout_runtime.py`, and
  `experiments/run_live_tworoom_planning.py`; artifacts:
  `results/gdd_wm_cross_environment_20261016/tworoom_history1/indexed_canonical_readout_safe_gated/`
  and `.../tworoom_history1_goal_latent/indexed_canonical_readout_safe_gated/`.

- PointMaze RGB online planning negative-result repair (2026-10-16): the
  frozen `wm_rgb` protocol was re-run on 80 root-disjoint held-out roots and
  achieved `5/80` success with mean final distance `0.9850`, while
  visual-greedy and visual-dynamics controls achieved `80/80`. A predeclared
  calibration-only RGB-velocity/block-dynamics gate was wired into
  `experiments/run_live_pointmaze_planning.py`. It falls back when the WM
  action's dynamics-score regret exceeds `0.10`, without using held-out state
  for policy input or selection. A 10-root mechanism panel changed from
  `0/10` to `10/10`; the independent 80-root panel achieved `80/80` with mean
  final distance `0.1483`. Against the original WM, paired mean final-distance
  delta was `-0.8367` (bootstrap 95% CI `[-0.8854,-0.7802]`) and the success
  delta CI was `[0.875,0.988]`. The gate fell back on `1290/1553` decisions
  (`83.1%`), identifying unsafe WM action ranking as the proximate failure
  mechanism. This is a scoped positive result for support-aware fail-closed
  control, not evidence that the unguarded frozen WM, universal world model, or
  commutator certificate is validated. Artifacts:
  `results/pointmaze_rgb_online_planning_v6_cuda_guarded_smoke10/report.json`
  and `results/pointmaze_rgb_online_planning_v7_cuda_guarded_panel80/report.json`;
  regression test: `tests/test_live_pointmaze_planning.py` (`5 passed`).

- Reacher calibration-safe readout correction (2026-10-16): the initial
  standard-CEM evaluation was identified as an inactive intervention because
  gated readouts require `_candidate_bank_mask`; standard CEM supplies none and
  correctly returned frozen base costs. The corrected sequence-bank protocol
  uses `capacity17` and supplies the mask. On 8 paired seeds, Euclidean bank
  CEM achieved `2/8`, replacing supported-bank costs with the offline readout
  achieved `0/8`, and an additive conservative configuration
  (`base_rank_quantile=0.25`, `correction_multiplier=0.5`) recovered `2/8`.
  On a fresh 32-seed panel, Euclidean and additive conservative readout both
  achieved `10/32` with identical episode-success vectors (`0/0/32` paired
  wins/losses/ties). The code now rejects gated readouts in standard CEM and
  requires the sequence runner. This repairs negative transfer by
  non-degradation/fail-closed behavior, but does not establish an online
  latent-transfer gain. Artifacts:
  `results/gdd_wm_cross_environment_20261016/live_reacher_capacity17_smoke8/`
  and `.../live_reacher_capacity17_panel32/`.

- Cube post-contact temporal-memory diagnosis and repair (2026-10-16): the
  branch runner now fails closed on RGB approach non-coverage instead of
  silently fitting `None`, and exposes an explicitly non-RGB
  `privileged_world` branch-isolation mode. The original temporal negative was
  traced to previous-action support mismatch: calibration used the close
  action, while later deployment steps used push actions. Sequence-aware
  calibration expands every calibration root with the fixed previous-action
  bank. On the development 40/32 CUDA panel, raw temporal-minus-static was
  `-0.01219 m` (bootstrap CI `[-0.03156,+0.00108]`, `5/4/23` wins/losses/ties),
  but on an independent 40/32 panel it was `+0.00275 m` (`3/8/21`), so the
  repair does not establish a universal gain. Fixed compact pooling and an
  event/support-gated residual are retained as diagnostic arms; the gated
  smoke was non-degrading but no-gain. Artifacts:
  `results/gdd_wm_cross_environment_20261016/cube_hybrid_contact_latent_sequence_calibrated_panel40cal32test_cuda/`,
  `.../cube_hybrid_contact_latent_sequence_calibrated_independent_seed20262020_cuda/`,
  and `.../cube_hybrid_contact_latent_gated_residual_smoke_cuda/`.

- WA-JEPA-inspired PointMaze future/action intervention (2026-10-16): first
  repaired an evaluation inconsistency where untrained cost heads contaminated
  per-seed ranking for non-cost objectives. The corrected runner adds
  asymmetric future/action objectives and a direct pairwise preference head.
  On the history-4 panel, asymmetric joint improved task-effect RMSE to
  `0.28842` and paired regret gain to `+0.02761` across five seeds, but the
  root-level improvement was concentrated in one outcome root. On the earlier
  single-frame negative panel, asymmetric joint reduced the negative gain from
  `-0.11883` to `-0.00471` but did not make it positive; direct preference was
  `-0.07336`. This identifies symmetric multi-task interference as one cause,
  while missing root-specific action-effect observability/support remains.
  The previous standalone `+0.07477` result is not promoted because its random
  baseline varied by seed and was not the same paired control. Artifacts:
  `results/decision_benchmark_candidate_bank_20261016/joint_rgb_future_task_history4_v3_cuda.json`,
  `.../joint_rgb_future_task_singleframe_v3_cuda.json`, and
  `.../joint_rgb_future_task_singleframe_v4_preference_cuda.json`;
  implementation: `experiments/decision_benchmark/train_joint_rgb_future_task_adapter.py`;
  regression: `tests/test_train_joint_rgb_future_task_adapter.py` (`13 passed`).
  The follow-up artifact
  `results/decision_benchmark_candidate_bank_20261016/joint_rgb_future_task_history4_final_v7_cuda.json`
  adds temporal-derivative and root-relative pair-effect interventions. The
  derivative term raises future change-magnitude ratio from `0.624` to `0.736`
  but does not improve ranking; relative/pair-effect variants remain below
  `asym_joint`. This is a representation/action-direction boundary, not a
  released planning gain.
  The offline realized-future oracle
  `results/decision_benchmark_candidate_bank_20261016/joint_rgb_future_task_history4_realized_future_oracle_v8_cuda.json`
  has all five `asym_joint` seed gains negative despite lower successor RMSE;
  future reconstruction error alone is therefore not the remaining cause.
  The direct sequence-direction readout in
  `.../joint_rgb_future_task_history4_direct_sequence_v9_cuda.json` also has
  mixed seed signs and is not promoted.
  The offline realized-future oracle
  `results/decision_benchmark_candidate_bank_20261016/joint_rgb_future_task_history4_realized_future_oracle_v8_cuda.json`
  has all five `asym_joint` seed gains negative despite lower successor RMSE;
  future reconstruction error alone is therefore not the remaining cause.

- Root-bootstrap and uncertainty-gate audit (2026-10-16): the active
  reference-probe artifact was re-audited with root as the bootstrap unit,
  averaging the five seeds within each root before resampling. For `v17`,
  `asym_joint` had root mean `+0.01495`, but the root-bootstrap 95% CI was
  `[-0.00920,+0.05593]` and leave-one-root-out means reached `-0.00366`;
  the apparent gain is therefore not stable. The reusable audit is
  `experiments/decision_benchmark/audit_adapter_bootstrap.py`, with artifact
  `results/decision_benchmark_candidate_bank_20261016/adapter_root_bootstrap_audit_v17_vs_v15.json`.
  A calibration-only support gate on the passive RGB panel changed
  `asym_joint` from `-0.00153` to `+0.00153`, but its root-bootstrap CI was
  `[-0.01597,+0.01797]`; adding uncertainty penalty was rejected by
  calibration. A rank-instability gate changed it to `-0.00005` with CI
  `[-0.01773,+0.01684]`. The latent-history control was also negative
  (`-0.00092`, CI `[-0.01862,+0.01603]`). These are diagnostic/negative
  artifacts, not planning gains. Implementation:
  `experiments/decision_benchmark/run_uncertainty_gated_selection.py`;
  regression: `tests/test_uncertainty_gated_selection.py`.

- Goal-relative task-effect chart intervention (2026-10-16): the previous
  hybrid action frame changed only the input chart while the effect target
  remained raw XY. `experiments/decision_benchmark/run_task_frame_intervention.py`
  now compares raw-XY supervision against goal-relative effect supervision,
  mapping predictions back to physical XY before ranking. On the original
  16-root near-goal outcome, all five goal-relative `asym_joint` seeds were
  positive (`+0.01873,+0.01040,+0.01837,+0.01556,+0.01727`) and the ensemble
  gain was `+0.01642` versus `-0.00153` for raw-XY; the root-bootstrap CI
  still crossed zero (`[-0.00952,+0.05746]`), so this is a mechanism pilot.
  A frozen 20-root outcome made from previously unused roots then failed
  reproducibly: goal-relative `asym_joint` was `-0.01034` with CI
  `[-0.01622,-0.00503]`, while raw-XY was `-0.01000` with CI
  `[-0.01649,-0.00440]`. The 20 new roots had source-goal distance
  `2.19–2.79` and zero success candidates, versus `3/16` success-supported
  roots in the near-goal panel. The positive result is therefore support/
  geometry-conditioned, not universal. Artifacts:
  `results/decision_benchmark_candidate_bank_20261016/task_frame_history4_neargoal_hybrid_v1_cuda.json`
  and
  `results/decision_benchmark_candidate_bank_20261016/task_frame_external_outcome20_hybrid_v1_cuda.json`.
  The external panel passed `84/84` RGB renderer/source gates and complete
  PCA successor extraction in
  `.../rgb_successor_history4_canonical_external_outcome20_v2_cuda/`.
  Regression: `tests/test_task_frame_intervention.py`.

- Goal-directed action-support intervention (2026-10-16): a separate
  `goal_directed_mixture_v1` bank was generated on CUDA and then assigned the
  exact frozen 48/16/20 root split used by the canonical external panel. The
  new RGB bridge passed renderer/source consistency and successor extraction
  on all 84 retained roots. On the 20 held-out outcome roots, mean best
  terminal distance improved from `2.1490` to `1.9513`, but successful support
  remained `0/20`, and the task-frame adapter's selected success remained
  `0/20`. A calibration-only uncertainty/context-support gate selected no
  fallback roots, showing that context support is not a reachability signal.
  The remaining cause is a fixed 20-step reachable-horizon boundary: a far
  root at source-goal distance about `2.79` moved only about `0.5` under a
  saturated pulse. This is action-support/reachable-horizon evidence, not a
  latent repair or planning gain. Artifacts:
  `results/decision_benchmark_candidate_bank_20261016/jepa_history4_goal_directed_external_outcome20_cuda/`,
  `.../rgb_successor_history4_goal_directed_external_outcome20_cuda/`,
  `.../task_frame_goal_directed_external_outcome20_hybrid_v1_cuda.json`, and
  `.../uncertainty_gated_history4_goal_directed_external_outcome20_cuda.json`.

- PointMaze explicit-split receding-horizon repair (2026-10-16):
  `experiments/run_live_pointmaze_planning.py` now accepts explicit
  calibration/test episode lists and rejects overlap, enabling exact reuse of
  the frozen 16/20 root split. With the original 12-block budget, guarded
  RGB planning improved final distance over unguarded `wm_rgb` but remained
  `2/20` successful and was worse than the calibration-only visual-dynamics
  fallback. With the same RGB-only protocol and a predeclared 24-block
  receding horizon, guarded planning reached `13/20` on the external outcome
  panel and `13/16` on the previously excluded panel, versus `2/20` and
  `2/16` for unguarded WM. Across both panels (`n=36`), guarded versus WM had
  paired distance delta `-0.7356`, bootstrap CI `[-0.9027,-0.5782]`, and
  success `26/36` versus `4/36`. Guarded versus calibration-only
  visual-dynamics was non-inferior (`26/36` versus `24/36`, delta `-0.0219`,
  CI `[-0.0780,+0.0275]`). The gate fell back on `80.2%` of decisions. This
  is a scoped support-aware RGB closed-loop repair, not evidence that pure
  frozen WM dynamics or universal transfer is solved. Artifacts:
  `results/pointmaze_rgb_online_planning_v8_cuda_external20_split/`,
  `.../pointmaze_rgb_online_planning_v9_cuda_external20_longer24/`, and
  `.../pointmaze_rgb_online_planning_v10_cuda_excluded16_longer24/`.

- PointMaze WM action projection intervention (2026-10-16):
  `experiments/run_live_pointmaze_planning.py` adds `wm_rgb_projected`, which
  interpolates the frozen WM-selected action toward the calibration-only
  visual-dynamics action until the declared support tolerance is met. On the
  exact external-20 panel it reached `5/20` success with mean final distance
  `0.5971`, versus pure WM `2/20`, `1.0773`, and guarded `13/20`, `0.3243`.
  On the independent excluded-16 panel it reached `10/16`, `0.4421`, versus
  pure WM `2/16`, `0.9330`, and guarded `13/16`, `0.2191`. Combined projection
  versus pure WM improved distance by `-0.4850` (bootstrap CI
  `[-0.6151,-0.3583]`) but remained worse than guarded by `+0.2507`
  (`[+0.1508,+0.3559]`). Projection was active on `80.3%` of decisions and
  is therefore a partial support repair, not a replacement for action
  identifiability or a pure frozen-WM planning gain. Artifacts:
  `results/pointmaze_rgb_online_planning_v14_cuda_external20_projected/` and
  `.../pointmaze_rgb_online_planning_v15_cuda_excluded16_projected/`;
  regression: `tests/test_live_pointmaze_planning.py` (`12 passed`).

- PointMaze alpha-gated projected-guarded follow-up (2026-10-16):
  `wm_rgb_projected_guarded` was evaluated on the independent excluded-16
  panel with the same 24-block receding-horizon protocol and a predeclared
  `projection-min-alpha=0.25`. The external-20 panel reached `5/20` with
  mean final distance `0.5507` and gate fallback rate `13.0%`; the excluded-16
  panel reached `9/16` with mean final distance `0.4310` and fallback rate
  `17.1%`. On the matched 36-root comparison, alpha-gated projection was
  `14/36` successful with mean final distance `0.4975`, versus `26/36` and
  `0.2775` for the predeclared guarded controller. The paired alpha-gated
  minus guarded distance delta was `+0.2200`, bootstrap CI
  `[+0.1400,+0.3074]`; success lost on `12` roots and won on none. Against
  pure `wm_rgb`, distance improved by `-0.5156`, CI
  `[-0.6590,-0.3767]`, but this does not establish a pure-WM gain because
  the guarded control uses fail-closed fallback. The intervention therefore
  reduces fallback while losing the held-out safety/performance guarantee.
  Artifact:
  `results/pointmaze_rgb_online_planning_v17_cuda_excluded16_projected_guarded_alpha025/report.json`;
  paired external artifact:
  `results/pointmaze_rgb_online_planning_v16_cuda_external20_projected_guarded_alpha025/report.json`.

- HTSC finite-sample calibration correction (2026-10-16):
  `experiments/decision_benchmark/task_contrast.py` now uses the valid
  split-conformal rank `ceil((n+1)(1-alpha))`. If that rank exceeds the
  number of independent calibration roots, calibration returns
  `status=insufficient_calibration` and `bound=null`; all certificate and
  selective-release callers fail closed rather than silently using the
  sample maximum. The task-contrast regression suite now has `5 passed` and
  explicitly covers both insufficient and finite-rank calibration. This
  repairs the implementation/theory mismatch, but does not establish online
  coverage for adaptive CEM candidate selection.

- PointMaze privileged fixed-bank candidate ceiling (2026-10-16):
  `run_live_pointmaze_planning.py` now exposes
  `ground_truth_candidate_oracle`, which evaluates each fixed action-bank
  candidate with one exact simulator block and restores the runtime state.
  It is a privileged counterfactual ceiling, not an RGB policy. On the
  root-disjoint external-20 panel it achieved `11/20` with mean final
  distance `0.3097`; on excluded-16 it achieved `13/16` with distance
  `0.2416`. Combined, the ceiling was `24/36` and `0.2794`. The ceiling is
  still below complete success on external roots, so fixed-bank reachability
  and local greedy planning are independent bottlenecks. Artifacts:
  `results/pointmaze_rgb_online_planning_v18_cuda_external20_oracle/report.json`
  and
  `results/pointmaze_rgb_online_planning_v19_cuda_excluded16_oracle/report.json`.

- PointMaze exact-planner protocol audit and root-bank fix (2026-10-16):
  the first `perfect_dynamics_mpc` implementation regenerated the root action
  bank from privileged simulator coordinates, whereas the established
  fixed-bank oracle generated it from the observed RGB-calibrated position.
  This small angular discrepancy changes wall-contact outcomes. The runner now
  keeps the observed root bank identical to the fixed-bank baselines and only
  uses exact simulator positions for successor-bank expansion. A second audit
  found that the earlier MPC runs used `max_blocks=12`, while the established
  oracle used a 24-block budget; their final success rates were therefore not
  comparable. With both fixes and a matched 24-block budget, depth-1/beam-32
  exactly reproduced the established external-20 oracle: `11/20` success,
  mean final distance `0.3097`, and `21.05` mean executed blocks. Regression
  coverage is now `21 passed` for the live PointMaze and task-contrast tests.
  Artifacts:
  `results/pointmaze_rgb_online_planning_v24_cuda_external20_perfect_mpc_d1_b32_rootfixed_max24/report.json`;
  implementation: `experiments/run_live_pointmaze_planning.py`;
  regression: `tests/test_live_pointmaze_planning.py`.

- PointMaze multi-step MPC smoke status (2026-10-16): the preliminary depth-4,
  beam-8 runs used the shorter 12-block budget and remain diagnostic smoke
  artifacts only: external-20 was `2/20`, mean final distance `1.4335`, and
  excluded-16 was `5/16`, mean final distance `0.9847`. They are superseded by
  the matched 24-block comparison recorded below and are not used for a
  finite-horizon claim.

- PointMaze planner-budget repair (2026-10-17): the matched 24-block panel
  established a shared fixed-bank ceiling of `23/32` for both the RGB
  visual-dynamics policy and the privileged one-step candidate oracle. An
  expanded-bank exact planner remained `23/32` at 24 blocks, but reached
  `32/32` with a predeclared 40-block execution budget (mean final distance
  `0.1141`). With the same 40-block budget, RGB visual-dynamics,
  effect-adapter, and effect-adapter+HTSC each reached `32/32` on roots
  `1900--1931`. On unopened roots `1932--1963`, all three again reached
  `32/32`; visual-dynamics had the lowest mean final distance (`0.1411`
  versus `0.1536` and `0.1613`). This repairs the earlier budget-induced
  negative result but does not show an HTSC gain over the strong visual
  baseline. The default PointMaze budget is now 40 blocks. Full boundary and
  artifacts: `docs/POINTMAZE_BUDGET_REPAIR_20261017.md`.

- World-fixed action-bank repair and bilinear decision adapter (2026-10-16):
  canonical Bank-only/No-dynamics/Shuffled controls showed that the old
  `+0.1090` indexed-action gain was a goal-relative bank shortcut. A shared
  world-frame bank removed that shortcut and exposed a missing
  `source x action` term. On a fresh deterministic hash split over 100 roots
  and two checkpoint contexts, the train-OOF/calibration gate selected the
  No-dynamics bilinear adapter. Independent outcome regret fell from `0.1103`
  to `0.0476`, paired CI `[-0.1196,-0.0130]`, and consistency rose from
  `0.5594` to `0.6735`, delta CI `[+0.0315,+0.2080]`. Full predicted dynamics
  did not beat No-dynamics and Shuffled-prediction was at least as good, so the
  positive result is decision repair from frozen RGB source latent, not
  learned dynamics or planning. A shared readout over four contexts from two
  predictor architectures reproduced the same gate decision; however, all
  contexts reuse the same already-inspected PointMaze roots, so this remains
  exploratory architecture-repeated evidence rather than an untouched final
  evaluation. Artifacts:
  `results/decision_benchmark_candidate_bank_20261016/worldfixed100_hashsplit_decision_repair_v9_s20261016_cuda/capacity_audit.json`.
  `results/decision_benchmark_candidate_bank_20261016/worldfixed100_hashsplit_2arch4context_decision_repair_s20261016_cuda/capacity_audit.json`.

- Matched privileged depth ceiling (2026-10-16): corrected depth-4/beam-8,
  24-block exact MPC reached `12/20`, distance `0.2967`, versus corrected
  depth-1/beam-32 `11/20`, `0.3097`. Paired success delta `+0.05` had CI
  `[-0.20,+0.30]`; distance improvement `+0.0130` had CI
  `[-0.0886,+0.1094]`. Both reports now correctly mark
  `perfect_dynamics_mpc` as privileged with runtime simulator-state access.
  Artifacts:
  `results/pointmaze_rgb_online_planning_v25_cuda_external20_perfect_mpc_d4_b8_rootfixed_max24/`
  and
  `results/pointmaze_rgb_online_planning_v26_cuda_external20_perfect_mpc_d1_b32_rootfixed_max24_privfixed/`.

- Phase/horizon-conditioned transfer-sufficiency integration (2026-10-16):
  `docs/PHASE_HORIZON_CONDITIONED_TRANSFER_SUFFICIENCY_20261016.md` turns the
  H-JEPA insight into a proposed LeWorldModel protocol. Different horizons and
  planning phases receive separate transfer, action-identifiability, support
  and subgoal-reachability gates; slow prediction alone is explicitly rejected
  as a certificate. This is a proposed method, not a validated H-JEPA result.
- Reacher deployment-aligned 128-root capacity replication (2026-10-16): the
  unchanged CUDA protocol used 64 train, 32 calibration and 32 untouched
  outcome roots. Outcome regret was `0.5319` for action-only, `0.2226` for the
  source-action MLP (paired delta `-0.3093`, 95% CI
  `[-0.4311,-0.1930]`), `0.3116` for full predicted dynamics and `0.1919` for
  shuffled prediction. The larger panel therefore replicates a phase-specific
  source-action decision signal while rejecting independent value from the
  current predicted dynamics. Artifact:
  `results/gdd_wm_cross_environment_20261016/reacher_deployment_aligned_panel128_capacity_frozen/capacity_audit.json`.
- Reacher proposal-relative decision repair and fresh replication
  (2026-10-16): the old `4.30%/4.48%` proposal-support numbers used inactive
  rows in their denominators and are invalid as conditional support estimates;
  the empirical `98.44%` gate was also too strict for finite candidate clouds.
  The repaired chart standardizes each action summary inside its own CEM cloud,
  and the gate freezes the fifth percentile of one-sided 90% Wilson lower
  bounds from calibration (`0.927105`). Offline action-only/source-action/full/
  shuffled regrets are `0.47152/0.21495/0.25498/0.19389`: source-action is
  useful, but predicted dynamics still fail no-dynamics/shuffled controls.
  Two untouched 64-root online batches changed success `32->38` and `36->39`;
  batch-stratified aggregation is `68/128 -> 77/128`, paired `+0.0703125`, CI
  `[+0.015625,+0.1328125]`, with `12/3/113` discordance/ties. Conditional
  active-row support is `0.93933` (`0.93359`--`0.94922`), with no abstentions.
  Final/minimum qpos-distance intervals cross zero, so the claim is a
  Reacher-scoped binary-success gain from source-action decision repair, not
  learned-dynamics or universal-world-model evidence. Aggregate artifact:
  `results/gdd_wm_cross_environment_20261016/live_reacher_top1_proposal_relative_wilson_two_batch_128_stratified.json`;
  code: `experiments/compare_reacher_capacity_planning.py` and
  `experiments/aggregate_reacher_capacity_batches.py`.
- Reacher predicted-cost incremental controls (2026-10-16): fixed `0.5`
  source-action/model-cost fusion on fresh roots produced no-dynamics/true/
  shuffled success `39/64`, `34/64`, `38/64`. True fusion improved final qpos
  distance by `-0.03426`, CI `[-0.06971,-0.00152]`, but reduced binary success
  and did not beat shuffled, so this is smoothing rather than candidate-paired
  dynamics evidence. A calibration-only source-action top-k tie-break audit
  tested `k={2,4,8,16,32,64}`; all true-vs-shuffled CIs crossed zero and the
  legal selector failed closed to `k=1`. The exploratory fresh `k=2` panel was
  `38/64`, `39/64`, `39/64`, with true-minus-shuffled success exactly zero.
  Current runtime policy remains no-dynamics. Artifacts:
  `results/gdd_wm_cross_environment_20261016/live_reacher_cost_fusion_seed20263600_64_paired.json`,
  `results/gdd_wm_cross_environment_20261016/reacher_model_cost_elite_selection/calibration_and_outcome.json`,
  and `results/gdd_wm_cross_environment_20261016/live_reacher_cost_elite_seed20263700_64_paired.json`.
- HTSC checkpoint-provenance repairability panel (2026-10-16):
  `generate_candidate_bank.py --split-assignment checkpoint_protocol_v1`
  prevents bank-level repartition from reusing predictor-fit episodes. The
  resulting CUDA panel has 54 train, 14 calibration and 32 outcome roots with
  16 candidates/root and realized RGB successors. Finite-sample fixed-bank
  coverage is `31/32` for the root-conditioned bank and `32/32` for the
  shared-world bank; exact-winner release remains `0` in both cases.
  Root-conditioned candidates fail the mandatory Bank-only control because
  candidate identity encodes goal direction. On the shared-world intervention,
  the train-only effect adapter reduces outcome regret from Bank-only `0.0851`
  to `0.0328`, paired gain `+0.0523`, 95% CI
  `[+0.0139,+0.0923]`; it also beats No-dynamics, shuffled prediction,
  predeclared unseen candidate IDs and matched temporal-order Bank-only with
  intervals above zero. Generic RGB-successor residual adaptation does not
  retain that benefit. Operational status remains abstain because exact-winner
  and preregistered bounded-regret permissions release zero roots. Compact
  artifact: `results/htsc_checkpoint_provenance_20261016/summary.json`; code:
  `experiments/decision_benchmark/summarize_htsc_repairability.py`.

- PointMaze single-pass preprocessing repair (2026-10-16):
  `experiments/theory_spectral_transfer/run_jepa_wm_pointmaze_temporal_eval.py`
  previously transformed dataset observations and then sent them through
  `model.encode`, which repeated normalization/augmentation and made offline
  latents incompatible with online RGB latents. The exporter now consumes raw
  0--255 observations, records `model.encode` as the sole preprocessing owner,
  and is protected by `tests/test_jepa_pointmaze_export.py`. A corrected
  100-root CUDA export is runtime-equivalent, but the exploratory rank grid has
  no legal winner: rank 8 passes all incremental counterfactual controls while
  its full-bank bound is `0.30093` and releases `0/32` at `epsilon=0.25`; rank
  16 lowers the bound to `0.08918` and releases `28/32`, but its Bank-only CI
  crosses zero and its unseen-candidate gain is negative. The old-PCA chart's
  `24/32` release is non-deployment evidence because the PCA came from the
  invalid double-transform representation. Status: preprocessing bug closed,
  deployment-aligned `Repair but Abstain`; no learned-dynamics or online-planning
  claim. Detailed adjudication is in
  `docs/FINAL_STAGE_RESIDUAL_NEGATIVE_DIAGNOSIS_20261016.md`.
