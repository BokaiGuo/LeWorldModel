# 实验入口注册表

本表是迁移期的入口索引。分类描述维护责任，不代表科学主张成立。

## Active

| 入口/目录 | 责任 | 状态 |
| --- | --- | --- |
| `le-wm-main/experiments/run_unified_research_protocol.py` | geometry、memory、contact-control 和显式 claim workflow | 正在接入统一 runtime |
| `le-wm-main/experiments/run_full_spectral_transfer.sh` | full-data spectral transfer 兼容入口 | 保留 CLI，逐步变薄 |
| `le-wm-main/experiments/empirical_spectral_audit/` | learned latent 与 transfer diagnostics | active |
| `le-wm-main/experiments/decision_benchmark/` | action-effect、ATM/IAEI、task contrast、PDTC、candidate bank、RGB successor、RGB-to-task selective audit | active / gated |
| `le-wm-main/experiments/decision_benchmark/audit_rgb_task_contrast.py` | candidate-collapse diagnostics、train-only joint action support、fail-closed release | active / gated |
| `le-wm-main/experiments/decision_benchmark/train_rgb_candidate_successor_adapter.py` | train-root-only counterfactual RGB successor adapter control；--history-windows supports causal RGB context | intervention-only / not planning evidence |
| `le-wm-main/experiments/decision_benchmark/verify_rgb_successor.py` | dataset-matched realized RGB successor verification；legacy two-frame and history-conditioned temporal contracts | active / gated |
| `le-wm-main/experiments/run_pusht_mode_aware_shared_candidate.py` | privileged PushT mode/action-effect controller；local observed-effect neighborhood and paired moment-sign probes | active / mechanism-only / pose gate failed |
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
