# 实验入口注册表

本表是迁移期的入口索引。分类描述维护责任，不代表科学主张成立。

## Active

| 入口/目录 | 责任 | 状态 |
| --- | --- | --- |
| `docs/FINITE_HORIZON_TRANSFER_RISK_THEOREM_20261018.md` | fixed-support finite-horizon transfer-risk bound, estimability gates, and explicit counterexamples | theory foundation / local theorem; not a universal certificate |
| `le-wm-main/experiments/audit_cube_planner_ceiling.py` | privileged fixed/expanded candidate-bank reachability ceiling from the cached RGB-prelude decision state; cycle-aware oracle stop; factorized approach/push support family | implemented / CPU exploratory artifact; GPU confirmation pending |
| `le-wm-main/experiments/audit_cube_planner_ceiling.py --resume` | per-root atomic checkpoint and stage-level progress for the long paired ceiling; prevents interrupted host runs from discarding completed roots | implemented / resume-safe; scientific artifact still requires host CUDA run |
| `le-wm-main/experiments/summarize_cube_failure_cause.py` | joins ceiling and paired RGB artifacts into predeclared support/contact/transfer failure classes; expanded support now prevents fixed-bank masking | implemented / attribution-only / classifier regression covered |
| `le-wm-main/experiments/run_cube_diagnostic_bundle.py` | locks ceiling, RGB panel, and failure attribution to one seed/root/task protocol | implemented / CUDA fail-closed; CPU exploratory only |
| `le-wm-main/scripts/run_cube_gpu_repair_panel.sh` | host entry point: CUDA preflight, factorized/goal-conditioned ceiling, RGB paired panel, and failure attribution | implemented / host execution required / no CPU fallback |
| `le-wm-main/experiments/gpu_preflight.py --render-smoke` | verifies real CUDA allocation plus the exact EGL/MuJoCo RGB render path before long visual panels | implemented / host-only verification |
| `le-wm-main/experiments/run_live_cube_rgb_macro_panel.py --resume` | arm-level checkpoint reuse for the paired RGB panel | implemented / protocol-compatible resume only |
| `le-wm-main/experiments/run_live_cube_rgb_macro_ranker.py --factorized-bank` | deployable finite bank with independent approach/push directions and phase-local action summaries | new predeclared support intervention; GPU held-out consequence pending |
| `le-wm-main/experiments/run_unified_research_protocol.py` | geometry、memory、contact-control 和显式 claim workflow | 正在接入统一 runtime |
| `docs/PUSHT_LITERATURE_TO_PROTOCOL_20261019.md` | 将局部视觉动力学、信息性探针、观测后验证、mode chart 与 fail-closed 机制映射到 PushT 实验约束 | literature-to-protocol evidence map / no independent result claim |
| `le-wm-main/experiments/audit_pusht_verified_orientation_linesearch.py` | XY 已闭合后的 active signed pulse、RGB/状态重观测和局部方向更新上限诊断 | new privileged mechanism audit / CUDA allocation required / not LeWM evidence |
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
| `le-wm-main/experiments/run_tworoom_effect_map_panel.py` | paired direction-gate versus root×phase-local RGB effect-map gate | CUDA fresh 32-root panel completed; `0/0/32` paired difference, no incremental planning gain |
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
| `le-wm-main/experiments/gdd_wm/` | GDD-WM D0-D2 proposed workstream | D1 control executed; standalone GDD closed-loop negative; guarded fail-closed recovery reaches `12/20` at zero tolerance but trails visual dynamics `14/20` |
| `le-wm-main/results/gdd_wm_teacher_goal_latent_cuda_20261019/` | goal-latent/goal-relative CUDA teacher-capacity intervention plus raw deployable readout | TwoRoom calibration fallback; Reacher raw predicted-dynamics gate passes; no universal teacher claim |
| `le-wm-main/experiments/run_live_reacher_sequence_candidate_planning.py` with `gdd_wm_teacher_goal_latent_cuda_20261019/reacher_raw_latentrelative_v2/capacity_readout.npz` | independent frozen RGB Reacher consequence for the train-only goal-aware capacity readout | v4 rank-affine repair: `35/64 -> 38/64`, distance delta `-0.02616` with CI excluding zero; success CI crosses zero; Reacher-scoped positive consequence, not statistically established GDD gain; TwoRoom smoke `3/8` |
| `le-wm-main/experiments/gdd_wm/summarize_live_teacher_consequence.py` | paired bootstrap and exact McNemar summary for frozen-teacher live consequences | implemented; preserves same-seed pairing and fail-closed claim boundary |
| `le-wm-main/experiments/gdd_wm/summarize_tworoom_teacher_consequence.py` | paired TwoRoom consequence summary with runtime activation diagnostics | fresh 32-seed CUDA panel: `6/32 -> 7/32`, `p=1.0`; confirms cross-environment ranking boundary |
| `le-wm-main/experiments/run_live_tworoom_planning.py --hybrid-door-phase-reactive` | phase-conditioned fail-closed recovery: RGB waypoint control during door phase, GDD readout retained during goal phase | same 32 roots: `11/32`; improves over unrestricted GDD `7/32`, but remains below RGB reactive `12/32`; recovery diagnostic only |
| `le-wm-main/experiments/gdd_wm/build_frozen_predictive_teacher_package.py` | train-root-only frozen predicted-latent→task-cost teacher; separates teacher score from realized outcome cost | CUDA-compatible control artifact; D-JEPA claim not established |
| `le-wm-main/results/gdd_wm_runtime_teacher_20261019_cpu_diagnostic/` | exact runtime repeated-pulse chart teacher and root-local capacity re-audit | global teacher correlation `0.049`; root-local action-only regret `0.01278`; predicted-dynamics increment fails calibration; CPU diagnostic only |
| `le-wm-main/results/gdd_wm_history4_teacher_20261019_cpu_diagnostic/` | genuine predictor-history-4 teacher plus root-local observed-effect audit | teacher correlation `0.402`, consistency `0.455`; frozen regret `0.2166` vs local observed-effect `0.01069`; diagnostic only |
| `le-wm-main/results/gdd_wm_runtime_history4_teacher_20261019_cpu_diagnostic/` | genuine history-4 predictor on exact runtime repeated-pulse chart | consistency `0.656`; frozen regret `0.2071` vs local observed-effect `0.0220`; diagnostic only |
| `le-wm-main/experiments/gdd_wm/run_formal_online_local_effect_history2_cuda.sh` | regenerate true history-2 runtime bank, fit train-only effect policy, and run fresh-root online verification panel | fail-closed CUDA protocol; queued until host device nodes are exposed |
| `le-wm-main/experiments/gdd_wm/run_formal_online_local_effect_history4_cuda.sh` | regenerate genuine history-4 runtime bank, fit train-only effect policy, and run fresh-root online verification panel | matched-contract re-audit completed on CUDA: support/release restored; HTSC `20/20`, visual baseline also `20/20` |
| `le-wm-main/experiments/gdd_wm/run_formal_online_local_effect_history4_matched_cuda.sh` | one-command deployment-matched history-4 teacher, effect policy, and fresh-root CUDA panel | reproducible matched-contract runner; final panel `20/20` HTSC and `20/20` visual baseline |
| `effect_adapter_rgb_online_probe` in `le-wm-main/experiments/run_live_pointmaze_planning.py` | four predeclared cardinal RGB probes followed by local observed-effect update | same fresh 20-root CUDA panel: `19/20` at tolerance `0.5`; positive versus frozen effect, not versus visual baseline |
| `effect_adapter_rgb_online_verified_local` in `le-wm-main/experiments/run_live_pointmaze_planning.py` | probe-verified local residual may release only when episode-local verification and support pass, while global teacher-bound failure remains explicit | history-4 strict `0.25` CUDA fresh-20: `18/20`, mean final distance `0.2005`, fallback `40.0%`; below visual dynamics `20/20`, no independent GDD gain |
| `effect_adapter_rgb_online_local_jacobian` in `le-wm-main/experiments/run_live_pointmaze_planning.py` | low-dimensional observed RGB effect Jacobian plus frozen task-consistency gate | history-4 gated CUDA fresh-20: `20/20`, mean final distance `0.1352` versus visual `0.1633`, but only `8.1%` local exploitation and paired bootstrap CI crosses zero; fresh external 12-root check also `12/12` but remains small |
| `experiments/run_tworoom_effect_map_panel.py` | root/phase-local RGB effect-map veto versus direction-only gate | CUDA fresh-32 rerun: both `14/32`, paired `0/0/32`; effect-map gate adds no planning gain |
| `le-wm-main/experiments/gdd_wm/audit_active_probe_support.py` | verifies active probes remain inside frozen runtime action support | 100-root/400-probe diagnostic: 99% support pass; no planning claim |
| `le-wm-main/experiments/gdd_wm/build_predicted_latent_feature_table.py` | provenance-checked student feature export with candidate context/rank/outcome inputs forbidden | projection-control input; no planning claim |
| `le-wm-main/experiments/gdd_wm/merge_rgb_successor_splits.py` | strict merge of disjoint train/calibration/outcome RGB successor artifacts with exact key/array coverage | implemented; removed successor-ref plumbing gap |
| `le-wm-main/experiments/gdd_wm/audit_gdd_projection_rgb_selection.py` | frozen projection evaluation on independent RGB successor predictions and held-out task costs | offline RGB-compatible consequence; closed-loop gate still missing |
| `le-wm-main/experiments/gdd_wm/train_projection_distillation.py` | projection-only GDD diagnostic with nested repairability-panel compatibility | completed diagnostic; prediction-preservation and independent planning gates missing |
| `le-wm-main/experiments/gdd_wm/calibrate_teacher_execution_alignment.py` | calibration-only root-bootstrap check that a fixed teacher orders realized task costs above chance | implemented; allows D2 diagnostic only for the exact dataset, never a planning claim |
| `le-wm-main/experiments/gdd_wm/audit_projection_execution_consequence.py` | identity-bound root-paired held-out D2 regret audit with strict-improvement and action-change coverage gates | goal-conditioned control: 1/20 changed roots, upper bound 0; no confirmed real-decision improvement |
| `le-wm-main/experiments/gdd_wm/audit_teacher_counterfactual_contrast.py` | read-only per-root/context diagnosis of predicted-effect collapse, latent-dynamics mismatch, endpoint-readout mismatch, and task-score mismatch | goal-conditioned PointMaze control: outcome has 3/17/12/8 groups respectively; mechanism evidence only, never teacher authorization or planning selection |
| `le-wm-main/experiments/gdd_wm/audit_prospective_repairability_routing.py` | train-root-only repair-family routing pilot with calibration submechanism and joint feature-support gates | old held-action teacher package is correctly rejected: train/calibration and train/outcome action overlap are both 0/5, hence no support-transport claim |
| `le-wm-main/experiments/gdd_wm/run_pointmaze_planning_shaped_repairability.py` | V7 advantage-router execution boundary for a frozen shared-candidate panel | exact held bank and design-root headroom are revalidated at execution; ordered-action advantage ridge selects regularization on train-root-only blocked folds before independent calibration; V7 GPU panel is complete, but its calibration certificate rejected routing, so the raw outcome-router improvement is diagnostic only |
| `le-wm-main/experiments/gdd_wm/preflight_pointmaze_planning_shaped_advantage_v7.py` | freezes V7 root/action support, design-root oracle headroom, router capacity, and executable router interface before any learned fit | V7 static ledger passed support, entropy, top-2-gap, cost-spread, 95-feature/128-root surplus, `advantage_ridge`, ordered-action representation, and train-only blocked-ridge-selection gates; completed CUDA panel isolates prospective-routing instability rather than a candidate-bank confound |
| `le-wm-main/experiments/run_live_pointmaze_planning.py --gdd-score-horizons` | explicit train/deployment horizon contract for frozen GDD projection scoring | CUDA fresh-20 matched audit: `h=4` `3/20`, legacy `h=1,2,4` `3/20`; protocol mismatch fixed, independent planning gain still absent |
| `le-wm-main/experiments/run_live_pointmaze_planning.py --gdd-candidate-count` | explicit candidate-support contract for frozen GDD projection scoring | training-consistent 11-candidate CUDA fresh-20: `3/20`, mean distance `1.6679`; improves continuous error but no success gain over 16-candidate control |
| `le-wm-main/results/gdd_wm_frozen_predictive_teacher_20261020_full16_cuda/capacity_audit/` | full 16-candidate train-only CUDA teacher-capacity audit | predicted-dynamics outcome regret delta `+0.0481` vs action-only, CI crosses zero; teacher increment remains unverified |
| `docs/GDD_WM_PROGRESS_20261019.md` | reader-facing record of frozen predictive teacher, projection diagnostic, runtime contract repairs, and independent PointMaze RGB consequences | 2026-10-19; standalone GDD gain not established; fail-closed boundary documented |
| `le-wm-main/experiments/gdd_wm/join_rgb_realized_refs.py` | provenance-safe join of dataset-matched realized RGB references into a copied candidate bank | audit utility / D1 still gated |
| `le-wm-main/experiments/gdd_wm/audit_nonleaking_teacher_control.py` | train-only predicted-latent task-readout control for separating label leakage from transfer failure | diagnostic only / not D-JEPA or planning evidence |
| `le-wm-main/experiments/gdd_wm/audit_transfer_sufficient_feature_families.py` | fixed train-only source/latent/action feature-family comparison | diagnostic only / no stable family yet |
| `le-wm-main/experiments/gdd_wm/audit_repairability_gate.py` | train-only calibration/nonlinear cross-fit gate separating repairable predicted-dynamics signal from action-only fallback | 7-artifact cross-environment audit: 2/7 permitted, 5/7 fallback; held-out evidence remains diagnostic-only |
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

- Cube macro-ranking online transfer diagnostic (2026-10-17):
  `experiments/run_live_cube_rgb_macro_ranker.py` was added to test the short
  macro ranker with RGB-only current/previous frames, rendered goal, and
  previous action. A reset-distribution smoke selected the zero macro and
  obtained `0/4`; three receding macros also obtained `0/4`. Matching the
  offline action duration (`action_block=1`) did not change this (`0/4`). A
  calibration/test run with the same frozen RGB approach prelude still
  obtained `0/4` and selected the zero macro. Thus the offline `93.75%`
  ranking result does not transfer from guided contact branches to an online
  reset distribution. The remaining failure is an RGB approach/contact-state
  observability gap, not merely macro duration. The runner is retained as a
  fail-closed diagnostic and no online Cube gain is claimed.

- Cube RGB probe-calibrated approach intervention (2026-10-17): the online
  runner now uses a fixed descent phase, a four-direction RGB-only probe fit
  for the local action-to-effector image Jacobian, and the physical red-object
  component for approach error. The probe fit uses only successive RGB frames
  and executed XY actions; simulator state remains evaluator-only. This is a
  diagnostic intervention aimed at separating camera/action chart failure
  from macro ranking failure. It has passed code-level tests but has no GPU
  online outcome yet because the current isolated runtime cannot expose the
  host CUDA device; no success claim is made.

- Cube contact-phase diagnosis (2026-10-17, CPU logic diagnostics only): the
  probe-calibrated runner reduced RGB approach error to about `0.011` but
  initially observed no contact because the historical offline macro used the
  wrong OGB-Cube gripper sign. The online intervention now uses `+1` for
  close and records evaluator-only contact/distance traces. With the corrected
  sign, contact was observed, but full-scale push pulses did not move the
  block. A fixed `push_scale=0.25` produced local distance improvement in one
  smoke (`0.2094` to minimum `0.1883 m`; three receding cycles reached
  `0.1698 m`), while an eight-cycle smoke ended worse (`0.2446 m`). These are
  single-root CPU diagnostics, not GPU or confirmatory planning results. They
  localize the remaining failure to unstable within-contact action effect /
  visual feedback, rather than approach visibility or gripper closure alone.

- Cube RGB online effect-identification intervention (2026-10-18, CPU logic
  diagnostics only): after contact, the runner can fit a local block-pixel
  displacement map from executed push actions, but only after rank-2 support;
  rank-1 histories fail closed and trigger one orthogonal support probe. In a
  single three-cycle smoke this produced `11` valid adaptive pulses and
  reduced distance from `0.2054` to `0.1522 m`. A five-cycle repetition still
  ended at `0.2018 m` (minimum `0.1843 m`) with success `0/1`. The result is a
  mechanism-level improvement and a reproducibility warning, not a stable
  online planning gain; independent GPU roots are required before promotion.
  The implementation passes the full repository regression suite (`899
  passed, 3 skipped`); this is code integrity evidence, not a planning result.

- Cube persistent-contact follow-up (2026-10-18, CPU logic diagnostics only):
  re-running descend/close on every receding cycle was replaced by a
  persistent-contact protocol that performs those phases only once. With
  `push_scale=0.25` and RGB effect adaptation, one five-cycle smoke improved
  final distance to `0.0889 m`, and one seven-cycle smoke to `0.0704 m`, both
  with observed contact. A nine-cycle repetition returned to `0.2054 m`, so
  the gain is strongly root/initial-contact dependent. This is a structural
  improvement hypothesis, not stable online success; it must be evaluated on
  independent GPU roots with fixed arms before entering the main claim.

- Cube RGB progress gate (2026-10-18, CPU logic diagnostics only): a
  fail-closed cycle gate now stops execution when the RGB block-goal error
  increases beyond the frozen threshold. In one five-cycle run the gate did
  not fire, RGB error decreased on every cycle, and final evaluator distance
  reached `0.0846 m`; the ranker nevertheless selected the zero macro once.
  This shows that zero-macro selection is not by itself a failure label and
  that RGB progress is only a safety signal, not a task-success certificate.

- Paired Cube RGB panel (2026-10-18):
  `experiments/run_live_cube_rgb_macro_panel.py` freezes one seed/root split
  and compares full-scale reclose, low-gain reclose, low-gain persistent
  contact, and low-gain effect-adaptive arms. It refuses CPU fallback when
  invoked with `--device cuda`, and writes per-arm artifacts plus a paired
  summary. The panel is implemented and unit-tested but awaits host-GPU
  execution; no arm is promoted from its exploratory CPU smoke result.

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
  projection-only geometry interface with hard D1 and train-only
  repairability gates, plus separate prediction/planning gates. Running it on
  the repeated-context PointMaze
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

- D2 repairability hardening (2026-10-18): projection distillation now
  requires a `permitted_predicted_dynamics` artifact from
  `audit_repairability_gate.py`; an `action_only` fallback blocks D2 before
  feature fitting. This makes the TwoRoom negative result operational rather
  than documentary and prevents a D1-passed but dynamics-unidentified model
  from entering distillation. Updated projection tests cover both permitted
  and fallback paths.

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

- Cube RGB contact-phase isolation (2026-10-18, CPU exploratory): the legacy
  `pre-descend -> RGB XY approach -> close` schedule produced contact on `0/3`
  root-disjoint roots. A mechanism arm that adds `post-approach descend ->
  close` produced contact on `1/3` with the 16-step approach window, and
  `3/3` when the predeclared approach window was extended to 32 steps. This
  repairs contact acquisition in the mechanism audit, but does not establish
  successful pushing or pose closure; the longer approach also increased
  final RGB XY error. The finding decomposes the negative result into a
  phase-order/support bottleneck plus a residual within-contact transfer
  bottleneck. Artifacts:
  `results/cube_contact_phase_audit_cpu_exploratory_20261018_roots0_2` and
  `results/cube_contact_phase_audit_cpu_exploratory_20261018_approach32_roots0_2`;
  code: `experiments/audit_cube_contact_phase.py` and the optional
  `--post-approach-descend-steps/--post-approach-close-steps` arm in
  `experiments/run_live_cube_rgb_macro_ranker.py`.
- Cube within-contact support-gate repair (2026-10-18, CPU exploratory): a
  matched root with post-contact alignment reached contact but the old gate
  kept executing a single rank-one effect direction because it only treated
  small mapping norm as weak. The repaired gate forces an orthogonal probe
  whenever the observed action history is rank deficient; the follow-up
  artifact records `support_probe_forced=2` and rank-2 effect support, while
  task success remains `0/1`. Thus the implementation defect is closed, but
  the remaining failure is genuine within-contact transfer / task geometry,
  not evidence of successful online planning. Artifact:
  `results/cube_postcontact_ranker_cpu_exploratory_20261018_root0_nogate_rankprobe`.
- Cube RGB pick-and-place transport diagnosis (2026-10-18, CPU exploratory;
  PyTorch CUDA initialization was unavailable at run time): world-chart
  approach repaired grasp/lift, but transport without vertical correction
  still dropped the carried block (`0/1`, final distance `0.169 m` in a
  matched recheck). Low-amplitude periodic lift (`+0.15` every 5 steps) and
  sparse full lift (`+1.0` every 5 steps) both worsened transfer (`0.268 m`
  and `0.322 m`), so z-correction is not a valid fix. Exposing the already
  calibrated RGB effect map to the transport controller improved one
  exploratory run to `0.138 m` (transport minimum `0.120 m`) but a same-seed
  repeat produced `0.457 m`; therefore the gain is not stable and is not a
  planning result. Lower-phase XY correction also failed (`0.304 m`, no
  observed contact). The implementation now keeps chart/effect/hybrid
  transport controllers and lower correction as explicit ablations, with
  the default restored to chart. The remaining boundary is unstable
  RGB-only within-carry transfer and reset/runtime variance, not contact
  acquisition. Artifacts:
  `results/live_cube_rgb_pick_place_cpu_20261018_root0_worldapproach_baseline_recheck`,
  `results/live_cube_rgb_pick_place_cpu_20261018_root0_worldapproach_micro15p5`,
  `results/live_cube_rgb_pick_place_cpu_20261018_root0_worldapproach_pulse1p5`,
  `results/live_cube_rgb_pick_place_cpu_20261018_root0_effecthybrid`, and
  `results/live_cube_rgb_pick_place_cpu_20261018_root0_effecthybrid_repeat`;
  code: `experiments/run_live_cube_rgb_pick_place.py`.
- Cube carry-state feedback and horizon/magnitude controls (2026-10-18, CPU
  exploratory): on the same frozen root, RGB relative vertical feedback
  correctly kept the carried block higher during transport (`block_z=0.284 m`
  versus chart `0.218 m`), but final distance worsened to `0.139 m` at gain
  1.0 and `0.136 m` at gain 0.5. A phase-aware re-lift at step 40 worsened
  distance to `0.177 m`. Extending chart transport to 80 steps worsened it to
  `0.167 m`; reducing the action magnitude to `0.1` for 80 steps was the best
  mechanism result at `0.105 m`, still `0/1` success. Effect-conditioned
  transport at the same low magnitude was `0.106 m`, so it did not improve
  over chart. The new RGB carry feedback and re-lift controls are retained as
  explicit ablations, but no controller is promoted: height observability is
  not sufficient for stable within-carry XY transfer. Artifacts:
  `results/live_cube_rgb_pick_place_cpu_20261018_root0_frozen_v2_verticalservo_gain1`,
  `results/live_cube_rgb_pick_place_cpu_20261018_root0_frozen_v2_verticalservo_gain05`,
  `results/live_cube_rgb_pick_place_cpu_20261018_root0_frozen_v2_relift40_steps80`,
  `results/live_cube_rgb_pick_place_cpu_20261018_root0_frozen_v2_chart_steps80`,
  `results/live_cube_rgb_pick_place_cpu_20261018_root0_frozen_v2_chart_mag01_steps80`,
  and `results/live_cube_rgb_pick_place_cpu_20261018_root0_frozen_v2_effect_mag01_steps80`;
  code: `experiments/run_live_cube_rgb_pick_place.py`.
- Cube reset-variance and transport-controller control (2026-10-18, CPU
  exploratory): repeated runs with the same nominal seed previously changed
  the initial cube-goal geometry because the runtime did not freeze
  `variation_values` and `qpos/qvel`. The pick-place runner now writes v2
  artifacts with frozen calibration/test initial conditions and can replay
  them exactly. On one frozen root, effect-hybrid transport reached final
  distance `0.14486 m`; replaying the same calibration and initial state gave
  `0.14535 m`, while the matched chart baseline gave `0.13092 m`. All were
  `0/1` task success. Thus reset variance was an implementation confound,
  but exposing the RGB effect map did not solve transfer and is currently
  worse than chart on the matched root. Artifacts:
  `results/live_cube_rgb_pick_place_cpu_20261018_root0_frozen_v2_effecthybrid`,
  `results/live_cube_rgb_pick_place_cpu_20261018_root0_frozen_v2_effecthybrid_replay`,
  and `results/live_cube_rgb_pick_place_cpu_20261018_root0_frozen_v2_chart_baseline`;
  code: `experiments/run_live_cube_rgb_pick_place.py`.
- Cube calibration-only approach fallback panel (2026-10-18, project
  `.venv` CUDA-enabled runtime; MuJoCo dynamics remain CPU-bound): a fixed
  50/50 blend of the world-coordinate approach chart and the RGB pixel
  Jacobian was added as `--approach-controller hybrid`. On the same frozen
  root-disjoint 3-root panel and the same close/transport protocol, world
  chart was `0/3` with mean final distance `0.0925 m`, pixel-only was `0/3`
  with `0.1721 m`, and hybrid was `1/3` with `0.0785 m`. This is a genuine
  mechanism improvement over both controls but not a stable planning result;
  the hybrid success is concentrated on one root and root 2 remains an
  approach/contact-geometry failure. A fixed 32-step post-descend RGB
  re-approach worsened hybrid to `0/3` and `0.1022 m`, so merely extending
  the contact phase is not the remedy. Artifacts:
  `results/cube_approach_fallback_panel_20261018/{world,pixel,hybrid,hybrid_postdesc32}.json`;
  code: `experiments/run_live_cube_rgb_pick_place.py`.
- Cube task-conditioned approach and readiness-gate follow-up (2026-10-18,
  same frozen 3-root panel): fitting separate approach charts per task from
  calibration roots only did not improve the fixed hybrid controller (`1/3`,
  `0.0784 m`). A predeclared minimum of eight RGB approach steps removed the
  false early-stop on the difficult root and reduced its final distance from
  `0.1437 m` to `0.1102 m`; aggregate performance became `1/3`, `0.0674 m`.
  Because success did not improve and the parameter was not selected from test
  outcomes, this is an interface repair plus residual task-geometry boundary,
  not stable online planning evidence. The new runtime flag is
  `--approach-min-steps`; artifacts:
  `results/cube_approach_fallback_panel_20261018/hybrid_task_conditioned.json`
  and `hybrid_task_conditioned_min8.json`.
- Cube contact-interface close-duration control (2026-10-18): with the
  repaired fixed eight-step RGB readiness gate and task-conditioned hybrid
  approach, reducing table-contact close duration from 15 to the predeclared
  five-step control changed the frozen 3-root panel from mean `0.0674 m` to
  `0.0652 m`, while success stayed `1/3` and the difficult root stayed above
  threshold (`0.1053 m`). Thus the remaining error is not explained by close
  dwell alone; it is within-contact action-effect transfer. Artifact:
  `results/cube_approach_fallback_panel_20261018/hybrid_task_conditioned_min8_close5.json`.
- Cube planner-reachability horizon intervention (2026-10-18): keeping the
  approach, calibration, contact, and transport magnitude fixed, extending
  the table-push transport horizon from 105 to a predeclared 180 steps changed
  the frozen pilot panel from `1/3` success and `0.0652 m` mean distance to
  `2/3` and `0.0435 m`. This confirms a genuine planner-reachability component
  of the earlier negative result, but it is not a learned-dynamics result.
  On a new independent 3-root panel with the frozen 180-step protocol,
  performance was only `1/3`, contact `1/3`, and mean distance `0.1062 m`.
  The positive pilot therefore does not establish cross-root stability.
  Artifacts:
  `results/cube_approach_fallback_panel_20261018/hybrid_task_conditioned_min8_close5_transport180.json`
  and `independent_seed20261904_h180.json`.
- Cube online/local-chart follow-ups (2026-10-18): an RGB-only non-progress
  fallback raised independent-panel contact from `1/3` to `3/3` but left
  success at `1/3` and worsened mean distance to `0.1237 m`; a reset-safe
  per-root RGB Jacobian probe also left success at `1/3` and worsened mean
  distance to `0.1259 m`. Lowering approach gain to `0.1` reduced contact to
  `2/3` and worsened mean distance to `0.1653 m`. These controls separate
  contact acquisition from usable action-effect transfer and are retained as
  negative mechanism arms. Artifacts:
  `independent_seed20261904_h180_adaptive.json`,
  `independent_seed20261904_h180_localpixel.json`, and
  `independent_seed20261904_h180_gain010.json`; code:
  `experiments/run_live_cube_rgb_pick_place.py`.
- Cube within-contact action-support intervention (2026-10-18): added
  `greedy_effect`, which restricts transport actions to eight directions and
  selects the direction with the largest observed RGB-effect projection toward
  the goal, avoiding unconstrained pseudoinverse actions. On the independent
  frozen panel it produced contact `2/3`, success `1/3`, and mean distance
  `0.1715 m`, worse than the chart control (`0.1062 m`). Thus fixed action
  support alone does not repair transfer; the useful effect is
  mode-conditioned and cannot be inferred from a single global RGB map.
  Artifact: `results/cube_approach_fallback_panel_20261018/independent_seed20261904_h180_greedy_effect.json`;
  code: `experiments/run_live_cube_rgb_pick_place.py`.
- Cube post-close contact-mode intervention (2026-10-18): added a fixed
  RGB-only re-approach after gripper close and before effect probes. On the
  independent panel it raised contact from `1/3` to `2/3`, and improved one
  difficult root (`0.1612 -> 0.1379 m`), but worsened another (`0.1308 ->
  0.1663 m`); success remained `1/3` and mean distance was `0.1099 m`.
  Therefore post-close re-approach is mode-dependent and cannot be enabled
  unconditionally. Artifact:
  `results/cube_approach_fallback_panel_20261018/independent_seed20261904_h180_postclose8.json`;
  code: `experiments/run_live_cube_rgb_pick_place.py`.
- Cube RGB geometry gate for contact-mode selection (2026-10-18,
  exploratory): close-phase RGB block-goal gaps on a diagnostic panel were
  about `0.74 px` for the difficult far-geometry root and `0.28/0.27 px` for
  the two visually near roots. The runner now supports
  `--post-close-reapproach-pixel-threshold` and records the RGB gap in every
  phase trace, so re-approach can be selected without task ID or privileged
  contact. A h240 replay with threshold `0.5 px` produced `1/3` success and
  `0.1148 m` mean distance; its phase trace confirms the gate selected
  re-approach only for the near roots. An earlier unconditional h240 run was
  `2/3`, but the difference was not stable under the same frozen initial
  conditions, so neither result is promoted as a general method. Artifact:
  `results/cube_approach_fallback_panel_20261018/independent_seed20261904_h240_rgbgate05.json`;
  code: `experiments/run_live_cube_rgb_pick_place.py`.
- Cube calibration-device reproducibility repair (2026-10-18): two nominally
  identical RGB-gate replays had silently used different chart-fit devices
  (`cpu` versus `cuda`), producing `0.1148 m` versus `0.1590 m` mean distance.
  The runner now exposes `--fit-device {auto,cpu,cuda}`, rejects an unavailable
  forced CUDA device, and records the resolved device in the artifact. With
  the same frozen initial conditions and explicit `--fit-device cpu`, two
  repeats were `0.1644 m` and `0.1605 m`, both `1/3` success. This closes a
  reproducibility confound but does not create a planning gain. Artifacts:
  `results/cube_approach_fallback_panel_20261018/independent_seed20261904_h240_rgbgate05_fixedcpu.json`
  and `..._fixedcpu_repeat.json`; code:
  `experiments/run_live_cube_rgb_pick_place.py`.
- Cube grasp-interface timing controls (2026-10-18, fixed CPU replay): adding
  three post-probe lift steps left task1/2 RGB effect probes effectively zero
  and kept success at `1/3` (`0.1600 m` mean distance). Moving five lift steps
  before close was worse (`0/3` success, contact `2/3`, `0.1962 m`). These
  controls rule out simple height timing as the missing transfer mechanism;
  evaluator finger contact is not equivalent to a block-effect contact mode.
  Artifacts:
  `results/cube_approach_fallback_panel_20261018/independent_seed20261904_h240_rgbgate05_fixedcpu_lift3.json`
  and `..._fixedcpu_grasplift5.json`.
- Cube oracle phase-order repair (2026-10-18, fixed CPU independent panel):
  the environment's official oracle uses above-block XY alignment, descent,
  fine alignment, then close. The runner now exposes
  `--approach-before-descend`; with a fixed 32-step post-descend fine
  alignment and table-push transport, the independent 3-root panel reached
  `2/3` success and `0.0935 m` mean distance (the 16-step fine-alignment
  control was also `2/3`, `0.0988 m`). Task1 and task3 acquired measurable
  block effect; task2 acquired finger contact but still had weak effect.
  This is the first direct repair of the earlier pre-descend lateral-contact
  bug, but remains a table-push mechanism result, not frozen-LeWM planning.
  Artifacts:
  `results/cube_approach_fallback_panel_20261018/independent_seed20261904_h240_above_hybrid_fine32_fixedcpu.json`
  and `..._fine16_fixedcpu.json`.
- Cube oracle-order lift control (2026-10-18, fixed CPU): moving lift before
  effect probes, as required for pick-and-place, produced `0/3` success and
  `0.1472 m` mean distance. Task1/3 blocks were lifted (`block_z` about
  `0.29 m`) but transport failed, so the table-push positive cannot be
  promoted to pick-and-place or carry transfer. Artifact:
  `results/cube_approach_fallback_panel_20261018/independent_seed20261904_h240_above_fine32_lift15_beforeprobes_fixedcpu.json`.
- Cube carry retention follow-up after phase-order repair (2026-10-18,
  fixed CPU): moving transport to the post-lift RGB effect chart did not
  preserve the grasp; task3 dropped from `z≈0.29` to the table and success
  remained `0/3`. A sparse `+0.15` vertical correction every five transport
  steps was also negative (`0/3`, `0.1829 m` mean distance). Thus the carry
  failure is not repaired by swapping XY effect charts or by naive z hold;
  stable grasp/contact retention requires a separate mode-conditioned
  interface. Artifacts:
  `results/cube_approach_fallback_panel_20261018/independent_seed20261904_h240_above_fine32_lift15_effect_fixedcpu.json`
  and `..._sparsehold_fixedcpu.json`.
- Cube controlled-lift carry follow-up (2026-10-18, fixed CPU): reducing
  oracle-ordered lift from 15 to 5 steps improved the carry arm from `0/3`
  to `1/3` success and `0.1116 m` mean distance; task1 reached `0.0315 m`.
  This confirms that the previous lift impulse was too aggressive. Adding
  RGB relative-height servo feedback at gain `0.5` did not improve the panel
  (`1/3`, `0.1112 m`), and task3 still failed. Thus carry transfer remains a
  mode-conditioned XY effect problem after grasp, not a height-only problem.
  Artifacts:
  `results/cube_approach_fallback_panel_20261018/independent_seed20261904_h240_above_fine32_lift5_beforeprobes_fixedcpu.json`
  and `..._lift5_servo05_fixedcpu.json`.
- Cube carry vertical-hold ablation (2026-10-18, fixed CPU): replacing the
  failed RGB relative-height servo with a predeclared continuous vertical
  hold did not recover carry. Full-amplitude hold (`z=1.0`) gave `0/3`
  success, `0.1787 m` mean distance, and `2/3` contact rate; a lower
  calibrated amplitude (`z=0.15`) also gave `0/3`, `0.1994 m`, and `2/3`
  contact. Both were worse than the matched lift5 no-hold arm (`1/3`,
  `0.1116 m`). This rejects continuous vertical hold as the carry repair;
  `--transport-lift-hold-magnitude` is retained only for reproducible
  ablation. Artifacts:
  `results/cube_approach_fallback_panel_20261018/independent_seed20261904_h240_carrydiag_lift5_hold_fixedcpu.json`
  and `..._lift5_hold015_fixedcpu.json`.
- Cube pre-close micro-lift carry probe (2026-10-18, fixed CPU): adding two
  pre-close steps with simultaneous small lift and gripper closure changed
  the same frozen 3-root panel to `1/3` success and `0.0904 m` mean distance,
  versus `1/3` and `0.1116 m` for the lift5 no-hold control. Task2 improved
  from `0.2198 m` to `0.1418 m`, but task3 still failed and contact rate was
  only `2/3`. This is a local grasp-geometry signal, not stable carry or
  pick-and-place evidence; it motivates an independent grasp-mode panel
  rather than promotion of the controller. Artifact:
  `results/cube_approach_fallback_panel_20261018/independent_seed20261904_h240_carrydiag_grasplift2_lift5_fixedcpu.json`.
- Cube low-impact carry ablation (2026-10-18, fixed CPU): jointly reducing
  horizontal transport and vertical hold to `0.03` did not preserve carry;
  the matched panel was `0/3`, `0.1555 m`, with `2/3` contact rate. Task3
  still returned to table height, while task1 regressed relative to the
  lift5 no-hold control. Together with the `z=1.0`, `0.15`, and `0.03`
  vertical-only controls, this rules out a single action-amplitude repair.
  The remaining bottleneck is a mode-conditioned grasp interface: RGB effect
  support measured after lift is not sufficient to guarantee retained carry.
  Artifact:
  `results/cube_approach_fallback_panel_20261018/independent_seed20261904_h240_carrydiag_xy003_z003_fixedcpu.json`.
- Cube RGB effect-loss re-grasp intervention (2026-10-18, fixed CPU): an
  online RGB-only gate was added that monitors effect magnitude, effect
  direction, and a post-lift block--effector gap; after bounded effect loss it
  performs at most one release/approach/descend/close/lift recovery. The
  direction-only version gave `0/3`, `0.1038 m`; adding the self-calibrated
  gap gate gave `0/3`, `0.1125 m`. Task3 still dropped without triggering the
  gate: its RGB gap stayed in `0.213--0.218`, despite evaluator block height
  returning to the table. Thus the current RGB rendering does not expose a
  reliable carry/drop state, and a more elaborate RGB gate cannot repair an
  unobserved mode. The controller now records
  `effect_loss_trace`, `rgb_gap_loss`, and `regrasp_count` for this boundary.
  Artifacts:
  `results/cube_approach_fallback_panel_20261018/independent_seed20261904_h240_carry_regrasp_effectloss_align02_fixedcpu.json`
  and `..._carry_regrasp_gap08_fixedcpu.json`.
- Cube privileged reachability ceiling (2026-10-18, fixed EGL): the official
  OGBench `CubeMarkovOracle`, run with simulator state on the same frozen
  three roots, reached `3/3` with mean final XY distance `0.00285 m`. This
  separates the earlier `0/3` fixed pick-place bank ceiling from task
  impossibility: the full closed-loop oracle can solve these roots, while the
  declared RGB-only carry primitive cannot. The result is an evaluator-only
  ceiling, not visual planning evidence. Reproducible script and artifact:
  `experiments/audit_cube_markov_oracle_ceiling.py` and
  `results/cube_approach_fallback_panel_20261018/cube_markov_oracle_ceiling_seed20261904.json`.
- Cube RGB visual-scale carry servo (2026-10-18, fixed CPU): a new RGB-only
  controller estimated a local block-area response to a small post-lift z
  probe and closed the loop on area error during transport. The matched
  panel was `0/3`, `0.1875 m`; task1/task3 had finite slopes and executed
  servo corrections, but the corrections destabilized transport, while task2
  had no usable slope. This rejects red-component area as a reliable height
  observation for the current renderer. The implementation remains an
  auditable ablation, not a promoted carry policy. Artifact:
  `results/cube_approach_fallback_panel_20261018/independent_seed20261904_h240_carry_visual_scale_servo05_fixedcpu.json`.

- GDD-WM train-only repairability replication (2026-10-18, CUDA when
  available): the capacity audit now has an explicit `--fit-device
  {auto,cpu,cuda}` contract and records the resolved device instead of
  silently refusing a runtime whose PyTorch CUDA initialization is
  unavailable. On matched two-context Reacher data, calibration selected a
  linear latent-residual readout and the held-out outcome regret delta versus
  action-only was `-0.5568` with paired root bootstrap CI
  `[-0.9933,-0.1958]`. However, the shuffled-prediction control was still
  better than the unshuffled predicted-dynamics readout, so this is not
  evidence that learned dynamics caused the gain. On matched TwoRoom data,
  the strict bilinear train-crossfit gate selected `action_only`; the
  predicted-dynamics gate did not pass and the held-out linear-residual delta
  was `-17.90` with CI crossing zero. These paired replications localize the
  negative result to architecture/environment-dependent capacity and
  action-template confounding. Artifacts:
  `results/gdd_wm_repairability_20261018/reacher_history1_vs3/` and
  `results/gdd_wm_repairability_20261018/tworoom_history1_vs3/`; code:
  `experiments/gdd_wm/audit_train_only_teacher_capacity.py`.
- GDD-WM repairability authorization gate (2026-10-18): added
  `experiments/gdd_wm/audit_repairability_gate.py`, which authorizes a
  predicted-dynamics readout only when both calibration non-degradation and
  root-disjoint train cross-fit dynamics gates pass. The matched panel
  authorizes Reacher as a train-only diagnostic (`1/2` environments), but
  falls back to action-only on TwoRoom. Held-out shuffled-control ambiguity is
  reported separately and cannot authorize the gate; Reacher is therefore not
  promoted as learned-dynamics or planning evidence. The panel artifact is
  `results/gdd_wm_repairability_20261018/repairability_gate_panel.json`.
- GDD-WM latent-relative goal conditioning intervention (2026-10-18, matched
  Reacher/TwoRoom panels): adding frozen `goal_latent_model - source_latent`
  and `predicted_latent - goal_latent_model` features improved Reacher's
  linear-residual held-out regret delta from `-0.5568` to `-0.6737`, with CI
  `[-1.1049,-0.3018]`, and removed the Reacher shuffled-control ambiguity.
  TwoRoom changed the linear-residual delta from `-17.90` to `-17.46` with a
  CI still crossing zero; its root-disjoint dynamics gate still selected
  `action_only`, and the shuffled control remained better. Thus goal
  conditioning is a scoped Reacher repair candidate, not a universal fix.
  Artifacts:
  `results/gdd_wm_repairability_20261018/reacher_goal_latent_history1_vs3/`,
  `results/gdd_wm_repairability_20261018/tworoom_goal_latent_history1_vs3/`,
  and `results/gdd_wm_repairability_20261018/goal_latent_gate_panel.json`.
- GDD-WM goal-relative action-frame intervention (2026-10-18, TwoRoom,
  matched CUDA panel): expressing the fixed action summaries in the
  source-to-goal frame reduced the action-only held-out regret from `57.78` to
  `2.05`. This is a strong decision-chart repair, but adding the latent
  residual was harmful (`+15.56` regret delta, CI `[+2.43,+31.76]`), and both
  calibration and predicted-dynamics cross-fit gates rejected it. The formal
  controller therefore keeps the goal-relative action chart and disables the
  latent residual on this environment. This separates a real task-metric/
  action-frame repair from a learned-dynamics repair. Artifact:
  `results/gdd_wm_repairability_20261018/tworoom_goalframe_goal_latent_history1_vs3/`
  and gate:
  `results/gdd_wm_repairability_20261018/tworoom_goalframe_gate.json`.
- GDD-WM root-centered target control (2026-10-18, TwoRoom, matched CUDA
  panel): after the goal-relative action-frame repair, subtracting each
  training root's mean task cost did not rescue the latent residual. The
  calibration gate still fell back to action-only; the linear residual stayed
  at `+15.56` held-out regret delta, while the full predicted-dynamics term
  was `+2.91` with CI crossing zero. This rules out absolute cross-root cost
  scale as the primary cause and localizes the remaining failure to
  within-support action-effect identifiability. Artifact:
  `results/gdd_wm_repairability_20261018/tworoom_goalframe_goal_latent_rootcentered_history1_vs3/`.
- GDD-WM proposal-relative action control (2026-10-18, TwoRoom, matched CUDA
  panel): normalizing each candidate action summary inside its root/context
  proposal cloud produced the same metrics as the raw goal-latent control
  (`-17.46` linear-residual delta, dynamics gate rejected). This null control
  rules out simple action-amplitude/chart scaling as the remaining cause;
  within-support action-effect identifiability remains unresolved. Artifact:
  `results/gdd_wm_repairability_20261018/tworoom_proposalrelative_goal_latent_history1_vs3/`.
- GDD-WM final-stage support-aware residual consequence (2026-10-18, TwoRoom,
  fresh CUDA RGB panel): a residual calibrated only on the final (`cem_iteration
  = 7`) proposal set passed the root-macro calibration gate (`1/144` candidates;
  calibration delta mean `-0.8702`, paired bootstrap CI
  `[-2.0295,-0.1396]`). However, the decisive independent comparison on the same
  32 roots (`20261800--20261831`) found no downstream change: the
  `capacity_safe_readout` and frozen Euclidean base produced the identical
  success vector, `10/32` versus `10/32`, with the same `8/0/24` paired
  wins/losses/ties against the random arm. The apparent `+0.25` improvement is
  therefore a frozen-base consequence, not evidence for a residual gain. The
  residual remains a calibrated offline diagnostic candidate, but its independent
  planning increment is currently `0`; it must not be promoted until a fresh
  paired outcome shows a change relative to frozen base. Artifacts:
  `results/gdd_wm_cross_environment_20261018/live_tworoom_final_stage_residual_fresh32/`
  and
  `results/gdd_wm_cross_environment_20261018/live_tworoom_frozen_base_fresh32/`;
  calibration artifact:
  `results/gdd_wm_cross_environment_20261016/tworoom_cem_final_stage_residual/`.
- GDD-WM final-stage runtime-scope repair audit (2026-10-18, TwoRoom, CUDA):
  trace comparison first exposed that the previous artifact was silently
  disabled by the runtime's explicit-bank fail-closed rule; base and residual
  CEM traces were exactly identical. The artifact contract was repaired with
  an explicit `support_scope=full_cem_final_stage`, while old bank-only
  artifacts retain fail-closed behavior. After regeneration and an independent
  paired fresh-32 run, the residual was active but did not help: frozen base
  achieved `10/32`, repaired residual `9/32`, with `0/1/31`
  residual-wins/losses/ties and success delta `-0.03125` (the only changed
  root was `20261820`). This correctly localizes the remaining negative result
  to the learned final-stage correction's out-of-root transfer, rather than a
  runtime no-op. Artifacts:
  `experiments/capacity_readout_runtime.py`,
  `results/gdd_wm_cross_environment_20261018/tworoom_cem_final_stage_residual_fixed/`,
  `results/gdd_wm_cross_environment_20261018/live_tworoom_final_stage_residual_fixed_fresh32/`,
  and
  `results/gdd_wm_cross_environment_20261018/live_tworoom_frozen_base_fresh32/`.
- GDD-WM enlarged calibration audit (2026-10-18, TwoRoom, CUDA): rebuilding
  the full CEM trajectory panel with 64 root-disjoint seeds (`32` train,
  `16` calibration, `16` outcome) removed the apparent small-panel
  authorization: `calibration_safe_candidate_count=0`, so the formal gate
  correctly falls back to frozen base. The best exploratory candidate had
  calibration delta mean `-0.9779` but CI `[-3.2441,+0.8618]`; on the held-out
  16 roots its regret delta was `+0.5468`, CI `[-2.3574,+4.0983]`. This confirms
  that the earlier `8`-root calibration was underpowered and that the learned
  final-stage correction does not yet transfer reliably. The repair is now
  fail-closed rather than outcome-tuned. Artifacts:
  `results/gdd_wm_cross_environment_20261018/tworoom_cem_trajectory_panel_64roots/`
  and
  `results/gdd_wm_cross_environment_20261018/tworoom_cem_final_stage_residual_64roots/`.
- GDD-WM no-dynamics Reacher replication (2026-10-18, CUDA, fresh 64 roots):
  a source/action-only MLP readout was tested against the frozen Euclidean
  sequence-bank CEM on exactly the same roots (`20264000--20264063`). It gave
  `5/4/55` wins/losses/ties, success `37/64` versus base `36/64`, but worsened
  mean final qpos distance by `+0.00227`; this is not a stable positive
  consequence. The stricter support-fraction gate abstained on all `35` active
  rows, yielding an exact `0/0/64` delta and no distance change. The result
  separates two useful properties: no-dynamics MLP can perturb decisions, but
  its current transfer is not reliable; support gating prevents degradation by
  falling back to frozen base, at the cost of zero intervention. Artifacts:
  `results/gdd_wm_cross_environment_20261018/live_reacher_base_fresh64/`,
  `results/gdd_wm_cross_environment_20261018/live_reacher_mlp_fresh64/`, and
  `results/gdd_wm_cross_environment_20261018/live_reacher_mlp_support_gate_fresh64/`.

- TwoRoom privileged goal-directed reachability ceiling (2026-10-18,
  evaluator-only replay): a source/goal-state candidate oracle was evaluated on
  16 fresh roots, using horizon `5`, action block `5`, and 289 candidates per
  root. Only `9/16` roots (`56.25%`) contained any successful candidate;
  `7/16` remained unreachable even after candidate construction used privileged
  source and goal positions. This is not visual-planning or learned-dynamics
  evidence. It establishes an independent planner reachability ceiling: the
  earlier CEM failures cannot all be attributed to latent ranking, and residual
  tuning must not proceed on those roots until the action interface or horizon
  is repaired. Artifact:
  `results/gdd_wm_cross_environment_20261018/tworoom_goal_directed_oracle16/report.json`;
  implementation: `experiments/audit_tworoom_goal_directed_oracle.py`.
- TwoRoom candidate-coverage interventions (2026-10-18, fresh CUDA RGB roots
  `20265000--20265015`): expanding the fixed bank from the legacy 17 templates
  to 33 evenly spaced angular templates did not change the outcome (`1/16`
  success, identical success vector). A second RGB-only bank built from the
  current/goal red-marker displacement with eight angular offsets and two
  magnitudes also produced the identical `1/16` outcome. These are paired
  negative results: simple angular coverage and direct RGB goal direction do
  not rescue the frozen model on this panel. They localize the remaining
  failure beyond fixed candidate density, toward candidate sequence structure
  (door/subgoal traversal) or learned cost ranking. Artifacts:
  `results/gdd_wm_cross_environment_20261018/live_tworoom_angular_bank_compare_legacy16/`,
  `.../live_tworoom_angular_bank_compare_angular16/`, and
  `.../live_tworoom_angular_bank_compare_vision_goal16/`.
- TwoRoom RGB door/subgoal candidate intervention (2026-10-18, same fresh CUDA
  roots): a visual door detector generated two-stage candidates (visible-door
  approach followed by goal-directed motion), with fallback-safe RGB inputs and
  no simulator-state access. It still produced `1/16` success with the exact
  same success vector as legacy, angular16, and direct vision-goal banks. This
  rejects the hypothesis that the current negative consequence is explained
  solely by missing door-crossing sequence templates. The remaining live
  bottleneck is learned candidate ranking or the short-horizon/control-step
  interface, and no candidate variant is promoted as the formal controller.
  Artifact: `results/gdd_wm_cross_environment_20261018/live_tworoom_angular_bank_compare_vision_door16/`.
- TwoRoom horizon intervention (2026-10-18, same fresh CUDA roots): extending
  the frozen RGB CEM horizon from `5` to `10` while retaining the legacy bank
  reduced success from `1/16` to `0/16` on the paired fresh panel. This rules
  out the simple claim that the current negative consequence is only caused by
  insufficient prediction/planning horizon; longer imagined rollouts can
  amplify the existing ranking error under this checkpoint. Artifact:
  `results/gdd_wm_cross_environment_20261018/live_tworoom_horizon10_legacy16/`.
- TwoRoom RGB task-geometry control (2026-10-18, same fresh CUDA roots): an
  outcome-free RGB-only cost model scored candidates by cumulative visual
  displacement, using a visible-door waypoint for cross-wall roots. It achieved
  `0/16`, below the frozen latent CEM's `1/16`. Thus replacing latent cost with
  a hand-coded RGB task metric is not sufficient; the remaining failure also
  involves action-effect/control-interface mismatch. This baseline is retained
  as a diagnostic control, not a proposed method. Artifact:
  `results/gdd_wm_cross_environment_20261018/live_tworoom_vision_geometry16/`.
- TwoRoom RGB reactive control ceiling (2026-10-18, fresh 32-root paired panel):
  a pure RGB feedback controller that re-observes after every one-step pulse
  and steers through a visually detected door waypoint achieved `16/32`
  successes. Frozen RGB LeWM CEM on the exact same roots achieved `8/32`; the
  paired comparison is `10/2/20` wins/losses/ties for reactive minus frozen,
  with success delta `+0.25`. This confirms that the action interface and
  visible task geometry can support substantially better control; the negative
  frozen-LeWM result is therefore not an unavoidable environment reachability
  failure. The controller is an evaluator-only diagnostic and is not claimed
  as learned-world-model evidence. Artifact:
  `results/gdd_wm_cross_environment_20261018/tworoom_rgb_reactive_controller32/report.json`.
- TwoRoom door detector repair audit (2026-10-18): the original RGB door
  proposal used the longest white run over the full image and could select
  wall-exterior background. The detector now restricts the gap search to the
  observed dark wall span, with unit tests. Re-running the corrected door bank
  and RGB geometry controls yielded `1/16` and `0/16`, respectively, so the
  corrected negative conclusions remain unchanged. Artifacts:
  `results/gdd_wm_cross_environment_20261018/live_tworoom_angular_bank_compare_vision_door16_fixed/`
  and `.../live_tworoom_vision_geometry16_fixed/`.
- TwoRoom latent/action ranker transfer audit (2026-10-18, CUDA, root-disjoint):
  a ridge readout trained on 16 calibration roots reduced offline held-out
  candidate-bank regret from `47.36` to `26.97`, but its independent live
  consequence did not improve. Frozen base achieved `5/16`; unrestricted
  ranker extrapolation achieved `3/16`. A support-gated ranker (bank candidates
  only, base-cost scale alignment for Gaussian proposals) also achieved `3/16`,
  and a one-shot ranker followed by frozen CEM achieved `4/16`. This isolates
  the offline-to-online gap: static candidate-bank readout gains do not
  transfer to receding-horizon execution, even after support gating. Artifacts:
  `results/gdd_wm_cross_environment_20261018/tworoom_latent_action_ranker_fresh32.json`,
  `results/gdd_wm_cross_environment_20261018/live_tworoom_ranker_fresh16_base/`,
  `.../live_tworoom_ranker_fresh16_supportgated/`, and
  `.../live_tworoom_ranker_fresh16_one_shot/`.
- TwoRoom RGB risk-gated fallback intervention (2026-10-18, CUDA, two fresh
  32-root panels): the frozen LeWM CEM action is compared with a one-pulse
  RGB reactive action using only the current RGB frame and rendered RGB goal.
  A fallback is triggered by a predeclared 90-degree direction conflict or a
  model-action norm below `0.08`; the next observation re-evaluates the gate.
  On panel A, frozen LeWM achieved `10/32`, always-reactive `11/32`, and the
  hybrid `14/32`. On independent panel B, the corresponding results were
  `5/32`, `13/32`, and `13/32`. Across 64 roots, frozen LeWM achieved `15/64`,
  always-reactive `24/64`, and hybrid `27/64`; hybrid versus frozen had
  `15/3/46` wins/losses/ties. This is a positive intervention on the learned
  action-ranking bottleneck and is close to the RGB reactive ceiling, but the
  paired data do not establish superiority over the reactive controller and
  are not evidence that frozen LeWM alone solves planning. The first panel
  was run before a telemetry-only NaN exclusion fix; its success outcomes are
  retained, while gate-distribution statistics are taken only from the
  corrected second panel. Artifacts:
  `results/gdd_wm_cross_environment_20261018/tworoom_rgb_hybrid_fallback32/`,
  `.../tworoom_rgb_hybrid_fallback32_b/`; implementation and tests:
  `experiments/run_live_tworoom_planning.py`,
  `tests/test_live_tworoom_planning.py`.
- TwoRoom online-effect extension (implementation, GPU rerun pending): the
  same fail-closed RGB gate now optionally fits a root-and-phase-local online
  least-squares map from executed 2-D actions to observed red-marker
  displacement. Maps are never pooled across parallel roots or door/goal
  phases. After at least
  four observations with rank-two action support, a model action is
  additionally vetoed when its predicted local image effect has non-positive
  progress toward the RGB reactive direction. The map uses only consecutive
  RGB observations and the previously executed action; it never reads
  simulator state, task outcomes, or privileged contact labels. This arm is
  exposed by `--hybrid-use-effect-map` and must be evaluated on fresh roots
  before any claim is made. The current positive numbers above therefore
  refer only to the direction-conflict gate, not this extension. Runtime now
  records per-root gate calls, fallback counts, phase counts, and effect-map
  rank so the eventual GPU audit can test whether gains are concentrated in a
  particular contact/door phase rather than pooling them invisibly. The
  deployment gate additionally requires at least six local observations,
  rank-two action support, condition number at most `100`, and normalized fit
  residual at most `0.75`; otherwise the map is diagnostic only and cannot
  veto the model action.
- TwoRoom failure decomposition artifact (2026-10-18): a unified descriptive
  report now keeps the privileged reachability ceiling, learned sequence
  ranking, offline ranker transfer gap, and RGB fallback consequence in
  separate panels. It reports `7/16` oracle-unreachable roots, sequence
  ranker mean Spearman `0.041` and top-1 `9.375%`, offline ranker regret
  reduction `20.39` without an online guarantee, and the two-panel hybrid
  result `27/64` versus frozen `15/64`. This artifact is an attribution aid,
  not a pooled causal estimate. Artifact:
  `results/gdd_wm_cross_environment_20261018/tworoom_failure_decomposition.json`;
  implementation:
  `experiments/summarize_tworoom_failure_decomposition.py`.
- TwoRoom paired uncertainty audit (2026-10-18): on the fixed two-panel
  64-root outcome set, hybrid minus frozen has mean success delta `+0.1875`,
  paired bootstrap 95% interval `[+0.0625,+0.3125]`, and exact two-sided
  sign-flip `p=0.00754` over 18 non-tied roots. Hybrid minus reactive has
  mean delta `+0.046875`, interval `[-0.015625,+0.109375]`, and `p=0.375`
  over only 5 non-tied roots, so no superiority over the reactive controller
  is claimed. Artifact:
  `results/gdd_wm_cross_environment_20261018/tworoom_hybrid_paired_statistics.json`;
  implementation:
  `experiments/audit_tworoom_hybrid_statistics.py`.
- The GPU rerun is frozen in
  `experiments/run_tworoom_effect_map_panel.py`: direction-gate and
  root-local effect-map arms use exactly the same fresh roots, with the
  effect-map gate as the only intervention. The panel refuses CUDA requests
  when CUDA is unavailable and reports paired wins/losses/ties plus per-root
  support diagnostics. On the fresh 32-root CUDA panel
  `results/tworoom_effect_map_panel_20261018_fresh_v2/panel.json`, both
  hybrid arms both obtained `14/32` (random control `3/32`) and the paired comparison was `0/0/32`; the effect
  map was usable on 31/32 roots but its fallback rate was 54.1%. This closes
  the simple local-effect-gate repair as a planning improvement and retains
  action-effect transfer as the active bottleneck.
- Cube RGB macro-transfer panel (2026-10-18, CUDA/EGL, 40 calibration and
  8 held-out roots): the first fixed panel compared reclose, open-loop,
  low-gain reclose, persistent contact, RGB effect adaptation, backtracking,
  and regression-stop arms. Persistent contact and backtracking each reached
  `2/8`, versus `1/8` for full-scale reclose; persistent contact reduced mean
  final distance by `0.0261` in paired roots but lost contact on two more
  roots. The effect-adaptive arm was `1/8`, so offline/local effect fitting
  did not yield a stable online gain. Artifact:
  `results/live_cube_rgb_macro_panel_20261018_cuda/panel.json`.
- Cube contact-probe replication (2026-10-18, CUDA/EGL, fresh seed,
  root-disjoint): a predeclared short RGB direction-probe was added before
  persistent low-gain pushing. Persistent was `2/8` with contact `5/8`;
  persistent+probe was `1/8` with contact `6/8`; paired final-distance
  reductions versus full-scale reclose were `0.0443` and `0.0431`,
  respectively. The probe improves contact support but not task success, so
  action-effect instability after contact remains the active boundary.
  Artifact:
  `results/live_cube_rgb_macro_panel_20261018_probe_replication/panel.json`;
  implementation:
  `experiments/run_live_cube_rgb_macro_panel.py`.
- Cube RGB-realized calibration target audit (2026-10-18, CUDA/EGL, fresh
  root-disjoint panel): the macro ranker was rerun with calibration targets
  obtained only from the realized RGB successor (goal-centroid minus final
  block-centroid), instead of privileged simulator task cost. Full-scale
  reclose, persistent, persistent+probe, and backtrack obtained respectively
  `0/8`, `0/8`, `0/8`, and `1/8`. The earlier privileged-target panel had
  persistent `2/8`; therefore its apparent gain is not evidence that RGB
  successor geometry solved task transfer. This closes the privileged-teacher
  shortcut as a causal explanation and leaves task-sufficient contact geometry
  as the active repair target. Artifact:
  `results/live_cube_rgb_macro_panel_20261018_rgb_goal/panel.json`;
  implementation:
  `experiments/run_live_cube_rgb_macro_ranker.py`
  (`--calibration-target rgb_goal`).
- Cube contact-aware RGB geometry repair (2026-10-18, CUDA/EGL): adding the
  visible effector centroid and effector-minus-block chart to the RGB ranker
  leaves the legacy feature family unchanged by default. On a fresh 8-root
  panel with RGB-realized calibration, contact-aware backtrack reached `2/8`
  versus `0/8` for full-scale reclose. The predeclared 16-root replication
  reached `2/16` versus `0/16`, contact `13/16` versus `10/16`, and paired
  final-distance reduction `+0.02035` (bootstrap 95% CI
  `[+0.00599,+0.03594]`); paired success delta was `+0.125` (CI
  `[0,+0.3125]`, 2 wins, 0 losses, 14 ties). This is the first repeated positive
  task consequence for a non-privileged RGB repair, but remains Cube-scoped
  and does not establish universal world-model transfer. Artifacts:
  `results/live_cube_rgb_macro_panel_20261018_contact_aware/panel.json` and
  `results/live_cube_rgb_macro_panel_20261018_contact_aware_replication16/panel.json`;
  implementation:
  `experiments/run_live_cube_rgb_macro_ranker.py` and
  `experiments/run_live_cube_rgb_macro_panel.py`.
- Cube horizon intervention (2026-10-18, CUDA/EGL, fresh root-disjoint
  16-root test set): with the contact-aware RGB features and non-privileged
  `rgb_goal` calibration held fixed, increasing the macro horizon to 8 was
  compared between the predeclared `fullscale_reclose` reference and
  `lowgain_backtrack`. Backtrack achieved `1/16` success versus `0/16`, and
  reduced paired final distance by `0.03793` (bootstrap 95% CI
  `[+0.00758,+0.07463]`; 10 wins, 6 losses). However, best-distance reduction
  was `0.00289` (CI `[-0.00813,+0.02053]`), contact observation was lower
  (`10/16` versus `14/16`), and paired success was only one win with no
  losses. Thus a longer horizon improves a local terminal-distance statistic
  but does not repair contact retention or stable task success; the active
  boundary remains within-contact action-effect transfer rather than a simple
  horizon shortage. Artifact:
  `results/live_cube_rgb_macro_panel_20261018_contact_aware_horizon8/panel.json`;
  implementation:
  `experiments/run_live_cube_rgb_macro_panel.py`.
- Cube privileged candidate-bank ceiling matched to the horizon-8 roots
  (2026-10-18, EGL MuJoCo evaluator): the fixed four-direction bank was
  reachable on `2/16` roots and the expanded eight-direction/low-gain bank was
  also reachable on `2/16`; expansion recovered `0` previously unreachable
  roots and reduced mean oracle distance only from `0.25070 m` to `0.24754 m`.
  The paired failure-cause audit is deliberately `unresolved`, because the
  online `lowgain_backtrack` arm uses RGB progress feedback, sign reversal,
  and gain changes that are not represented by the fixed open-loop candidate
  ceiling. Therefore this result diagnoses weak fixed-bank support but cannot
  be used to attribute the adaptive arm's failures entirely to reachability.
  Artifact:
  `results/cube_planner_ceiling_20261018_horizon8_fresh16/ceiling.json` and
  `results/cube_failure_cause_horizon8_fresh16/report.json`;
  implementation:
  `experiments/audit_cube_planner_ceiling.py` and
  `experiments/summarize_cube_failure_cause.py`.
- Cube adaptive action-protocol ceiling (2026-10-18, EGL MuJoCo evaluator,
  fresh root-disjoint 16-root panel): replaying the fixed online bank with the
  exact RGB progress-feedback/backtrack law and persistent contact gave
  `2/16` privileged reachability, the same rate as the open-loop fixed-bank
  ceiling (`2/16`), with mean oracle distance `0.24389 m` versus `0.25070 m`.
  The root-level decomposition is more informative than the aggregate:
  adaptive persistence recovered root `20263002` (task 3) that open-loop could
  not reach, but lost open-loop reachability on root `20263012`; the selected
  successful adaptive candidates did not require a gain backtrack. Thus the
  observed online backtrack success is explained primarily by a
  root-dependent persistent-contact/reclose protocol interaction, not by a
  universally useful feedback repair. Artifact:
  `results/cube_adaptive_protocol_ceiling_20261018_horizon8_fresh16/ceiling.json`;
  implementation:
  `experiments/audit_cube_adaptive_protocol_ceiling.py` and
  `experiments/audit_cube_planner_ceiling.py`.
- Cube mode-conditioned reclose intervention (implementation queued for CUDA
  validation): `lowgain_mode_reclose` retains persistent contact only when the
  previous cycle produces at least the predeclared RGB goal-error decrease
  `1e-3`; missing or insufficient RGB progress fails closed to reclose. The
  gate uses no simulator contact labels and records its per-cycle reason in
  `mode_reclose_trace`. It is a targeted repair for the root-dependent
  persistent/reclose interaction above, not yet a positive result: the formal
  RGB panel is blocked until CUDA is available. Implementation:
  `experiments/run_live_cube_rgb_macro_ranker.py` and
  `experiments/run_live_cube_rgb_macro_panel.py`.
- Cube adaptive action-protocol smoke ceiling (2026-10-18, CPU diagnostic
  because CUDA device nodes were unavailable): on two fresh roots, the exact
  RGB-backtrack protocol with privileged post-hoc candidate choice reached
  `0/2` and had mean final distance `0.17665 m`. This is stronger evidence
  that the current fixed bank/protocol has a reachability ceiling; it is not
  learned-model or online-policy evidence and is not pooled with the formal
  CUDA panel. Artifact:
  `results/cube_adaptive_protocol_ceiling_20261018_cpu_diagnostic/report.json`.
- Cube goal-conditioned support repair smoke (2026-10-18, CPU diagnostic):
  adding evaluator-only goal-relative push offsets reduced the one-root oracle
  distance from `0.20022 m` to `0.14715 m`, but reachability remained `0/1`.
  Thus directional support helps terminal distance without repairing the
  contact/transport interface; it is not evidence for a deployable RGB policy
  or learned-model transfer. Artifact:
  `results/cube_adaptive_goal_conditioned_ceiling_20261018_cpu_smoke/report.json`.
- Cube pick-place interface ceiling smoke (2026-10-18, CPU diagnostic):
  replacing planar push with an evaluator-only close/lift/transport/lower/
  release primitive still reached `0/1`; both the 8-direction and
  goal-relative variants had mean final distance `0.35479 m` and recorded no
  contact. This localizes the current failure one stage earlier: the RGB
  descent/approach prelude is not establishing reliable contact, so changing
  the learned ranker cannot fix this root. Artifact:
  `results/cube_pick_place_ceiling_20261018_cpu_smoke/report.json`.
- Cube approach-window intervention (2026-10-18, CPU diagnostic): extending
  the same RGB approach prelude from 16 to 32 steps raised observed contact
  to `0.9114` on the smoke root, but pick-place reachability remained `0/1`
  and final distance was `0.19740 m`. Thus approach duration was a real
  contact bottleneck, but solving contact alone does not solve transport or
  task closure. Artifact:
  `results/cube_pick_place_ceiling_20261018_cpu_approach32_smoke/report.json`.
- Cube pick-place transport-gain smoke (2026-10-18, CPU diagnostic): with the
  32-step approach and transport magnitude `0.5`, contact reached `1.0` and
  the best 8-direction pick-place distance improved to `0.15329 m`, but
  reachability remained `0/1`; the goal-relative variant was worse
  (`0.24807 m`). This rules out insufficient contact and points to an
  action-timing/transport-dynamics mismatch in the current primitive.
  Artifact:
  `results/cube_pick_place_ceiling_20261018_cpu_transport05_smoke/report.json`.
- Cube pick-place transport-duration smoke (2026-10-18, CPU diagnostic):
  increasing transport duration from 20 to 50 steps (magnitude `0.5`) reduced
  the best privileged distance further to `0.12936 m`, with contact `1.0`, but
  still yielded `0/1` reachability. The goal-relative variant remained
  unsuccessful (`0.19158 m`). Longer contact improves local displacement but
  does not close the task, so the remaining boundary is not merely horizon or
  contact duration. Artifact:
  `results/cube_pick_place_ceiling_20261018_cpu_transport50_smoke/report.json`.
- Cube table-push mechanism recheck (2026-10-18, CPU diagnostic, fresh
  2-root panel): the previously positive single-root RGB table-push protocol
  (105 transport steps, magnitude `0.1`, chart controller) had contact on both
  roots but success `0/2`, mean distance `0.10611 m`; the root-wise distances
  were `0.05604` and `0.15617 m`. The predeclared `approach-before-descend`
  intervention was worse (`0/2`, `0.17406 m`). This closes the apparent
  single-root repair as non-robust and identifies root-dependent approach /
  effect calibration as the remaining mechanism boundary. Artifacts:
  `results/cube_tablepush_repair_20261018_cpu_2roots/report.json` and
  `results/cube_tablepush_before_descend_20261018_cpu_2roots/report.json`.
- Cube root-local action-effect intervention (2026-10-18, protocol locked;
  CUDA validation pending): the fresh-root paired panel holds the roots,
  local RGB probes, action bank, approach/transport schedule, and goal
  threshold fixed, and compares `chart`, `effect`, and `hybrid` transport
  controllers. It reports paired action-effect rank, contact, final distance,
  success, and per-root deltas without selecting a controller from outcome
  roots. This directly tests whether the remaining root-dependent failure is
  an action-effect calibration problem. It is not yet a positive result.
  Implementation: `scripts/run_cube_action_effect_panel.sh`,
  `experiments/summarize_cube_action_effect_panel.py`; expected artifacts:
  `results/cube_action_effect_panel_20261018/`.
  The panel now requires `--lift-before-probes`: effect probes are collected
  after the object is lifted, so the fitted operator represents carried-object
  action effect rather than free-gripper/pre-contact motion. Pre-lift effect
  fitting is rejected fail-closed by the runner. A full-rank but weak,
  ill-conditioned, or high-residual effect fit is also rejected; the runtime
  records the fallback source instead of allowing a spurious action map to
  drive transport.
- Cube action-effect panel v1 (2026-10-18, CUDA/EGL, 8 fresh paired roots):
  chart, carried-object effect, and hybrid all had contact `1.0` and success
  `0/8`; mean final distances were `0.18716`, `0.19216`, and `0.19288 m`.
  This ruled out contact acquisition as the remaining cause, but the first
  implementation had a fallback-comparison contamination that was corrected
  before v2. Artifact: `results/cube_action_effect_panel_20261018/`.
- Cube action-effect panel v2 (2026-10-18, CUDA/EGL, same frozen roots):
  after making `effect` fall back to the identical RGB chart rather than a
  world chart, chart/effect/hybrid were still `0/8`, with mean distances
  `0.14727`, `0.16349`, and `0.15719 m`. This rejects the hypothesis that
  the prior negative result was only caused by fallback contamination.
  Artifact: `results/cube_action_effect_panel_20261018_v2/`.
- Cube online effect-validation panel v3 (2026-10-18, CUDA/EGL, same roots):
  rejecting an effect map after a failed first RGB displacement reduced the
  effect penalty to `-0.00624 m` paired improvement (effect mean `0.15351 m`)
  but did not produce success (`0/8`). The gate improves safety, not task
  completion. Artifact: `results/cube_action_effect_panel_20261018_v3/`.
- Cube greedy-effect control (2026-10-18, CUDA/EGL, same roots): replacing
  continuous inverse mapping with an eight-direction discrete effect choice
  still produced `0/8`, mean distance `0.15458 m`, paired improvement
  `-0.00731 m`. This rejects inverse-amplification as the sole explanation.
  The current evidence therefore sets a failure boundary: contact is
  observable and repeatable, but carried-object RGB action-effect transfer is
  not a reliable cross-root operator for this protocol. Artifact:
  `results/cube_action_effect_panel_20261018_v4/`.
- Protocol repair for Cube action-effect panel (2026-10-18): the earlier
  panel passed `--local-pixel-approach` but inherited the runner default
  `approach_controller=world`, so the local pixel approach was not actually
  used. The locked panel now explicitly passes `--approach-controller pixel`;
  v5 is a fresh-root rerun required before interpreting transport effects.
- Cube pixel-approach action-effect panel v5 (2026-10-18, CUDA/EGL, fresh
  8-root paired panel): with the corrected explicit pixel approach,
  chart/effect/hybrid all remained `0/8` and had mean final distance
  `0.17624 m`; effect was identical to chart and hybrid did not improve
  distance. Contact was `0.75`, `0.75`, and `0.875`, respectively. Together
  with v1--v4, this shows that the negative result is not explained by a
  single inverse solver, fallback contamination, pre-lift probes, or the
  world-versus-pixel approach default. The remaining boundary is unstable or
  insufficiently observable carried-object RGB action-effect transfer under
  these roots; no controller success claim is made. Artifact:
  `results/cube_action_effect_panel_20261018_v5/`.
- Probe-scale repair for Cube action-effect panel (2026-10-18): v1--v5
  calibrated effect at magnitude `0.15` but executed transport at `0.5`.
  Because the Cube contact/friction interface may be amplitude-thresholded,
  this is an invalid local-linear extrapolation. The locked v6 protocol now
  matches `probe_magnitude=transport_magnitude=0.5` before evaluating effect
  transfer.
- Cube matched-scale effect panel v6 (2026-10-18, CUDA/EGL, fresh 8-root
  paired panel): matching probe and transport magnitudes raised contact to
  `1.0` for all arms, but chart/effect/hybrid remained `0/8`; mean distances
  were `0.18031`, `0.18060`, and `0.18060 m`, with paired effect delta
  `-0.00029 m`. Thus amplitude-threshold mismatch is not sufficient to explain
  the negative result. The surviving boundary is that RGB carried-object
  effect information is not independently transferable to task closure under
  this protocol. Artifact: `results/cube_action_effect_panel_20261018_v6/`.
- Cube calibrated RGB approach and verified local-effect panel (2026-10-19,
  CUDA/EGL, fresh 8-root paired panel): the prior approach implementation
  conflated image displacement with action displacement and approached before
  descending. The repaired protocol uses a calibration-only pixel/action
  Jacobian, shared by all arms, with `15 descent -> 10 RGB approach -> 5
  close`. Verified local RGB effect improved over the paired static carrier
  on `4/8` roots, tied `4/8`, lost `0/8`, with mean final-distance delta
  `-0.05925 m`. The frozen-WM agreement gate was identical and recorded
  disagreement fallbacks; rank-normalized fusion was weaker at `-0.03930 m`.
  RGB geometry alone remained non-robust (`3/8` losses, `+0.01035 m`). This
  repairs a control-interface bottleneck and supports local verified transfer;
  it does not establish independent frozen-WM causal dynamics or universal
  planning. Artifact:
  `results/cube_rgb_calibrated_approach_20261019_fresh8/closed_loop.json`.
- TwoRoom observed-effect gate and inverse-policy audit (2026-10-19, CUDA,
  root-disjoint): the 32-root gate-only panel was neutral (`0/0/32` paired
  wins/losses/ties). Replacing the frozen planner action with a locally
  inverted RGB effect map was harmful (`0/10/22`), and adding rolling
  two-step hold-out verification reduced but did not remove the harm
  (`0/3/29`). The cause is task-level door/wall waypoint semantics not
  represented by instantaneous marker displacement. The formal TwoRoom
  controller therefore stays fail-closed: use the observed map only to veto a
  conflicting model action, then use the reactive fallback; never overwrite a
  planner action that already passes the gate. Artifacts:
  `results/tworoom_verified_effect_20261019_fresh32/`,
  `results/tworoom_verified_effect_policy_20261019_fresh32/`, and
  `results/tworoom_verified_effect_policy_holdout_20261019_fresh32/`.
- PointMaze held-out support/risk-gate consequence (2026-10-19, CUDA,
  root-disjoint): calibration/readout roots were `1800--1849` and the fresh
  test roots were `1870--1889`. Raw `wm_rgb` achieved `2/20` success and mean
  final distance `1.0186`; visual-dynamics achieved `14/20`, `0.2813 m`; the
  RGB support/dynamics-gated frozen-WM arm achieved `15/20`, `0.2927 m`; the
  ungated effect adapter achieved `11/20`; HTSC consistency achieved `14/20`,
  `0.2810 m`. The gate rejected `78.7%` of WM decisions and consistency
  rejected `93.3%`. This is a positive selective-repair result for raw WM
  failure, not proof of independent frozen-WM causal dynamics: most decisions
  used the calibration-only visual-dynamics fallback. Artifact:
  `results/pointmaze_verified_effect_20261019_test1870_1889/report.json`.
- PushT active observed-effect orientation line search (2026-10-19--22,
  CUDA, privileged mechanism audit): the first large-force protocol preserved
  boundary contact on `16/16` roots but drove mean XY error to `181.61 px` and
  achieved only `3/16` orientation successes. Reducing the normal/tangent
  pulse to `(0.02, 0.04)` reduced mean XY error to `21.17 px` but left only
  `2/16` orientation successes; a pure tangent pulse `(0.00, 0.08)` kept mean
  XY error at `3.57 px` but achieved `0/16` orientation successes. The
  balanced `(0.03, 0.12)` panel also achieved `0/16`. This is a decisive
  mechanism boundary: contact retention alone is insufficient; orientation
  requires normal force that creates translation, so a valid controller must
  add task-level XY restoration/phase planning rather than only improve sign
  estimation. Artifacts:
  `results/pusht_verified_orientation_linesearch_20261019_cuda/`,
  `results/pusht_verified_orientation_linesearch_20261020_small_cuda/`,
  `results/pusht_verified_orientation_linesearch_20261021_tangent_cuda/`,
  `results/pusht_verified_orientation_linesearch_20261022_balanced_cuda/`.
  Adding an observed XY-restoration phase to the balanced protocol did not
  close the gap: the fresh 16-root CUDA panel reached `2/16` orientation
  successes, mean final angle error `1.6164 rad`, mean XY error `28.15 px`,
  and preserved the boundary contact on `16/16`. The restoration loop only
  preserved the declared XY budget on `4/16` roots, so it is not promoted to
  the formal controller. Artifact:
  `results/pusht_verified_orientation_linesearch_20261023_restore_cuda/`.
  An observed-effect XY restoration variant lowered mean XY error further to
  `9.90 px` but lost contact on `7/16` roots, confirming that unconstrained
  residual minimization selects separation actions. Enforcing an inward
  contact lease restored contact to `16/16`, but mean XY error increased to
  `61.36 px` and orientation success remained `1/16`. Artifact:
  `results/pusht_verified_orientation_linesearch_20261024_observed_restore_cuda/`
  and `results/pusht_verified_orientation_linesearch_20261025_contact_lease_cuda/`.
- PushT matched orientation response ceiling (2026-10-25, CUDA, 16
  edge/fraction roots): equal-norm opposite-moment pairs were valid on
  `16/16` roots, but signed response consistency was only `9/16`; the mean
  paired angular separation was `0.00995 rad` and common drift `0.00558 rad`.
  Thus a local paired probe contains a small orientation signal, but its sign
  is not uniformly transferable across contact strata. Artifact:
  `results/pusht_orientation_ceiling_20261025_cuda/orientation_response_audit.json`.
- PushT structured temporal torque gate v2 (2026-10-21, **protocol queued;
  no outcome claim**): the preceding independent RGB-history confirmation
  failed after a 3,848-dimensional linear decoder interpolated 48 train roots
  but transferred poorly (`0.458` calibration balanced accuracy).  The fresh
  successor uses new root-seed bands, 128/64/128 balanced train/calibration/
  outcome roots per class, a fixed 4x4 RGB motion chart, train-only PCA to 24
  directions, and a calibration-only confidence abstention gate.  It refuses
  output reuse and must fail closed if calibration lacks sufficient supported
  accuracy.  Its launcher is GPU-only; no CPU proxy will be reported as an
  efficacy result.  Code:
  `le-wm-main/experiments/run_pusht_structured_temporal_torque_gate.py` and
  `le-wm-main/experiments/run_pusht_structured_temporal_torque_gate_cuda.sh`.
  This is only a selective offline torque-preference test, not an action
  execution, planning, pose-closure, or world-model claim.
- PushT literature-derived three-arm matched benchmark (2026-10-19, CUDA,
  boundary-contact, 16 train and 16 held-out roots): the shared candidate
  panel reached mode accuracy `0.9978` and boundary-contact loss `0/16` for
  the frozen state-chart predictor, state-only fallback, and adaptive local
  effect arms. The predictor arm had pose success `1/16`, final XY `55.48 px`,
  final angle `1.101 rad`, and predicted/observed effect error `0.222`; the
  adaptive local-effect arm had `4/16`, `38.86 px`, `1.203 rad`, and `0.174`,
  respectively. The safe fallback had `2/16` pose success and `39.18 px`
  final XY. Thus local probing reduced effect residual and improved pose count
  relative to the predictor arm in this privileged panel, but the predictor
  did not beat the fallback on pose/XY and the paired confidence intervals
  remain wide. This is selective local-repair / permission-to-act evidence,
  not frozen RGB/LeWM independent gain or universal transfer. Artifact:
  `le-wm-main/results/pusht_literature_three_arm_cuda_20261019/`;
  machine-readable decision:
  `three_arm_summary.json`.
- PushT phase-level contact-repair intervention (2026-10-19, CUDA,
  independent 12-root panel): enabling ordered paired orientation contrast,
  reverse phase gating, XY shielding, and contact relocation produced `4/12`
  pose successes, `18.26 px` final XY error, and `0.860 rad` final angle
  error. The matched state-only fallback produced `3/12`, `56.35 px`, and
  `0.706 rad`. The improvement is therefore concentrated in translation and
  strict pose count, while angle error itself is not improved. This identifies
  the remaining bottleneck as phase-level task utility/translation coupling,
  not simply sign estimation. The intervention is retained as a conditional
  privileged repair candidate; it is not evidence for frozen RGB/LeWM gain.
  Artifact: `le-wm-main/results/pusht_contact_relocation_cuda_20261019/`;
  paired summary: `phase_repair_summary.json`.
- PushT phase-shield and action-support follow-up (2026-10-19, CUDA):
  tightening the XY shield from `2 px` to `0 px` on an independent high-angle
  8-root panel did not restore pose closure (`0/8`); adding three declared
  normal magnitudes (`0.02/0.04/0.08`) reduced held-out ranking regret to
  `0.0615` but also remained `0/8` pose success. Extending the same roots
  from 16 to 36 closure steps still remained `0/8`. The high-angle roots had
  mean initial angle error `1.87 rad`, versus `0.95 rad` in the earlier
  positive-count panel. This separates three effects: support coverage can
  improve one-step ranking, horizon can reduce XY error, but neither resolves
  within-contact orientation identifiability at high initial angle. Artifacts:
  `le-wm-main/results/pusht_phase_shield0_cuda_20261019/`,
  `le-wm-main/results/pusht_multinormal_support_cuda_20261019/`, and
  `le-wm-main/results/pusht_multinormal_extended_cuda_20261019/`.
- PushT high-angle support-coverage intervention (2026-10-19, CUDA): using
  the same difficult 8 held-out roots (mean initial angle error `1.870 rad`)
  while increasing training roots from `8` to `32` raised mode accuracy from
  `0.919` to `0.969` and reduced predicted/observed effect error to `0.095`
  for the state-chart arm. Nevertheless, every closed-loop arm remained
  `0/8` pose success; the best paired phase macro reached `17.41 px` XY but
  retained `1.617 rad` angle error. This rules out small training support as
  the sole cause and narrows the boundary to high-angle within-contact
  controllability/action-effect identifiability. Artifact:
  `le-wm-main/results/pusht_highangle_train32_cuda_20261019/`.
- PushT dense-bank relocation ablation (2026-10-19, CUDA, high-angle 8-root matched panel): relocation-enabled focused three-arm learned run had `0/8` pose success for predictor/fallback/local repair, with final XY `61.01/57.13/61.50` px and local-repair angle `1.959` rad. The predeclared no-relocation repeat remained `0/8` for all arms, but reduced final XY to `30.69/28.73/25.53` px and local-repair angle to `1.641` rad. Local observed-effect residual remained lower (`0.174`) than predictor (`0.210`) without producing pose closure; wrong-sign events remained `4.25/root`. This isolates relocation drift as a translation confound, not the sole cause: within-contact action-effect identifiability remains unresolved. Artifacts: `le-wm-main/results/pusht_highangle_densebank_focused8_cuda_20261019/` and `le-wm-main/results/pusht_highangle_densebank_norelocation_focused8_cuda_20261019/`.
- PushT contact-lease and active-excitation follow-up (2026-10-19, CUDA, same high-angle 8-root panel): explicit contact lease produced `0/8` pose, `22.96 px` local-repair final XY, `1.840 rad` final angle, and residual `0.154`. Adding maximum-moment escalation after wrong-sign reduced wrong-sign events from about `5.00` to `1.625/root`, but worsened local-repair final XY to `44.60 px` and remained `0/8` pose. A privileged horizon-2 beam ceiling with 18-step budget reached `2/8` pose, `5.81 px` XY, and `1.245 rad` angle; it is an upper-bound diagnostic with a different budget, not learned or frozen-LeWM evidence. Artifacts: `le-wm-main/results/pusht_highangle_densebank_contactlease_focused8_cuda_20261019/`, `le-wm-main/results/pusht_highangle_densebank_contactlease_escalate_focused8_cuda_20261019/`, and `le-wm-main/results/pusht_highangle_successor_beam_cuda_20261019/`.
- PushT observed local-Jacobian follow-up (2026-10-19, CUDA, no-relocation high-angle 8-root panel): fail-closed local action-effect Jacobian reduced local-repair predicted/observed residual to `0.138` and final XY to `14.19 px`, but final angle remained `1.750 rad` and strict pose remained `0/8`; online Jacobian exploitation rate was `0` because deployed roots lacked same-region paired rank-2 support. Artifact: `le-wm-main/results/pusht_highangle_densebank_localjac_focused8_cuda_20261019/`.
- PushT terminal observed-effect control-flow repair (2026-10-19, CUDA, same no-relocation 8-root panel): high-angle roots had been skipping adaptive observed-effect during `terminal_handoff`; enabling `--terminal-observed-effect` raised paired probe rate to `0.930` and reduced residual to `0.108`, with final XY `11.79 px`, but online Jacobian exploitation remained `0`, final angle `1.844 rad`, and pose remained `0/8`. This fixes an implementation omission without closing the within-contact identifiability boundary. Artifact: `le-wm-main/results/pusht_highangle_densebank_terminal_localjac_focused8_cuda_20261019/`.
- PushT stiction probe-scale audit (2026-10-19, CUDA, same no-relocation 8-root panel): `breakaway-paired-probe` increased probe strength but produced local-repair residual `0.357`, XY `33.87 px`, angle `1.783 rad`, and pose `0/8`; lowering informative threshold `0.005 -> 0.001` retained residual `0.108`, XY `11.79 px`, angle `1.844 rad`, Jacobian exploitation `0`, and pose `0/8`. Strong probes leave the local chart while weak probes lack stable task-direction information. Artifacts: `le-wm-main/results/pusht_highangle_densebank_breakawaypair_focused8_cuda_20261019/` and `le-wm-main/results/pusht_highangle_densebank_terminal_localjac_threshold001_focused8_cuda_20261019/`.
- PushT extended action-support matched audit (2026-10-19, CUDA, same high-angle 8-root manifest): privileged extended-bank successor oracle reached `4/8` pose, `8.53 px` XY, and `1.072 rad` angle; the matched learned panel remained `0/8` pose, with predictor angle `1.657 rad`, local-repair angle `1.844 rad`, and local residual `0.077`. This rules out action-bank size as the sole cause and isolates sequence-level contact-conditioned ranking failure. Artifacts: `le-wm-main/results/pusht_highangle_successor_oracle_extended_cuda_20261019/` and `le-wm-main/results/pusht_highangle_extendedbank_terminal_focused8_cuda_20261019/`.
- PushT literature-derived train-root-only sequence-ranking intervention (2026-10-19, CUDA, boundary-contact, matched 8/8 panel): a joint two-pulse ExtraTrees preference model was fitted only on training-root outcomes; deployment re-observed after the first pulse before scoring the second. The sequence arm remained `0/8` pose, `37.22 px` final XY, and `1.900 rad` final angle, worse than frozen predictor (`0/8`, `11.66 px`, `1.815 rad`) and adaptive local repair (`0/8`, `11.16 px`, `1.817 rad`). The phase-repair arm reached `1/8` pose and `1.510 rad` angle. This is a negative but diagnostic result: re-observation alone does not repair out-of-support sequence preferences; the joint ranker trained on privileged state transitions is not a frozen RGB/LeWM transfer certificate. Artifact: `le-wm-main/results/pusht_highangle_densebank_sequence_ranker_focused8_cuda_20261019_v2/`.
- PushT sequence-ranker context-matching repair (2026-10-19, CUDA, same boundary-contact 8/8 panel): macro-ranker training now uses the frozen train-only predictor's imagined first state for first-pulse candidate generation, matching deployment; real first successors remain labels only. This reduced sequence-arm angle from `1.900` to `1.630 rad` and XY from `37.22` to `33.48 px`, but strict pose remained `0/8`. A sequence-first/phase-second hybrid further reached `1.561 rad` and `27.43 px`, still `0/8`, while the fixed phase arm remained `1/8`. Therefore train/deploy context shift is real but not sufficient; residual failure is sequence-risk calibration/action-effect transfer. Artifacts: `le-wm-main/results/pusht_highangle_densebank_sequence_ranker_contextmatched_focused8_cuda_20261019/` and `le-wm-main/results/pusht_highangle_densebank_sequence_phase_hybrid_contextmatched_focused8_cuda_20261019/`.
- PushT paired/unpaired local-chart diagnostic (2026-10-19, CUDA, same high-angle 8/8 panel): strict same-region +/- paired probes forced identification-probe rate to `1.0` but online Jacobian exploitation remained `0` and pose `0/8`; allowing same-region rank-2 unpaired effects did not change exploitation or pose. This rules out strict sign pairing alone as the sole failure: observed probe effects do not form a reusable action-effect chart under the current high-angle contact transitions. Artifacts: `le-wm-main/results/pusht_highangle_densebank_paired_localjac_focused8_cuda_20261019/` and `le-wm-main/results/pusht_highangle_densebank_unpaired_localjac_focused8_cuda_20261019/`.
- Cube phase-entry repair (2026-10-19, CUDA, fresh 16-root paired panel): the original verified-local protocol changed actions but produced 13/16 tied outcomes and mean final-distance delta `-0.00423 m`, identifying ineffective pre-contact phase entry as a bottleneck. A fixed RGB-only approach phase was added before the unchanged descent/closure prelude, with 40 calibration roots and 16 new test roots. Verified RGB local effect achieved `9/16` wins, `0` losses, `7` ties, mean delta `-0.03053 m`, bootstrap 95% CI `[-0.05193,-0.01299]`, exact sign-test `p=0.00390625`; fusion achieved `-0.03019 m` with the same win/loss pattern. Temporal-only and support-only controls were non-positive. This is a scoped cross-environment repair consequence for phase-entry plus verified-local action effect, not universal GDD-WM or causal latent-dynamics evidence. Artifacts: `le-wm-main/results/gdd_wm_cube_verified_local_approach_20261019_cuda_fresh16/` and `le-wm-main/results/gdd_wm_cube_verified_local_approach_20261019_cuda_fresh16/aggregate.json`.
- Cube RGB-successor-only preference-transfer panel (2026-10-19, CUDA, 40 calibration and 16 fresh test roots): `rgb_goal` calibration used realized RGB successor goal error without privileged task labels. Low-gain effect-adaptive versus fullscale reclose improved final distance by `0.03441 m` on average (intervention-minus-baseline delta `-0.03441`, bootstrap CI `[-0.06709,-0.00590]`), with 10/16 root improvements and 6/16 degradations; exact two-sided sign test `p=0.4545`. Success was `2/16` versus `0/16`, not sufficient for a success-rate claim. Effect-stop was weaker (`-0.01708 m`, CI `[-0.04610,0.00487]`). This is evidence for RGB realized-effect preference plus low-gain contact adaptation, not frozen LeWM/D-JEPA teacher validation or universal transfer. Artifacts: `le-wm-main/results/gdd_wm_cube_rgb_goal_teacher_20261019_cuda_fresh16/panel.json` and `.../paired_summary.json`; adjudicator: `le-wm-main/experiments/summarize_cube_rgb_macro_panel.py`.
- Cube RGB approach-horizon replication (2026-10-19, CUDA, fresh 16-root panel): increasing the fixed RGB approach from 10 to 20 steps raised adaptive contact observation to `13/16` (`0.8125`) and improved task-4 contact from `1/3` to `2/3`, while task 5 remained `1/3`. Adaptive final-distance delta was `-0.03441 m`, bootstrap CI `[-0.05458,-0.01598]`, 12/16 wins and 4/16 losses, exact sign-test `p=0.0768`; endpoint success was only `1/16`. This confirms partial phase-entry repair and leaves a task-specific contact/geometry boundary. Artifacts: `le-wm-main/results/gdd_wm_cube_rgb_goal_teacher_approach20_20261019_cuda_fresh16/`.
- Cube task-5 GDD-WM teacher-to-effect closed-loop repair (2026-10-19, CUDA, fresh 16-root panel): a calibration-only RGB approach sweep selected 20 approach steps at gain 1.0, yielding front contact-proxy support on `15/16` calibration roots. On 16 new test roots, the frozen temporal ranker alone had mean final-distance delta `+0.00896 m` versus the static carrier (bootstrap CI `[-0.00482,0.02252]`, 5 wins/9 losses/2 ties), so the teacher was not released as a standalone controller. Adding a fail-closed RGB observed-effect map after the same frozen candidate proposal produced `7` improvements, `0` degradations, and `9` ties, mean delta `-0.01742 m`, bootstrap CI `[-0.03025,-0.00601]`, exact sign-test `p=0.015625`. This validates a scoped recovery chain—frozen proposal -> realized RGB effect verification -> fallback—at distance level, not D-JEPA teacher correctness, task-success improvement, or universal transfer. Artifacts: `le-wm-main/results/gdd_wm_cube_task5_teacher_closed_loop_20261019_cuda_fresh16/closed_loop.json` and `aggregate.json`; protocol and claim boundary: `docs/GDD_WM_PROGRESS_20261019.md`.
- PointMaze GDD projection verified-local replication (2026-10-19, CUDA, root-disjoint 20-root panel): the new runtime arm first proposed with the frozen D2 projection, then used four waypoint-aligned RGB probes, an observed affine action-effect chart, and a fixed `0.10` local-versus-visual consistency gate. The initial dispatch/coordinate-contract failure was repaired and excluded. Final results were visual dynamics `20/20`, frozen GDD projection `3/20` with mean final distance `2.300`, and GDD plus verified-local `3/20` with mean final distance `1.347` and mean progress `0.322` versus `-0.631` for standalone GDD. This materially repairs distance failure relative to the frozen student but does not beat the saturated RGB dynamics baseline; the probe budget and residual task-level mismatch remain boundaries. Artifact: `le-wm-main/results/gdd_wm_pointmaze_gdd_verified_local_20261019_cuda_v8_waypoint_consistency.json`.
- PointMaze GDD action-support gate follow-up (2026-10-19, CUDA, same 20-root paired panel): because the frozen paired action family reaches magnitude `1.0` while the local chart is identified with fixed `0.25` probes, release was restricted to candidates within `1.5x` probe support and fallback actions were excluded from local-chart updates. The verified-local arm changed from `3/20` to `4/20` success and mean final distance `1.347 -> 1.343`; paired distance change was only `-0.00358 m` (`10/20` improvements, `10/20` degradations). Standalone GDD remained `3/20`, `2.300 m`; visual dynamics remained `20/20`, `0.143 m`. This repairs an extrapolation mechanism but does not establish independent planning gain. Artifact: `le-wm-main/results/gdd_wm_pointmaze_gdd_verified_local_20261019_cuda_v9_support_gate.json`.
- PointMaze disagreement-triggered verified-local follow-up (2026-10-19, CUDA, fresh root-disjoint 20-root panel): the new `gdd_wm_rgb_disagreement_verified_local` arm performs the four fixed RGB probes only when the frozen GDD and calibration-only visual-dynamics proposals have cosine below the predeclared `0.80` threshold. It triggered on `15/20` roots, executed a mean `3.0` probes, and reached `11/20` success with mean final distance `0.8136`, versus frozen GDD `6/20`, `1.7400`; visual dynamics remained `20/20`, `0.1631`. Paired distance improved on `16/20` roots versus frozen GDD, but the arm still lost to visual dynamics (`9` baseline-only successes, `0` intervention-only). Skipped probe records are excluded from the executed budget counter. This repairs the GDD failure and isolates the remaining ceiling boundary; it is not independent GDD gain or universal transfer evidence. Artifact: `le-wm-main/results/gdd_wm_pointmaze_gdd_disagreement_verified_local_20261019_cuda_fresh20.json`.
- PointMaze post-action verification and RGB path-gate follow-up (2026-10-19, CUDA, same root-disjoint 20-root panel): after each released local action, the RGB successor effect is checked against the local prediction with fixed tolerance `0.25`; failed charts are quarantined and failed transitions are not added back to the chart. A calibration-only RGB occupancy segment gate rejects predicted endpoints outside free-cell support. The complete arm reached `13/20` success and mean final distance `0.6906`, versus frozen GDD `6/20`, `1.7400` and visual dynamics `20/20`, `0.1631`; post-action verification passed `98.35%` of checks, chart quarantine occurred on `40%` of roots, and the path gate rejected all local candidates on `39` calls. Paired distance improved on `15/20` roots versus frozen GDD (`8` intervention-only versus `1` baseline-only success transitions), but there was still no gain over the saturated visual baseline. This validates a stronger fail-closed recovery chain, not a universal teacher or transfer certificate. Artifact: `le-wm-main/results/gdd_wm_pointmaze_gdd_pathgated_20261019_cuda_fresh20_correct.json`.
- PointMaze route-phase quarantine follow-up (2026-10-19, CUDA, same root-disjoint 20-root panel): a waypoint change now quarantines the episode-local effect chart, executes that transition with the RGB visual-dynamics fallback, and prevents stale local releases in the new phase. The phase-gated arm reached `15/20` success and mean final distance `0.4863`, versus frozen GDD `6/20`, `1.7400` and visual dynamics `20/20`, `0.1631`; distance improved on `16/20` roots versus frozen GDD, with `10` intervention-only and `1` baseline-only success transitions. The gate was invoked on `25` transition blocks. This is the strongest scoped repair and supports phase-conditional action-effect transfer, but it remains below the saturated visual baseline and is not universal teacher evidence. Artifact: `le-wm-main/results/gdd_wm_pointmaze_gdd_phasegated_20261019_cuda_fresh20.json`.
