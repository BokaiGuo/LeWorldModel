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
| `le-wm-main/experiments/run_causal_visual_pusht_world_model.py` | intervention-supervised RGB-only visual world model with contact-frame/torque pilot and online planning audit | goal-pose label repair + CUDA pilots completed; goal readout repaired, current-pose/within-contact orientation and online-planning gates remain failed |

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
