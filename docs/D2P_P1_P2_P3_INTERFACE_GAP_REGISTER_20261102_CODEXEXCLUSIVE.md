# Planning Compiler P1/P2/P3 interface gap register

Status: exclusive interface preparation only.  
Date: 2026-11-02.  
Scope: new contract and CPU tests; no model execution, GPU execution, historical-result rewrite, or gate change.

## Boundary

The current repository has useful compiler, feedback, composition-witness, and
resource-ledger components. Their tests establish local invariants and fixture
replay. They do not by themselves establish a physical result, an independent
outcome, equal total compute, or scientific stage admission.

The dependency remains:

`P1 same-root mechanism + independent outcome -> P2 searched composition -> P3 dynamic scheduling`.

Therefore this register records interfaces that can be implemented now, while
keeping all admission fields false.

## Current implementation map and exact gaps

| Stage | Existing assets | What they establish | Exact missing item |
|---|---|---|---|
| P1 | `experiments/spectral_reachability/d2p_local_compiler.py`, `d2p_feedback_controller.py`, `d2p_stage_admission.py` and their tests | Local geometry, bounded action witnesses, observed-context handoff, and fail-closed stage helpers | A frozen identity-bound receipt joining non-greedy prefix/reconfiguration, own-state capability change, and independent closed-loop outcome on the same root |
| P1 | `d2p_planning_compute_ledger.py` and existing P0/P1 audits | Explicit resource vectors and root/method binding | A single pre-outcome task/source manifest that binds outcome rows and strong-baseline controls; ledger binding alone is insufficient |
| P2 | `d2p_composition_witness_v2.py` | Given edges can be replayed and concatenated with exact context/action checks | Search-and-select contract; per-edge predicted postcondition, observed context, action witness, replay receipt, post-observation recompile, and retained failed/partial branches |
| P2 | `d2p_p2_comparison_contract.py` and contextual-composition tests | Some comparison/accounting guards and fixture orchestration | Same candidate library, same feedback, same budget, plus explicit no-composition and flat-search controls in one outcome receipt |
| P3 | `d2p_p3_paired_accounting.py`, `d2p_planning_compute_ledger.py`, and scheduler tests | Resource accounting, fixed paired plans, no-refund accounting helpers | A pre-outcome schedule ABI mapping declared dynamics/geometry signals to horizon/depth/allocation, with actual allocation and unused budget in the receipt |
| P3 | Existing dynamic/fixed schedule fixtures | Formula and bookkeeping behavior under fixtures | Real matched feedback comparison, allocation-before-outcome evidence, signal/noise sensitivity, and complete success/cost curves |

The new module
`experiments/spectral_reachability/d2p_p1_p2_p3_interface_contract_20261102_codexexclusive.py`
implements only these receipt-level checks. It does not promote any existing
fixture or historical report.

## Contract schemas

### `d2p_p1_admission_contract.v1`

Required:

- immutable identity: `root_id`, `task_id`, `protocol_id`, `frozen_model_id`,
  checkpoint/task-manifest/protocol/source SHA256 values;
- `outcome_sealed_before_selection=true` and
  `source_disjoint_outcome=true`;
- same-root evidence flags for `non_greedy_prefix`,
  `own_state_capability`, and `closed_loop_outcome`;
- nonempty outcome rows containing the bound root;
- a named strong-baseline matrix.

Forbidden for this preparation contract:

- a positive `P1_admission`;
- replacing source identity with a row index;
- treating a calibration or fixture row as an independent outcome.

### `d2p_p2_composition_search_contract.v1`

Required:

- sealed candidate library before outcome;
- explicit search execution, not only replay of supplied edges;
- per-edge input context, predicted postcondition, observed context,
  action witness, replay receipt, and post-observation recompile;
- same budget/feedback and explicit `no_composition` plus `flat_search`
  comparators;
- failure and partial branches retained.

The existing composition witness remains useful but is a verifier for supplied
edges; it is not this search contract.

### `d2p_p3_dynamic_schedule_contract.v1`

Required:

- schedule rule sealed before outcome;
- declared signal source (`model_dynamics` or `declared_geometry`) with
  horizon/depth/predict-budget fields;
- allocation before outcome under a common feedback and budget cap;
- planned, actual, and arithmetic unused-budget fields;
- no refund of failed work and no encoding unknown work as zero;
- fixed-horizon, fixed-budget, and randomized-schedule controls.

This contract does not claim that a signal is useful, safe, optimal, or cheaper.
Those require real matched outcomes after P1 and the necessary P2 evidence.

## Testing and promotion boundary

The exclusive test file checks missing identity, missing action witness,
replay-only P2 edges, absent recompile, censored failures, outcome-dependent P3
allocation, hidden/zeroed work, arithmetic budget consistency, and positive
admission flags. These are interface tests, not physical experiments.

The following remain open and must not be inferred from this patch:

1. P1 eligible task-family and independent root/source-disjoint outcome.
2. P1 same-root non-greedy mechanism plus own-state capability improvement.
3. P2 real search and incremental success/cost over no-composition and flat search.
4. P3 real dynamic schedule under the same feedback and budget, including
   signal construction cost and noise sensitivity.
5. Any GPU execution or new historical-result interpretation.

All future result producers should write to a new exclusive path and preserve
`P1_admission=false`, `P2_admission=false`, and `P3_admission=false` until the
existing scientific gates are independently satisfied.
