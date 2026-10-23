# PointMaze V24: Stratum-calibrated repair on fresh roots

Status: `completed on host CUDA; high-motion transfer is positive, while the
direct-zero-motion stratum retains a coverage shortfall`.

## Question

V23 showed that a single global epsilon-regret envelope mixed two observable
regimes with different error scales.  V24 tests, on a new root-disjoint
panel, whether a predeclared route and separate rootwise calibrations can
preserve useful repair where the visible history contains motion without
claiming that the same repair applies to a direct, zero-motion regime.

## Frozen protocol

- The route is fixed before outcome scoring: visible motion `>= 0.02` uses
  recurrent repair; visible motion `< 0.02` uses the action-only fallback.
- Each observable stratum has its own fixed-bank, rootwise contrast and risk
  calibration.  The candidate bank, regret budget (`epsilon = 0.12`), model,
  and route were fixed before outcome labels were opened.
- Train, calibration, and outcome panels contain 256 mutually root-disjoint
  roots each.  The held 16-action bank is supported in training; gate hashes
  are bound to the subsequently created outcome labels.
- This is an offline fixed-bank C3 test.  It does not establish conditional
  safety, online MPC, contact transfer, or a universal world-model property.

## Fresh-root outcome audit

| Observable stratum | Released / roots | Outcome coverage | Paired regret gain vs. matched action-only fallback | 95% bootstrap interval | Success gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| visible high motion | 121 / 128 | 0.90625 | +0.17589 | [+0.14633, +0.20577] | +0.45313 |
| direct zero motion | 20 / 128 | 0.87500 | 0.00000 | [0.00000, 0.00000] | 0.00000 |
| pooled descriptive total | 141 / 256 | 0.89063 | +0.08795 | [+0.07006, +0.10628] | +0.22656 |

All covered-root regret bounds held.  The high-motion stratum therefore
provides a positive, independently evaluated fixed-bank transfer result under
the frozen route.  The zero-motion stratum does not validate a 90% marginal
coverage claim: its Wilson 95% interval is `[0.80657, 0.92157]`, and its
observed point coverage is below 0.90.  Pooled coverage must not be reported
as a global calibrated-control result.

## Integrity and claim boundary

The post-outcome audit verified root disjointness, predecessor root/action
isolation, training support for every held action, and exact sealed-gate to
outcome hash binding.  Its diagnosis is:

`high_motion_repair_transfers_with_direct_stratum_residual_coverage_shortfall`.

V24 outcomes are opened and immutable.  They cannot be used to modify the
motion threshold, epsilon, model width, or calibration rule.  A successor must
use new roots and retain the high-motion route unchanged; it may test a
predeclared stricter abstention/probe policy or a separately calibrated,
observable direct-zero-motion model.

## Execution preservation and audit

The original run was launched through the host-only wrapper (not the Codex
sandbox).  That wrapper runs the CUDA preflight and fails closed rather than
falling back to CPU.  Its persistent host log records the RTX 5060 Ti,
CUDA 13.0, and a successful CUDA allocation before the outcome run.

Because the outcome labels are immutable, invoking the wrapper again against
the existing result directory correctly terminates with
`FileExistsError` rather than reopening V24.  The safe verification command
is therefore:

```bash
.venv/bin/python -m experiments.decision_benchmark.audit_pointmaze_c3_stratified_repair_v24
```

Artifacts are retained in
`results/pointmaze_c3_stratified_repair_v24_20261023_cuda/`.  A computational
replication must create a new protocol version with fresh roots and a new
result directory.
