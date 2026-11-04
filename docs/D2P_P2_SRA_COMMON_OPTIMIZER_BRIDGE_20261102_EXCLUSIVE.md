# Known-root SRA and common-optimizer development bridge

The existing `p2_no_composition_sra_20261027_v2` implements joint-H2
no-composition SRA: zero/DLS/spectral bank of 40, complete search and replay.
The existing `codex_sra_online_run_bank_20261030` already connects that bank
to a current context and validates its selected action; its strict consumer
also requires the outer journal and checkpoint/source proofs. Existing
four-arm P2 contracts are metadata validators rather than an executor.
`codex_four_arm_leaf_suffix_20261030` provides a different P1-oriented
gradient callback, not a complete common-budget P2 paired runner.

The new `p2_sra_common_optimizer_bridge_20261102_exclusive` adds:

- extraction of the full ten-action selected SRA winner, using the existing
  `audit_candidates` checks. This alone does not authenticate the enclosing
  SRA result; use the existing strict source/journal consumer separately.
- the same eight-step projected raw-action gradient routine for composition
  and SRA initializers, evaluated in fixed order. Each arm has nine score
  callbacks and eight gradients; the best evaluated iterate is retained,
  with earliest-index tie-breaking, including the final post-update score.
  The objective/gradient provider is an interface, not a native model meter.
- attempt-before-callback, return/failure events and fail-stop behavior.
  These are emitted to the caller; this kernel alone does not provide a
  durable journal, independently verified CUDA execution, or total resource
  enforcement.
- a contract built from the real SHA-bound saved B0 capture and pinned SRA
  source, while explicitly blocking production execution. Contract SHA-256:
  `ddb48fa3ab5e1d0ed6bb016aaab86d90b48e6e280e160266da68f72b965b6b40`.

The SRA initialization schedule is the existing 88 predict and action
encoder calls, 1 Jacobian, 2 SVDs and 1 solve, with 192 requested scalar VJP
directions; it is not a measurement of this new run. The common optimizer
counts are callback counts only, not verified predict/backward internals.
Composition initialization costs, a shared end-to-end budget and a native
CUDA adapter with nested operation metering are not yet bound. Consequently
`total_budget.status=blocked`, `production_execution_authorized=false` and
`equal_total_compute=false`. Unknown FLOPs, energy and I/O stay unknown.
The eight-step / 0.05 update is a fixed development configuration chosen for
this bridge after B0 outcomes were already known, not a strong-optimizer
validation, a preregistered scientific comparison, or an established
faithful composition-vs-SRA mechanism test.

Production requires: strict authentication of both same-input initializers,
including an action-generating composition witness; an authenticated,
frozen native objective and gradient adapter for both arms; durable
attempt/return logging and full model-operation hooks; an agreed total budget
that includes both potentially unequal initializer costs, all common
optimization, replay and shared setup; and an independent post-plan physical
handoff. The capture is a known historical root. No new outcome or P1/P2
admission follows from this contract.

Validation (CPU only): 34 tests passed across this bridge, the existing online
SRA adapter and native SRA code. The bridge's optimization tests use an
analytic fixture, not real neural or physical execution.
