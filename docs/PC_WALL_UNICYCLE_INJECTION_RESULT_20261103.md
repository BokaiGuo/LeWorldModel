# Wall-unicycle successful-chain injection: development diagnosis

For each of the three exposed seeds, inject that seed's successful waypoint
control chain as four context-bound packets. All three runs produced the
same offer-count pattern `[2, 1, 0, 0]`: the first packet was offered at two
visits to the initial context, and the second was offered at one visit to
the context after the first packet. The contexts after packets two and
three were never visited; consequently packets three and four were never
offered. No returned complete branch matched the privileged witness
context or action chain. The selected terminal errors remained 0.505090,
0.503323 and 0.502205.

The first-packet successor context being visited is not by itself proof
that a unique named trajectory survived: multiple histories can map to the
same full context. However the lack of a visit to the second successor
directly establishes that this successful continuation did not reach the
later bank in these runs. No direct internal beam trace was captured, so
this result does not identify the precise competitor or prove an
implementation bug. A subsequent direct trace of the unmodified beam
assignment found that the unique named `injected_0` → `injected_1` node
ranked **6th** in all three cases and was discarded by the width-two cut.
Its intermediate model costs were 0.831365, 0.600414 and 0.583625. The
observed retained nodes matched the stable current-cost sorted slice. This
is an actual pruning decision, not merely an inference from missing visits.
Adding a branch can change later random-number
consumption, making this an instrumented injection diagnosis rather than a
paired unchanged-bank counterfactual.

Each injection run used 370 compiler rollouts and 36 Jacobians, including
576 finite-difference rollouts and 3,868 state transitions; each separately
used 16 transitions to construct the context bindings and 16 for selected
replay. This is more than the original 3,320-transition Atlas search. The
injected chain was discovered by a different planner and is privileged;
none of these outcomes is Atlas performance, independent confirmation,
learned-model evidence, or P1 admission.

- Protocol: `docs/PC_WALL_UNICYCLE_INJECTION_PROTOCOL_20261103.md`
- Runner: `le-wm-main/experiments/spectral_reachability/pc_wall_unicycle_injection_20261103.py`
- Result: `le-wm-main/results/pc_wall_unicycle_injection_20261103_DEV01.json`
- Tests: `le-wm-main/tests/test_pc_wall_unicycle_injection_20261103.py`
- Direct beam trace:
  `le-wm-main/results/pc_wall_unicycle_injection_beam_trace_20261103_DEV01.json`;
  protocol `docs/PC_WALL_UNICYCLE_INJECTION_BEAM_TRACE_PROTOCOL_20261103.md`.
- Related regression set including the direct trace: **22 passed**.
