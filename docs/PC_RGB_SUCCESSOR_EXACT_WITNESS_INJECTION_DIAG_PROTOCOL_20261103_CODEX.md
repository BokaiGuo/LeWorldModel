# Exact successor known-plan injection diagnostic

This post-outcome engineering diagnostic runs all 24 previously exposed wall
development tasks with the unchanged successor policy, except for one
diagnostic hook in its joint-refinement call. The hook spends one of the
existing refinement samples to score an analytic feasible full chain, and
returns the lower-scoring chain between that input and the regular refinement
result. The remaining refinement calls, 480-token total cap, score function,
final zero-prefix comparison and tie rule remain the original policy.
Privileged witness information is used only inside this diagnostic and is
never introduced to the published planning run or counted as method success.

For each root choose the first minimum-step analytic witness that reaches the
goal in at most 15 actions, using the existing feasibility routine; if none
exists, fail the audit rather than selecting roots. Convert physical actions
to normalized actions and preserve the original float64 requested chain;
the real prediction interface performs its usual float32 conversion. Record
normalized bounds, float32 round-trip action discrepancy, physical endpoint
and score, actual input batch/score, refinement winner, final selected
chain, zero-prefix comparison and physical replay. Verify that the witness
actually occurs in the prediction trace and that the selected score is the
minimum of both refinement candidates and the zero-prefix comparison.

This tests a known full chain at the refinement/complete-chain selection and
fallback boundary. It does **not** test the earlier candidate generator,
intermediate pruning, real RGB or an actual deployed environment. It cannot
show the optimizer could discover the witness, and the injected witness must
not be reported as performance. Record all roots and all failures; do not
adjust selection, seeds, budget or success thresholds from outcomes.
