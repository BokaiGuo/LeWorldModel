# Non-progress bank: feedback replanning and common suffix

Use all four exposed obstacle roots and the sealed non-progress bank. The
three prefixes are the model-selected non-progress candidate, the
same-bank unconstrained minimum-score candidate, and zero, each repeated
for five high-level steps. Fail closed if a selected candidate is absent.

For each prefix use actual step-5 RGB and causal velocity, then four
successive feedback decisions at steps 5, 10, 15, 20 with the same
random-initialized AP-compatible CEM, seeds, 50 candidate predictions per
decision, action bounds and frozen model. Execute the resulting actions
without early stopping. Verify each prefix RGB against the sealed bank and
replay every full trajectory in a fresh RGB bridge. Score position through
the audited position backend after execution only. Record elapsed planning,
resources and frame/action arrays; prefix-generation costs are not equalized.

Separately cross each of the three prefixes with each of three twenty-step
suffixes: the just-planned selected tail, the just-planned greedy tail, and
zero. These are open-loop physical interventions; all 36 combinations are
evaluated, and each relative-to-zero effect is computed within the same
prefix. Reproduce original diagonal endpoint positions. This is a
post-selection developer diagnostic on already exposed roots, not a
pre-registered independent confirmation, full-method equal-compute study or
P1 gate. Differences in initial position and velocity remain confounds.
