# Exact orientation mechanism: development result

The protocol is `PC_EXACT_ORIENTATION_MECHANISM_PROTOCOL_20261103_CODEX.md`;
this is an analytical, privileged known-dynamics experiment, not an RGB
planning result, independent validation, or an admission decision.

For 16 predetermined goal directions, turn-only restricted candidates yield
14 common-suffix and independently optimized zero-prefix comparisons in favor
of the selected restricted prefix. This establishes task-level existence for
the specified restricted diagnostic. The candidate construction explicitly
sets first-step translation to zero, so it does not establish that an
unrestricted algorithm discovers or benefits from such a prefix.

| Method | Mean terminal distance |
| --- | ---: |
| Restricted response/SVD turn-only bank | 0.130309 |
| Full-action multistart gradient, 8 evaluations | 0.468188 |
| Full-action multistart gradient, 16 evaluations | 0.073176 |
| Full-action multistart gradient, 32 evaluations | 0.000009 |
| Best of four 32-sample CEM seeds | 0.088690 |

The gradient optimizer beats the restricted method on 16/16 roots; the best
of four CEM seeds beats it on 12/16. The latter is explicitly an aggregate of
four runs, not a single 32-sample planner. Evaluation types are not equated:
each restricted bank uses 45 transitions and nine SVDs; the gradient optimizer
uses 30–32 closed-form objective/analytic-gradient evaluations at its nominal
32 limit. Full per-root traces, seeds, ledgers, and wall time are in the JSON.
An exact analytical gradient is privileged and is a strong diagnostic, not a
deployable RGB baseline.

The combined selector chooses gradient in all 16 cases. The JSON field
`combined_witness` is **only** a structured-branch witness flag, since its
implementation explicitly requires the structured branch to win; its value
zero must not be interpreted as “the selected gradient chains have no
non-immediate mechanism.” Replaying the already sealed selected gradient
actions reveals four actual selected chains with non-improving first-step
distance. For roots 3, 4, 11, and 12 respectively, first-step distances are
0.9602, 0.9498, 0.9168, and 0.9723 versus initial distance 0.9. On those same
chains, applying the selected second action after the selected first action
changes distance relative to its own zero suffix by -0.9602, -0.9498, -0.9168,
and -0.9723; applying that same second action after a zero first action changes
distance by +0.2440, +0.4998, +0.5334, and +0.3256. The final distances of
these gradient chains are approximately zero and below the analytically
optimized zero-prefix suffix distances. This is a post-seal diagnostic of
actual gradient-selected actions, not a result produced by the restricted
spectral candidate bank or a completed fair mechanism comparison to all
baselines. In particular, it does not show spectral structure reduces search
cost, nor does it complete a real RGB or physical intervention.

Conclusion: this task exhibits control-response changes and even four selected
non-immediate exact-dynamics chains, but ordinary full-action gradient search
already solves it. The present spectral turn-only construction does not show
incremental search value. Do not reopen the stopped PointMaze mechanism from
this result, and do not transfer the exact state or gradient to the visual
selector.

Artifacts:

- `le-wm-main/results/pc_exact_orientation_mechanism_20261103_codex.json`
- `le-wm-main/experiments/spectral_reachability/pc_exact_orientation_mechanism_20261103_codex.py`
- `le-wm-main/results/pc_exact_orientation_selected_audit_20261103_codex.json`:
  separately replays the actual selected chain and analytically optimizes the
  suffix after each prefix; four of 16 meet its stated witness conditions.
- `le-wm-main/tests/test_pc_exact_orientation_mechanism_20261103_codex.py`:
  three tests passed, including the gradient-selected witness audit.

The performance line remains separately exploratory: the previous four-root
feedback comparison has an incomplete end-to-end cost certificate and
unverified independent-root/checkpoint provenance. The PointMaze loader's
normalizer provenance also remains bounded by
`PC_POINTMAZE_NORMALIZER_CONFIRMATION_BOUNDARY_20261103.md`. Do not freeze or
run a purported independent confirmation until these contracts are satisfied.
