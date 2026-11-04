# Exact successor known-plan injection diagnostic

Per `PC_RGB_SUCCESSOR_EXACT_WITNESS_INJECTION_DIAG_PROTOCOL_20261103_CODEX.md`,
each of the 24 exposed wall roots was given one privileged analytic full
trajectory **only through a diagnostic joint-refinement hook**. The hook
used one existing refinement sample to score that trajectory, compared it
with the normal refinement result, and let the original final zero-prefix
comparison and selector run. The total remained 480 prediction tokens/root.
The regular refinement budget was consequently one sample smaller than in
the original unmodified run.

| Check | Result |
|---|---:|
| Eligible witness replayed | 24/24 |
| Float32-scored witness within `<0.05` | 24/24 |
| Final selection is the injected witness | 24/24 |
| Final zero-prefix fallback selected | 0/24 |
| Maximum float32 action conversion discrepancy | `2.87e-8` |
| Maximum final Euclidean distance after conversion | `2.01e-7` |

All 24 injected actions were found in the actual prediction trace; the final
score equaled the minimum of the normal refinement, injected witness and
final zero-prefix comparison. The selected chain was independently replayed
using the exact wall backend, and tests verified trace accounting and that
the temporary hook is restored. Six related tests passed.

This supports the narrow conclusion that **when a complete good chain is
provided at the joint-refinement boundary, this backend's action conversion,
full-chain scoring and final fallback do not prevent retaining it**. The
hook is privileged and changes the candidate pool: these 24 successes are not
planner outcomes, not evidence of automatic discovery, and must not be
combined with the original 0/24 as a performance result. The experiment does
not exercise earlier candidate pruning, visual observation, or physical RGB
execution, and it does not pass PC2 or PC3.

The audit-time traces and hashes are saved at
`le-wm-main/results/pc_rgb_successor_exact_same_policy_dev_20261103_codex_witness_injection_diagnostic.json`.
Together with the original coverage audit (no generated prefix crossed the
wall, no sampled full chain succeeded), the finding prioritizes diagnosing
candidate coverage and suffix search rather than asserting an interface
failure at final selection. It does not uniquely distinguish those causes.
