# RGB successor policy on exact dynamics: development result

Under `PC_RGB_SUCCESSOR_EXACT_SAME_POLICY_DEV_PROTOCOL_20261103_CODEX.md`,
the existing RGB successor policy's first decision was run without modification
on all 24 previously exposed wall tasks. The predictor was replaced by exact
wall dynamics; this privileged state-based diagnostic is not an RGB result
or an independent confirmation. The stopped six-dimensional block-program
construction was not rerun.

The policy reached mean final distance **0.52813**, with **0/24** successes
at the predeclared `<0.05` threshold; the final zero-prefix fallback was
selected on 8/24 roots. Each decision consumed 480 prediction tokens, with
11,520 total tokens and 1,800 recorded model batches. The exact 24-root
feasibility witnesses previously replayed for this panel were not given to
the policy. These results establish neither a theoretical impossibility nor
performance relative to a sufficiently tuned strong optimizer.

After sealing plans and auditing all input/output batches, interventions
were run on each *final selected chain*, including fallback decisions:

| Predeclared diagnostic | Count |
|---|---:|
| Prefix without immediate progress | 9/24 |
| Selected suffix relative response beats zero and same-bank greedy | 8/24 |
| Same-budget own-prefix replanning beats both | 7/24 |
| All three on the same selected chain | **0/24** |

Each prefix was evaluated with the same six suffixes and received 120 CEM
samples; actual absolute distances, zero-relative changes and search scores
are in the sealed intervention JSON. Same-bank greedy is selected from the
policy's actual first bank, not a post-hoc alternative chain. These conditions
are a limited mechanism screen; random bridging, lookahead and passive-motion
diagnostics have not been completed for this policy. PC2 is not passed.

## Engineering diagnosis

All 1,800 recorded batches were recomputed under the exact dynamics and their
input casts, positions, scores, selected chain and token sums checked. A
separate post-hoc analysis rebuilt the 72 recorded first and successor response
matrices. At the original 0.05 finite-difference scale the maximum absolute
matrix discrepancy was 0.0. Changing the scale to 0.01 or 0.1 changed the
matrix by over 10% in 6/72 and 10/72 contexts respectively; maximum absolute
Frobenius changes were 2.179 and 1.128. Relative changes use a denominator
floor of `1e-12`, so absolute errors are also reported. This supports a
scale-sensitivity risk, not a demonstrated implementation bug or a unique
cause of failure. The diagnostic did not modify the selected actions or
choose a new scale.

The plan, batch audit, intervention and response diagnostic are saved under
`le-wm-main/results/pc_rgb_successor_exact_same_policy_dev_20261103_codex_*`.
The same-chain intervention test independently recomputes all common suffix
and own-replanning results. The related test selection passed 21 tests.
This is a negative development result for this version and panel; it cannot
be combined with other policy versions to claim PC2, PC3, or spectral benefit.
