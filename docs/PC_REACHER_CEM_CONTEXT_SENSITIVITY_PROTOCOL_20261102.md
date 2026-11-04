# CEM context sensitivity on previously opened Reacher roots

Development-only diagnosis of whether the nearly equal tails from predicted
and real-RGB successors arise across fixed CEM seeds and budgets. The two
roots and all their physical outcomes have already been exposed. This is not
an independent method comparison, not an outcome-based selector, and cannot
establish P1.

Use the saved root and cp5 RGB/action-history contexts for prefixes 0 and 12
on seeds 20263160 and 20263161. Verify the original journal chain, identity,
prefix and history bindings and source hashes. Recompute the predicted
successor from the root and the real successor from cp5 RGB using the same
frozen checkpoint, goal, action normalization and model-only CEM.

For each root/prefix, compare the two successor contexts using paired seeds
`root + 5000 + offset` for offsets 0, 1, 2, and budgets `(16,2,4)` and
`(32,3,4)` for samples, iterations and elites. The same seed is used in both
contexts within a pair. Record both complete suffix parameter arrays, best
costs, elite trace, model call counts, context discrepancy and wall time.
Do not select a setting based on these results for a confirmatory test.
There are 48 CEM invocations, 3,072 rollout evaluations and 6,144 predicted
sample evaluations, in addition to encoding/prefix preparation. No new
physical actions, model training, CPU neural inference or oracle state
are allowed. Save all arms, failures and model hashes.
