# Fixed rejected mixed prefixes: physical development results

All 32 previously sealed proposals were physically executed, followed by
their own ten-step CEM suffix planned from the real cp7 RGB-derived context.
No proposal was retroactively promoted to a model-selected method.

| Root | Prediction vs actual RGB cp7 reversals | Prediction vs physical cp7 reversals | Physical nonprogress | Outcome-exposed candidates beating zero and greedy |
| ---: | ---: | ---: | ---: | --- |
| 8343 | 2/28 | 1/28 | 7/8 | none |
| 8345 | 1/28 | 1/28 | 7/8 | none |
| 8349 | 2/28 | 6/28 | 0/8 | raw 4; spectral 3 |
| 8351 | 3/28 | 3/28 | 1/8 | none |

The 28 pairs in each row share one root and are not independent episodes.
Physical nonprogress means cp7 position distance was at least cp2 distance.
Neither root 8349 candidate that beat both controls was nonprogress: raw 4
improved by about 0.100 at cp7 and reached distance 0.4494 at cp17; spectral
3 improved by about 0.067 and reached 0.6466. Zero and greedy on that root
reached 0.7095 and 0.8302 respectively. No candidate across the four roots
both met this physical nonprogress condition and beat both controls after its
own replanning.

The model had selected the shared greedy prefix in the source run and
predicted terminal costs of 0.4438 for greedy, 0.6944 for raw 4, and 0.9852
for spectral 3 at root 8349. This identifies an outcome-exposed selection
failure for the complete feedback procedure on that root, **not** a direct
same-action terminal prediction reversal: the earlier lookahead tails and the
later real-RGB replanned tails need not be the same. The finding motivates
a fixed-action terminal comparison and selector evaluation; it does not
establish a reconfiguration mechanism or a deployable spectral advantage.

The completed rerun used 65 RGB/state encodes, 1,600 two-packet model
samples, 544 RGB-backend steps and 832 position-backend replay steps, in
addition to the source experiment's costs. A first attempt executed all
branches but failed before saving because an old rank helper expected nine
instead of eight candidates. That attempt's costs are additional and its
unsaved outputs are not evidence; see
`PC_CAUSAL_MIXED_REJECTED_PHYSICAL_FIRST_ATTEMPT_20261103.md`.

The saved audit checked 32 source-prefix, action, score and physical-distance
bindings and passed; it did not rerun model inference or independently
replay the saved RGB histories. Six focused tests passed. No common-suffix
intervention, new roots, strong baseline, complete resource certification,
native task success or scientific gate is included.

- Protocol: `docs/PC_CAUSAL_MIXED_REJECTED_PHYSICAL_PROTOCOL_20261103.md`.
- Runner: `le-wm-main/experiments/pc_causal_mixed_rejected_physical_20261103.py`.
- Results: `le-wm-main/results/pc_causal_mixed_rejected_physical_20261103_DEV01.json`
  and its same-name NPZ.
- Audit: `le-wm-main/results/pc_causal_mixed_rejected_saved_audit_20261103_DEV01.json`.
