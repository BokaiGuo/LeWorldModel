# Causal mixed-prefix search: development result

Both candidate arms selected the shared greedy prefix on all four exposed
roots. Neither arm selected a model-nonprogress candidate that strictly
beat both zero and greedy in its predicted terminal score. Candidate
eligibility is not the same as qualification:

| Root | Raw eligible / qualified | Spectral eligible / qualified | Zero cp17 | Greedy cp17 |
| ---: | --- | --- | ---: | ---: |
| 8343 | 3 / none | 4 / none | 1.0001 | 0.0892 |
| 8345 | 3 / none | 0 / none | 1.6653 | 0.7831 |
| 8349 | 1 / none | 0 / none | 0.7095 | 0.8302 |
| 8351 | 0 / none | 0 / none | 1.8376 | 0.7751 |

Distances are physical position-to-goal distances after the two observation
steps, five-step prefix and ten-step replanned suffix. The reported raw and
spectral winners were both greedy, so their complete saved physical actions
and outcomes were identical to the greedy branch on each root. They are
duplicate executions, not independent improvements. The historical zero and
greedy trajectories also matched the prior H10 run in all saved actions and
RGB frames. All 16 branches were executed, but no non-greedy winner was
available for a physical mechanism claim.

The model predicts the greedy final task score to be better than all eight
new proposals at every root. This identifies the current selector's actual
decision on this bank; because the unselected new proposals were not
physically executed, it does **not** establish whether none would be useful
in the real environment. Prior fixed-action evidence already shows that
model and physical ranking may disagree. The raw and spectral proposal
distributions differ deliberately; 4 proposals per arm are insufficient to
rule out other spectral or ordinary search designs.

The run used 2,400 three-packet and 800 feedback two-packet model samples,
21 encodes, 272 physical RGB-backend steps and 416 separate physical
position-backend replay steps. It inherited 40 JVP and 80 finite-difference
samples from the source bank. Matching model sample counts does not certify
equal overall resource usage. The checkpoint was frozen. The causal context
uses two images to estimate state and encodes the latest image; action
history is recorded but not explicitly fed into that one-frame context.

The saved-output audit checked candidate generation, selection, branch
bindings, physical distances, hashes and historical zero/greedy replay:
4 roots and 16 branches passed. It did not rerun model inference or certify
normalizer provenance. Seven focused kernel and selector tests passed. This
is not a common-suffix intervention, independent-root confirmation, native
success or a PC gate result.

- Protocol: `docs/PC_CAUSAL_MIXED_PREFIX_SEARCH_PROTOCOL_20261103.md`.
- Execution: `le-wm-main/experiments/pc_causal_mixed_prefix_search_20261103.py`.
- Report and arrays: `le-wm-main/results/pc_causal_mixed_prefix_search_20261103_DEV01.json`
  and its same-name NPZ.
- Saved audit: `le-wm-main/results/pc_causal_mixed_saved_audit_20261103_DEV01.json`.
