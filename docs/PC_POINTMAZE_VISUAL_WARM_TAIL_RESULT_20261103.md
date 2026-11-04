# Model-scored retained-tail RGB-feedback development result

The previously selected initial packet for each of four development roots
and both initial CEM arms was physically replayed; the real cp5 RGB was
encoded on host CUDA and used to score tails. Cold and retained-tail searches
each used 3 × 16 one-packet model predictions, the same random innovations,
task mask, goal, frozen model, action bounds and feedback opportunity. The
retained tail occupied one existing proposal slot. Each selected tail was
then physically executed in a separate native RGB bridge rollout.

| Root | Initial arm | Cold distance | Retained-tail distance |
| --- | --- | ---: | ---: |
| 1884 | random | 1.02661 | 1.02661 |
| 1884 | spectral | 1.67904 | 1.67904 |
| 1885 | random | 0.99383 | 1.08699 |
| 1885 | spectral | 2.40032 | 2.40032 |
| 1886 | random | 0.82040 | 0.82040 |
| 1886 | spectral | 0.95427 | 0.95427 |
| 1887 | random | 2.01301 | 2.01301 |
| 1887 | spectral | 1.60678 | 1.60678 |

Distances are position-backend replay relative to archive frame 10, not
native success. In seven of eight matched groups the tail action, image and
distance were identical. In the remaining group, root 1885 random arm,
retention changed the search and selected a physically farther result; its
model score also worsened from 0.77395 to 0.78702. Thus retaining a candidate
guarantees only that warm CEM's **model** score is no worse than that
retained candidate's model score. It cannot guarantee improvement over
cold CEM, nor over the old plan's physical outcome.

Cold tails, scores and RGB outcomes matched the sealed source. The saved
proposals, generation updates, source masks, image hashes and budget counts
were rechecked; four tests passed across this and the physical-CEM diagnostic.
Each tail used 48 prediction samples in three model calls; the initial two
packet plan cost 48 prediction samples / 96 block transitions, and the
spectral arm also used two JVPs and one SVD. These are recorded operation
counts, not a full end-to-end wall-time equality certificate.

This is a post-outcome development intervention on four exposed roots, with
the source's fixed raw-zero proprio and full-archive normalizer. The planner
does not consume evaluator state, but the checkpoint's original training
interface remains uncertified. Native RGB and position replay use distinct
backends and no independent confirmation of these new endpoints is claimed.
These results do not establish the non-greedy mechanism, P1 or generalization.

- Protocol: `docs/PC_POINTMAZE_VISUAL_WARM_TAIL_PROTOCOL_20261103.md`
- Runner: `le-wm-main/experiments/a1_pointmaze_visual_warm_tail_20261103.py`
- Result SHA-256:
  `2017ad6558a2b15c276843089c4cb02ae9088241568ed7417ba16f7ae02d4724`
- Arrays SHA-256:
  `b433cd8ebb2bd0de6f2d72128ccc4c94d796adc79a07d8991779c0ca2a7f4dea`
- Tests: `le-wm-main/tests/test_a1_pointmaze_visual_warm_tail_20261103.py`
