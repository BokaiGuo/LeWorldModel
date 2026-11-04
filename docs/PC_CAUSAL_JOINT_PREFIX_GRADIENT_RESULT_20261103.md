# Joint prefix/suffix gradient search development result

All four exposed roots completed projected nonlinear gradient search using
the predeclared raw/spectral initializations, fixed zero/greedy prefixes,
strict model nonprogress criterion and equal 50-sample real-RGB feedback
suffix CEM. The model returned one qualifying prefix: raw initialization at
root 8349, candidate 3, iterate 11. The unrestricted winners are separate
from this qualifying result.

| Root | Model-qualified prefix | Qualified cp7 distance | Qualified cp17 distance | Zero cp17 | Greedy cp17 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8343 | none | — | — | 1.0001 | 0.0892 |
| 8345 | none | — | — | 1.6653 | 0.7831 |
| 8349 | raw 3, iteration 11 | 1.6674 | 0.9690 | 0.7095 | 0.8302 |
| 8351 | none | — | — | 1.8376 | 0.7751 |

For root 8349 the qualified prefix's model task score was 1.6839 against
initial 1.6555, and physical position distance increased from 1.6554 to
1.6674. Its separately replanned physical terminal distance was nevertheless
worse than both controls. This does not establish useful non-greedy
reconfiguration, though the result motivates the separately recorded fixed
common-suffix intervention.

The two unrestricted root-8349 candidates reached 0.4696 (raw) and 0.3640
(spectral), versus 0.7095 and 0.8302 controls; both already improved
physical distance during the prefix and do not meet this experiment's
nonprogress criterion. These are model-selected development outcomes on an
exposed root, not independent confirmation. Raw and spectral candidates are
jointly optimized from different distributions, and the spectral candidate
generation reuses a previously computed Jacobian; equal search settings do
not establish equal total resource use.

Run cost: 520 three-packet model samples, 48 batched action-gradient calls,
850 feedback CEM samples, 22 encodes, 289 RGB-backend steps and 442 separate
position-backend replay steps, plus inherited candidate/Jacobian work. The
model checkpoint was frozen. The saved-output audit checked all optimizer
decisions and physical prefix bindings; it did not independently rerun
inference or prove normalizer provenance. Candidate trajectories and
gradients are retained in the same-name NPZ.

- Protocol: `docs/PC_CAUSAL_JOINT_PREFIX_GRADIENT_PROTOCOL_20261103.md`.
- Runner: `le-wm-main/experiments/pc_causal_joint_prefix_gradient_20261103.py`.
- Report: `le-wm-main/results/pc_causal_joint_prefix_gradient_20261103_DEV01.json`.
- Joint and common-suffix audit:
  `le-wm-main/results/pc_causal_joint_common_suffix_saved_audit_20261103_DEV01.json`.
