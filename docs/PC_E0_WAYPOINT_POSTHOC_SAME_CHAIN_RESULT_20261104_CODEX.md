# E0 waypoint-CEM: post-hoc same-chain intervention result

All eight previously selected final chains were examined under the fixed
[post-hoc protocol](PC_E0_WAYPOINT_POSTHOC_SAME_CHAIN_PROTOCOL_20261104_CODEX.md).
The inherited v5 descriptive mechanism predicate yielded:

| Predicate | Roots meeting it |
|---|---:|
| Selected vs zero and same-bank greedy | 2/8: roots 0 and 7 |
| Additionally vs lookahead and random bridge | 0/8 |

Neither result is a PC2 pass: this is a hand-designed waypoint-CEM baseline,
the task family and observed plans were already exposed, and the follow-up
protocol was written after planning. It does not validate v11, a spectral
contribution, or independent generalization.

For roots 0 and 7, the lookahead control and the actual selected prefix are
exactly the same after conversion to the float32 actions used by the backend.
Consequently, strictly better selected behavior than lookahead is impossible
for those roots. This is a property of the controls, not proof that the
waypoint mechanism is absent. The two prefixes may differ in their saved
float64 decimal representation, so comparison must use executed float32.

Root 2 had a successful selected chain with terminal distance 0.0259. Applying
its selected actual suffix after the zero prefix produced distance 1.7807;
however, the three equal-budget suffix searches from zero initialization after
the selected prefix averaged 0.5011, versus 0.5004 after the zero prefix.
Thus the fixed-tail intervention shows dependence on the prefix in that
setting, while these finite independent replanners did not recover the
original useful suffix. A failure to rediscover a suffix is not a proof that
no useful suffix exists.

On root 0 the selected/zero same-tail distances were 0.0126/1.4247 and their
own-replan mean distances were 0.3427/0.5007; on root 7 they were
0.0558/1.4452 and 0.1779/0.5015. Root 7 did not meet the original
`<0.05` terminal success threshold. All five prefix conditions, absolute
distances, zero-relative changes, all replans, response matrices and
branch-stability flags are in the raw results.

The local response calculation clips the selected suffix's first five actions
to `[-0.9, 0.9]` before finite differences: root 2 and root 6 have a maximum
clip of 0.1. It is therefore a response around that specified nearby nominal,
not an unqualified derivative exactly at the executed tail. The selected
finite-difference batches were branch-stable at both tested scales on roots
0, 2, 6 and 7, but not on roots 1, 3, 4 and 5.

The audit rebuilt all eight results and raw intervention arrays with the same
code. An independently implemented wall-intersection calculation checked
57,960 common-tail and own-replan chains. This is independent geometry for the
same abstract system, not another physical environment or an independent neural
replication. [Raw intervention and audits](../le-wm-main/results/pc_e0_waypoint_posthoc_same_chain_20261104_codex/).
The original fixed panel and all thresholds remain unchanged.
