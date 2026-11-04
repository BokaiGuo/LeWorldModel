# Fixed-population tail intervention result

The predeclared development protocol is `PC_REACHER_TAIL_FIXED_POOL_PROTOCOL_20261102.md`.
On both known roots and all three sealed prefix arms, the original 32-sample
successor CEM population was reproduced and the original imagined tail was
evaluated separately as sample 33. All six real physical action witnesses
and all six final qpos distances exactly match the original physical run.
The cp5 raw action and successor RGB/action-history identities also match.

The new selector's strict numeric comparison labels the old tail as chosen in
all three arms of root 20263160. In each of those arms its parameter array is
**identical** to the original best CEM plan; apparent cost improvements of
about 1–2e-7 arise between singleton and batch evaluation, not a different
physical plan. On root 20263161 the selected and zero imagined tails differ
from the original CEM best parameters, but their singleton costs tie at saved
float32 precision and the preset tie rule retains the original plans. The
greedy imagined and original parameters are equal, with a small singleton vs
batch cost difference.

Therefore this fixed-population intervention provides no new control gain.
In particular, the previous inserted-tail experiment's root-20263161 zero
degradation is consistent with changed CEM candidate distribution rather
than a direct benefit of the selected prefix. This is still a two-known-root
development observation, not a general causal claim or a gate result.

Per arm the new selector costs 33 rather than 32 suffix rollouts, corresponding
to 66 rather than 64 horizon-weighted prediction samples, plus one singleton
encoding/goal evaluation. It is not equal-budget. The `selected_old_tail`
flag must not be interpreted as a distinct action without checking parameters
and raw actions. Three tests for fixed-population preservation, strict
selection, and equivalence to original CEM passed, alongside the prior
predicted-successor and carryover tests.

Artifacts:

- `le-wm-main/results/reacher_pc_fixed_pool_tail_20261102_v1/report.json`
- `le-wm-main/results/reacher_pc_fixed_pool_tail_20261102_v1/wrapper_receipt.json`
- `le-wm-main/experiments/spectral_reachability/reacher_pc_fixed_pool_tail.py`
