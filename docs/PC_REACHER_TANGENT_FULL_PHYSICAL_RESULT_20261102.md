# Fixed tangent bank: full physical common-tail intervention

The fixed protocol `PC_REACHER_TANGENT_FULL_PHYSICAL_PROTOCOL_20261102.md`
completed all 70 branches: two exposed roots × seven saved prefixes × five
saved common tails. Every branch executed 15 steps without termination or
truncation; all 1,050 step rewards were zero. The saved normalized actions
reconstructed the raw actions exactly, both roots and goals were bound, and
the model-only selector remained `null` for both roots. No candidate below is
a model-selected policy.

The following table uses evaluator-only qpos distance, with negative
terminal difference favoring the candidate over prefix 0 under the *same*
tail. Prefix change is relative to the common initial state. The adjusted
range is, over tail indices 1–4, the candidate's terminal change relative to
its own tail-0 terminal minus the same quantity for prefix 0. Tail 0 is a
neutral-parameter tail, not asserted to be bitwise zero raw torque.

| Root | Prefix | Initial-to-prefix change | Range of 5 terminal differences vs prefix 0 | Range of 4 adjusted differences |
| --- | ---: | ---: | ---: | ---: |
| 20263160 | 1 | +0.020118 | +0.014063 to +0.017946 | -0.003691 to +0.000192 |
| 20263160 | 2 | -0.001668 | -0.014348 to -0.000306 | -0.003297 to +0.010745 |
| 20263160 | 3 | +0.033749 | +0.028284 to +0.035884 | -0.007217 to +0.000384 |
| 20263160 | 4 | -0.009679 | -0.028663 to -0.003860 | -0.006698 to +0.018105 |
| 20263160 | 5 | +0.061109 | +0.057089 to +0.071683 | -0.013824 to +0.000770 |
| 20263160 | 6 | -0.025113 | -0.057164 to -0.004565 | -0.013823 to +0.038776 |
| 20263161 | 1 | +0.068147 | +0.002166 to +0.010311 | -0.004977 to +0.003168 |
| 20263161 | 2 | +0.058602 | -0.005574 to -0.004701 | +0.000003 to +0.000873 |
| 20263161 | 3 | +0.073471 | +0.004495 to +0.020664 | -0.009899 to +0.006269 |
| 20263161 | 4 | +0.054407 | -0.011113 to -0.009372 | +0.000010 to +0.001742 |
| 20263161 | 5 | +0.084160 | +0.009653 to +0.041403 | -0.019520 to +0.012230 |
| 20263161 | 6 | +0.046160 | -0.022067 to -0.018637 | +0.000048 to +0.003431 |

Initial distances were 1.003500 and 1.182600. Prefix 0 changes those by
+0.006530 and +0.062843, respectively. On the second root, prefixes 2, 4 and
6 all worsen the initial distance yet start their tails closer than prefix 0;
their better absolute terminals therefore do not isolate improved capability.
Their positive adjusted differences also do not favor those candidates under
any of the four active tails for this qpos metric. On the first root the
corresponding better absolute terminals follow immediate distance
improvements. Other candidates show mixed adjusted effects.

Predicted Q deltas and physical qpos deltas have different units and task
meaning. Their differing signs are a diagnostic lead, not a direct estimate
of world-model prediction error. No true RGB successor encoding was compared
with predictions, no native success was certified, and each root shares
actions across branches. The results cannot establish independent
generalization, equal-budget advantage or P1 admission.

The complete stepwise data are in
`le-wm-main/results/reacher_pc_tangent_full_physical_20261102_v1/report.json`,
SHA-256 `2c3956da23c504f20b4b05d113e30c3f3055169c021e10e63b6732c963c90f74`.
