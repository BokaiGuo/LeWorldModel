# Fixed-population predicted-tail development intervention

On the previously observed two Reacher roots and selected, zero, and
same-bank-greedy prefixes, recover each prefix's predicted-model CEM tail
using the existing sealed model report. Reproduce the original successor
CEM's random stream, all 32 samples and distribution updates exactly.
After its final iteration, evaluate the recovered old tail separately as a
33rd rollout and select it only if its successor model cost is strictly
lower than the original final-iteration best. Ties keep the original.

This design holds the original candidate population and updates fixed; it
costs one additional two-packet model rollout per branch, plus a singleton
encoding/goal evaluation, and is not an equal-budget comparison. Singleton
and batched model evaluations can differ numerically, so report costs and
do not infer significance from tiny cost differences.

The prefix, history, observed RGB and frozen checkpoint must match the
previously sealed physical branch. The source contains the original
model-predicted tails only through reproducible fixed seeds and scores;
their recovery adds experimental overhead. Execute all six selected
suffixes physically. Compare raw old/new suffix actions, real final qpos,
and original versus new selector costs. These roots and prior outcomes
were observed already; the intervention is exploratory, not an independent
confirmation or P1/PC gate.
