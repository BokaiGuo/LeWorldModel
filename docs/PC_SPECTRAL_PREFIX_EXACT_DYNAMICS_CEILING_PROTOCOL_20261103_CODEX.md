# Sealed prefix bank: privileged exact-dynamics suffix diagnostic

Use all 16 already executed prefixes from the sealed same-action result. The
known fixed 12-action histories and reset states remain unchanged. In the
audited MuJoCo 2.2 position backend replay each full history and prefix, then
optimize the remaining five raw actions using the same two-start CEM budget
for every prefix: three iterations of 16 samples, four elites, final
comparison of the zero prior and last mean, for 100 terminal evaluations
per prefix. Generate two fixed normalized standard-normal noise arrays per
root with predeclared seeds 2026110810 and 2026110820, respectively; use the
same arrays for all eight prefixes of that root. Convert actions with the
frozen dataset normalizer and enforce raw bounds [-1,1].

All physical positions and optimized suffixes are privileged evaluator-only
information and must never enter the RGB selector. For exact replay use the
full original reset/history/prefix per candidate rather than resetting from a
possibly incomplete intermediate state. Save every sample, score, elite,
final comparison, actions and count. Independently verify the physical cp12,
cp17 and original cp22 endpoints against the existing saved evaluator.

Report initial and each of the five prefix-step physical distances,
including whether an early worsening is later followed by progress, and
compare each prefix's own optimized final distance to the zero and greedy
prefixes under the same optimizer. Define physical non-immediate progress
using the already used whole-prefix position rule: cp17 distance >= cp12.
A negative result from this finite optimizer is not proof that no feasible
suffix exists or that the task mathematically requires no reconfiguration.
This is a two-root exposed development diagnostic, not a method, selection
rule, independent experiment, same-suffix mechanism intervention, or admission.
