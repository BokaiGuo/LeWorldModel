# Native direct full-horizon CEM diagnostic

After observing the fixed-bank development diagnostic, compare a privileged
direct full-15-action optimizer on the same four starts, MuJoCo 2.1.2,
official XML, original reset and terminal position distance. No learned
model or RGB controller participates. This is an exposed-data diagnostic,
not a frozen confirmatory comparison.

Before running, fix one common search of 24 independent CEM starts per
root, NumPy seed 2026110319. Each start has three generations of 16
15-by-2 sequences, four elites, initial standard deviation 0.5, floor
0.02, action bounds `[-1,1]`; initial mean is zero for start 0 and a uniform
random sequence for starts 1–23. After each start, additionally evaluate
its final mean and a zero sequence, yielding 50 evaluations per start.
Report best-so-far actual objective and its action sequence at 100, 300,
600 and 1200 full-trajectory evaluations. All 1200 scores and action
sequences are saved. No early stopping or reward replacement is allowed.

The prior bank spent 100 suffix evaluations for each of 12 prefixes, plus
prefix-generation and common-suffix costs. The 1200-evaluation full-CEM
point is therefore a useful physical-call-matched diagnostic, not equality
of wall time or all preprocessing costs. At smaller budgets report the
curve, not only a chosen favorable point. Search mechanisms and random
seeds differ; comparing exposed outcomes does not establish statistical
or independently generalizable advantage.
