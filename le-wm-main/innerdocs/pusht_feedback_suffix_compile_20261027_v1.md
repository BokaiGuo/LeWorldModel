# Feedback-context suffix recompilation, fixed known-root calibration

The previous fixed-probe experiment localized substantial imagined-prefix drift.
This new protocol compiles executable suffixes instead of reinterpreting the old
probe outcomes. The same one already used root, three original roles and both
imagined/actual contexts are retained. No roots, modes, radii, thresholds or metric
are selected using physical outcomes. This is not an independent confirmation.

Exactly the original full10 logit chart, norm cap sqrt(5)*.75, damping .002 and
two spectral radii are used. A fresh 192x10 suffix Jacobian around zero yields the
original ten-candidate bank. All ten are evaluated nonlinearly with the original
Q/goal; stable minimum cost wins. Winner verification uses the independent
CudaReplay adapter on that arm's saved context. Prefixes never change. No visual
forward is added: the observed context is inherited from the sealed prior real
RGB encoding, whose upstream cost must not be mistaken for free deployment.

Per context there are 1 nominal + 1 vectorized Jacobian-forward + 10 nonlinear
candidate + 1 verification predictions; total 78. Six dense Jacobians entail
1152 scalar-output VJP directions, not 1152 individual Python backward calls.
The original generator performs two SVDs per context (basis and ridge), 12 total.
Compilation, serialization and six-context diagnostic cost are not an equal
total-compute claim. No environment execution or P1 admission occurs here.

Prepare checks all old source/input seals, projects saved contexts onto generation
inputs (excluding real terminal vectors/diagnostic scores), and seals new source,
test, doc, upstream modules and protocol before loading the model. Failed runs
retain reports and launch receipts; no overwrite/retry in the same output. The
subsequent physical consumer must verify exact prefixes and source/context
bindings and retain safety-stop failures. Success cannot be inferred from model
Q improvement; original object pose and official physical metrics stay unchanged.

Commands from project Python root (second requires main host CUDA approval):

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_pusht_feedback_suffix_compile_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.pusht_feedback_suffix_compile_20261027_v1 prepare --output results/pusht_feedback_suffix_compile_20261027_v1_prepare_v1
.venv/bin/python -B -m experiments.spectral_reachability.pusht_feedback_suffix_compile_20261027_v1 run --output results/pusht_feedback_suffix_compile_20261027_v1_prepare_v1
```
