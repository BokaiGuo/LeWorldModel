# P1 coherent suffix family: known-root physical intervention

This protocol is exploratory on one previously observed translated PushT root.
It is not independent confirmation, an equal-controller-compute comparison,
or authorization to open P1. No new model planning or neural inference occurs.

## Fixed intervention

The complete roster is the ordered Cartesian product of the six actual prefixes
`structured`, `native`, `random`, `immediate_progress`, `zero`, and
`same_bank_greedy` with seven common 15-step tails: `zero15` and the six
previously sealed own planned15 tails. The same seven tails are applied to every
prefix, including cross-prefix transfers; no tail is selected based on an outcome.
All 42 20-action schedules must be finite, in [-1,1], and established before
creating an environment. A common tail may not be replaced by an identical
looking but differently sourced plan.

The six-prefix summary is pinned by SHA-256. Preflight verifies its entire
source map, the two sealed planning reports, bindings, trace selections and
ledgers via their existing read-only collectors, and the four existing physical
reports and their source/protocol closure via the existing summary reader. For
each prefix, the planned 20 actions must equal the actual own physical report;
the first five actual actions and cp5 state/pixels must also agree with both
existing common/zero physical branches. The summary cp5 metrics must agree.
Every source is rechecked before dispatch, before each prefix's seven branches,
and after the run; source drift stops subsequent dispatch. This is source-file
closure over the recorded manifests, not a claim that all third-party runtime
dependencies or hardware are independently sealed.

For each branch, use the existing exact centered reset, ten-action warmup with
RGB comparisons at 0/5/10, goal attachment, and root-state comparison. Record
issued actual actions, step attempts/returns, state, RGB, and existing physical
metrics every five steps and on early stop or canvas censoring. Compare the
actual cp5 pixel and state digests to that prefix's source; a mismatch fails that
branch. A terminated or censored trajectory is retained as stopped, never
padded to 20 steps. The runner creates a new directory, writes the full protocol
once before physics, checks its digest during and after execution, and prints
branch start/end. Per-branch JSON and PNGs preserve partial progress on ordinary
exceptions; abrupt process loss may leave an incomplete report.

## Review and execution

CPU contract tests use fixtures and do not execute a simulator or a model:

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
CUDA_VISIBLE_DEVICES='' .venv/bin/pytest -q tests/test_p1_coherent_suffix_family_20261101_leibniz.py
```

Read-only full source/roster preflight, no physical execution:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.spectral_reachability.p1_coherent_suffix_family_20261101_leibniz
```

Only after main-thread review and explicit physical authorization:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.spectral_reachability.p1_coherent_suffix_family_20261101_leibniz --authorize-physical --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_coherent_suffix_family_20261101_leibniz_NEW
```

The output directory must not exist. This runner has not been physically run as
part of implementation; a validated schedule is not a result.
