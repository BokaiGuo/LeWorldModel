# Four-arm known-root RGB engineering exploration

This is a new exploratory protocol, not independent confirmation, P1 admission,
or a claim of equal total cost. The fixed root is slot 0 of the prior RGB
prepare. Each arm resets and verifies the same original state/history/goal,
then observes its own actual RGB at boundaries 0, 5, 10 and 15, plans afresh,
and executes only the first five raw commands. Native early termination is
recorded separately; an unexecuted endpoint is not a success. Existing RGB and
canvas guards remain in force; exceptions fail closed and stop later arms.

The arm algorithms are in `codex_four_arm_own_planners_20261030.py`.
Structured uses spectral8 H3 and per-candidate gradient refinement; native is
a self-implemented two-generation raw CEM variant, **not** the installed
native optimizer or an authenticated strong baseline. Random uses two pools;
immediate progress screens one packet before H3 replay. These are different
optimizers and carry unequal costs; this is not the common-suffix mechanism
comparison. Each node persists candidate events and a separate consumed ledger;
the episode journal records actual physical and visual calls. These counters
do not attest every internal operation, FLOPs, GPU time, or equal budgets.
Action arrays use raw [B,T,5,2] and the checkpoint's per-axis mean/std.

The prior spectral8 prepare has root binding
`6812ab89db47612390424790b001921168d0102dde47b39f07669bfeba6cd0a1`;
the RGB slot-0 task binding here is
`a3bd09470de75652506544b8bea0ff3c5fbae0c0d3b28775da47ee3e1adbb6f7`.
Their serialized root contexts and goals were compared equal; the hashes differ,
so this does not independently establish identical physical root provenance.
The existing setup additionally checks the actual RGB, history, dynamics, and
physical root against the fixed RGB source when run.

CPU-only prepare, executed once in this work:

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.codex_four_arm_rgb_physical_20261030 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/codex_four_arm_rgb_physical_20261030_PREP01.json
```

Observed SHA-256:
`3ec898356cc9e71fa8b0ff94f8bb0ea56336e13e0279cdf50d2520c558553546`.
Prepare outputs are exclusive. Any source or input drift requires a new
exclusive prepare path and using its newly printed hash.

After independent review and explicit authorization, the main thread may run
the following on the host CUDA environment, choosing a fresh output directory:

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.codex_four_arm_rgb_physical_20261030 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/codex_four_arm_rgb_physical_20261030_PREP01.json --sha256 3ec898356cc9e71fa8b0ff94f8bb0ea56336e13e0279cdf50d2520c558553546 --output /home/gbk/code/LeWorldModel/le-wm-main/results/codex_four_arm_rgb_physical_20261030_RUN01 --authorization MAIN_AUTHORIZES_FOUR_ARM_RGB_CUDA_AND_PHYSICS
```

This command has not been executed. CPU substitute tests are not a real CUDA
or environment integration test. Confirm available host CUDA, frozen model
and dependencies, disk, guard behavior, action ABI and full call coverage
before authorizing. Limits are cooperative checks, not hard preemptive limits:
1800 seconds, current allocated 3 GiB and reserved 4 GiB, output 512 MiB
with final reserve, plus fixed per-node and physical/visual caps. A long call
can exceed a limit before the next check. A hardware policy hash is not
hardware attestation. Model class files are checked against prepared sources;
this is not proof of a complete transitive runtime closure.

Inspect `report.json`, each episode, all node receipts and event journals;
nonzero exit or `partial_or_failed` is not a completed panel. A
`recorded_known_root_not_admitted` report means the four attempts were
recorded, not that each reached 20 steps, succeeded, or established fairness.
Some failures before output creation cannot produce a report; I/O failure can
also prevent a complete failure report.
