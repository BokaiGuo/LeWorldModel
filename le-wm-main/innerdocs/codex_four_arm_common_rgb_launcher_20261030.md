# Known-root common-suffix RGB exploration

This is a new exploration, not independent confirmation, P1 admission,
authenticated strong baselines, or established total-cost fairness. It is
separate from RUN01 and does not change its files. Slot 0 is fixed by the
existing RGB source prepare; its task root binding is
`a3bd09470de75652506544b8bea0ff3c5fbae0c0d3b28775da47ee3e1adbb6f7`.
Each arm starts from the verified same source setup, receives its own fresh
actual RGB and executed raw history at 0/5/10/15, and executes the first
five commands of the finally replayed H3 selection. Existing guards remain
unchanged; exceptions fail closed. Early native termination is recorded
without treating an unreached 20-step endpoint as achieved.

All arms use the same 8-candidate, two-step raw projected-gradient suffix and
final H3 replay. Structured uses the spectral8 H3 initializer; the other
initializers are a zero-centered Gaussian native-style engineering variant,
random Gaussian, and one-packet immediate-progress screening. The native-style
variant is not installed native CEM or a verified strong baseline. This is a
mechanism comparison, not a comparison of separately optimized methods.

The same pre-fixed per-node vector cap covers the largest arm requirement on
each accounted axis. Spectral Jacobian/VJPs/SVD, screening, suffix, and final
replay are charged in durable node ledgers; actual totals differ by arm. Visual
encoding and physical calls are recorded and capped separately by the native
meter, not in that node vector. Internal operations, FLOPs, GPU time and full
end-to-end cost are not independently attested. Caps and wall/CUDA memory
checks are cooperative, not hard preemption; the checks use current allocated
and reserved memory, not a guarantee on peak use.

CPU-only prepare executed:

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.codex_four_arm_common_rgb_launcher_20261030 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/codex_four_arm_common_rgb_launcher_20261030_PREP01.json
```

Prepare SHA-256:
`5faee027ab3a7971edd55be8b5531764835e396e79966ddb318b5caa21152be4`.
It includes 5,253 source/file bindings inherited from the prior prepare and
the explicit new launcher/node dependencies. Run reconstructs and checks
these before execution, checks actual model-class files against the manifest,
and checks source and model state again afterward. This is not a proof of
complete transitive closure, runtime binary integrity, or commitment time.
Any drift requires an independently reviewed new exclusive prepare.

Only after independent review and explicit human authorization, a host with
the required frozen checkpoint, CUDA, native environment, and sufficient disk
may use a fresh output path:

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.codex_four_arm_common_rgb_launcher_20261030 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/codex_four_arm_common_rgb_launcher_20261030_PREP01.json --sha256 5faee027ab3a7971edd55be8b5531764835e396e79966ddb318b5caa21152be4 --output /home/gbk/code/LeWorldModel/le-wm-main/results/codex_four_arm_common_rgb_launcher_20261030_RUN01 --authorization MAIN_AUTHORIZES_COMMON_SUFFIX_RGB_CUDA_AND_PHYSICS
```

The command was not run. CPU integration uses analytic model and environment
substitutes and does not validate the real checkpoint CUDA ABI, native RGB,
or physics. Runtime checks callable model methods and then validates actual
tensor shapes and results during operation; neither replaces a real ABI
validation. Inspect `report.json`, episode records, and every node ledger and
candidate journal. A nonzero exit or `partial_or_failed` is not complete;
`recorded_known_root_not_admitted` says only that all arm outcomes were
recorded, including possible early native stops. Failures before the output
store is created, and some I/O failures, may leave no complete failure report.
