# Mixedmode pre/post probe contract v2: bounded full closure

v1 source/tests/doc/results remain sealed. MAIN's actual v1 build failed with
`ValueError: complete prepare8MiB` before creating a prepare file; 43 RAM tests
did not exercise the full 43k-artifact closure. This is engineering capacity,
not a model/physics/scientific negative result.

v2 is a standalone copy with own NAME/schema/source/test/doc binding, and one
capacity change: metadata/prepare/read/write/journal cap64MiB instead of8MiB.
No global replacement/monkeypatch of v1. Probes/order/40 denominator/raw buffers,
source/goal/Q/normalizer, resource accounting, selection and scientific flags
are unchanged. Complete artifact paths/hashes retained; no payload copied.

The v1 innerdoc remains the detailed unchanged ABI/mathematical reference:
`innerdocs/agent_mixedmode_capability_probe_20261028_v1.md`.
Every own schema/stem in commands/evidence must instead be v2, not spoof v1.
The NEW executor `agent_mixedmode_capability_physical_20261028_v1` supplies its
own prespecified render/physics/wall/output limits; probe prepare still says
execution_ready=false because the contract alone does not grant authority.

Actual saved-only in-memory build regression (no prepare/simulator/neural):

```bash
RUN_MIXEDMODE_SAVED_BUILD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
 .venv/bin/python -m pytest -q -s -p no:cacheprovider tests/test_agent_mixedmode_capability_probe_20261028_v2.py
```

MAIN prepare command, only after review; worker does not execute it:

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_mixedmode_capability_probe_20261028_v2 prepare \
 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_mixedmode_prefix_20261028_v1_cuda_MAIN01 \
 --report-sha d6166b98958af70cecb8529135396c77467b7c803bfe12a1a5de29c422c22ad2 \
 --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_mixedmode_prefix_20261028_v1_prepare_MAIN01.json \
 --prepare-sha 8f6d027116b76334e80f9a761688ffd33825bc25d825484012d37230f083ed6b \
 --source-sha ec912669c678960aac0d004a2e764b4bf036194673da5d1c63d0b535bc199396 \
 --destination results/agent_mixedmode_capability_probe_20261028_v2_prepare_MAIN01.json
```

64MiB is a hard engineering upper bound, with200MiB floor; actual serialization
bytes are printed by the saved regression. No physical evidence created.
Interpretation remains known-root/time-confounded finite response calibration,
not P1 admission or theory refutation. Failure/earlystop denominators remain40.
