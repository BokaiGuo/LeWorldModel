# P1 native reset source readiness

The existing calibration source explicitly supports a bounded reset/render
exploration with incomplete global history, but its existing calibration2
seeds have already been used and its three RGB frames are causal padding,
not a new actual moving history. It is not a prospective confirmation pool.
The existing `fresh_candidate` source path requires its historical certificate
and original source/closure gates; the currently pinned saved root manifest
records coverage and source-run authorization as false. Do not use its four
reserved slots in calibration mode to work around that requirement.

The separately named readiness program reads and hashes exact existing sources,
statically checks the native reset call structure, and creates or verifies an
exclusive JSON receipt. It does not choose seeds, generate candidates, import
the simulator, call reset, train, use CUDA, or authorize anything. The static
inspection reports two direct `_set_state` calls per reset, each with one
`space.step`, and one direct render call. This is a source-level inference,
not runtime telemetry, a complete transitive closure, or a zero-physics claim.
Source drift causes rejection; the prior manifest and all gates are unchanged.

Run from `le-wm-main` with a new result name:

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_native_reset_readiness_20261101_hypatia create --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_native_reset_readiness_20261101_hypatia_NEW.json
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_native_reset_readiness_20261101_hypatia verify --preseal /home/gbk/code/LeWorldModel/le-wm-main/results/p1_native_reset_readiness_20261101_hypatia_NEW.json --sha256 RETAINED_PRESEAL_SHA256
```

The resulting asset is a **readiness receipt, not an executable candidate
producer**. Before an actual source run, the main thread must establish the
applicable source contract and historical exclusion/identity evidence, freeze
the actual candidate roster and selection-free rule, bind and test the native
runtime and real-history/goal receipt, and meet the original fresh-source and
operational requirements. Source, training membership and global freshness
remain unknown; no outcome-based replacement or threshold change is allowed.
