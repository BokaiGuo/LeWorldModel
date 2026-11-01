# E1 forced-candidate six-slot native physical adapter

Review candidate only; no worker production physics or neural inference. It
binds the actual fixed-contract six raw10 and the complete saved replay receipt
`0605d651ffcb7b4edafed61c12604a89453594e7df5b573b3a944a048c548dc5`,
report `ba2df80d87c825452cdbf3050813e6720f074cd69ba70aea4f716b9007d5717e`,
completion `8e1af678def94bb603ca2e50452452892b1188aa19a35e85281815c2e7e68783`
and timing `dd3c6dfb30fb906f06f22b3b188cf490010d473377d217acc8633b192423d8db`.
Prepare independently reconstructs the complete upstream saved replay receipt;
that saved check performs no neural forward. Per slot it joins exact root,
task, raw10, compiled target and paid whole replay contexts, derives the original
target/goal model costs, and bridges the exact source root/history/goal. This is
not an assertion that whole/seg equality on one job is a universal gate.

The adapter is a new source-local derivative of the already audited E1 six-slot
physical adapter, not a patch or a global monkeypatch. Fixed order is two roots
× bank_winner/zero_extra/witness_extra; all actions remain forced. It replays the
original warm10 and native branch, keeps original early-stop/done, captures all
frames and cp0/5/10, and independently reconstructs journal hash chain,
source RGB/state/snapshot, exact actions/time, native metrics, full operation
intents/returns, resource counts and missing checkpoints. Partial evidence is
preserved; report publication failure may leave an emergency marker and a
filesystem failure may prevent even that. It inherits native physical limits:
120 public, 1224 physics, 132 explicit PNG, 6 constructors/closes, 12 resets,
24 set-state and 12 goal renders. These are ceilings under the sealed recipe,
not observed new costs. The 32 MiB storage census is a hard allocation, not a
proof that every possible native output fits. Scientific/P1 gates remain false,
known roots are calibration, and total costs/FLOPs are not claimed equal.

MAIN after audit, with new absolute paths and emitted hashes:

```
.venv/bin/python -B -m experiments.spectral_reachability.agent_e1_forced_candidate_physical_20261029_v1 prepare --output /ABS/prepare.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_e1_forced_candidate_physical_20261029_v1 verify --prepared /ABS/prepare.json --prepare-sha PREP_SHA
.venv/bin/python -B -m experiments.spectral_reachability.agent_e1_forced_candidate_physical_20261029_v1 run --prepared /ABS/prepare.json --prepare-sha PREP_SHA --output /ABS/NEW_OUTPUT --authorize-physics
.venv/bin/python -B -m experiments.spectral_reachability.agent_e1_forced_candidate_physical_20261029_v1 consume --output /ABS/NEW_OUTPUT --report-sha REPORT_SHA --receipt /ABS/NEW_RECEIPT.json
```

Fixture test `.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_e1_forced_candidate_physical_20261029_v1.py` uses fake native environments and temporary files only; the one real saved-ABI test does not call upstream fullstrict.
