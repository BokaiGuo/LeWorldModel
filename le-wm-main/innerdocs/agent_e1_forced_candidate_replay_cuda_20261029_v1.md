# E1 forced-candidate six-slot paid replay

MAIN review candidate; no worker CUDA or physics. Fixed actual forced contract
`494f806e5bc5d46221b28ce2ad7f288bee4b46d0b3eb3020706f4a07ecf2ed37`
is fully reconstructed from its saved parents and sealed/reloaded before model
loading. Six slots are root0/1 × original bank winner/zero/witness-prefix;
no CEM, selection, clipping or padding. Each runs literal raw10 whole H2 and
sequential 1+1, 4 predict/API/action-encoder/projector calls per slot, total 24
each. Full native normalized action histories, module inputs/outputs, predicted
contexts and operation attempts/returns are journaled by the original Dispatch
and backend guard. All six saved outputs are sealed before whole/segmented bit,
max-absolute and ULP diagnostics. These are not physical outcomes.

The runner binds the exact original saved task/target/source and declared
model closure, frozen config/weights buffers, CUDA-only model, weight tensor
hash before and after, TF32/autocast disabled and requested MATH. 4 GiB CUDA
allocator cap, 900 s through completion fsync, 1 GiB Store cap with 64 MiB
terminal reserve and 200 MiB floor are inherited hard boundaries, not proof
unseen CUDA payloads fit; a failed disk may prevent a durable terminal report.
The consumer checks external prepare/report/completion/timing SHA, model/source
closure, archive/hash-chain and exact all-slot saved operation reconstruction;
it does not rerun neural computation. Receipt inherits parent strict evidence.
FLOPs remain null, proposal generation and old controller costs are separate,
and no equal-total-cost, physical ability, fresh-root or scientific claim.

Fixture tests use only a synthetic backend and temporary disk. MAIN CLI after
review from le-wm-main with fresh absolute paths and emitted actual SHA:

```
.venv/bin/python -B -m experiments.spectral_reachability.agent_e1_forced_candidate_replay_cuda_20261029_v1 prepare --output /ABS/prepare.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_e1_forced_candidate_replay_cuda_20261029_v1 verify --prepared /ABS/prepare.json --prepare-sha PREP_SHA
.venv/bin/python -B -m experiments.spectral_reachability.agent_e1_forced_candidate_replay_cuda_20261029_v1 run --prepared /ABS/prepare.json --prepare-sha PREP_SHA --output /ABS/NEW_OUTPUT
.venv/bin/python -B -m experiments.spectral_reachability.agent_e1_forced_candidate_replay_cuda_20261029_v1 consume --output /ABS/NEW_OUTPUT --report-sha REPORT_SHA --completion-sha COMPLETION_SHA --timing-sha TIMING_SHA --receipt /ABS/NEW_RECEIPT.json
```

`.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_e1_forced_candidate_replay_cuda_20261029_v1.py`
