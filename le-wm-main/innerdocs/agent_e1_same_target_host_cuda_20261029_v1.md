# E1 same-target four-slot host CUDA candidate

Source-first implementation for MAIN review; worker did not run production.
The entry uses pinned original controller prepare/report/receipt and exact
compiled result/task join, fixed two roots and shared prefix target. Prepare
rehashes the inherited declared source/model bindings and local execution
sources; it must fail if any differs. It is not a proof that an unstated
installed dependency is in the historical closure. Model loading binds the
same hashed config/weights bytes, CUDA only, frozen before/after tensor hash,
TF32/autocast disabled, MATH requested, 4 GiB allocator peaks, original
native guard and original task normalizer. No CPU fallback or physics.

Four original 72-bank controllers, two exact paired-bank checks and global
barriers precede the four extra candidates, four fixed choices and four paid
whole/segmented replays. 144+2+4 packet operations per slot, 600 per model
axis, no training/AD/encoder/physics. Raw10 is reloaded from the sealed choice;
original Dispatch applies one normalizer and saves actual module IO. Proposal
cost and information remain separate; no equal FLOPs or native reachability.

The inherited 1 GiB Store, 64 MiB final reserve, 200 MiB floor, 10,000 event
limit, bounded record/payload, and 900 s through completion fsync are hard
guards, not proof all unseen native values fit. Its stated bound derives from
admitted record ceilings; storage/fsync failures can prevent even final report.
Timing witness fsync itself is outside its own scope. Partial/error evidence
is rejected by consumer and attempted/completed counts persist when publication
succeeds. Original MAIN receipt is inherited, no upstream fullstrict rerun.

Consumer binds external prepare/report/completion/timing SHA, reloads sources,
task and target, checks all file hashes/array extents/hash-chain, original model
attestation, 4-slot timeline, saved controller reconstruction, 73-way selection
and replay, exact resource axes and after-source closure. It does not rerun a
neural forward or independently attest arbitrary omitted dependencies. Fixtures
use an explicitly artificial backend and synthetic attestation, not CUDA.

MAIN review and independent tests are required before any production dispatch.
Commands from `le-wm-main`, with fresh absolute paths and emitted actual hashes:

```
.venv/bin/python -B -m experiments.spectral_reachability.agent_e1_same_target_host_cuda_20261029_v1 prepare --output /ABS/prepare.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_e1_same_target_host_cuda_20261029_v1 verify --prepared /ABS/prepare.json --prepare-sha PREP_SHA
.venv/bin/python -B -m experiments.spectral_reachability.agent_e1_same_target_host_cuda_20261029_v1 run --prepared /ABS/prepare.json --prepare-sha PREP_SHA --output /ABS/NEW_OUTPUT
.venv/bin/python -B -m experiments.spectral_reachability.agent_e1_same_target_host_cuda_20261029_v1 consume --output /ABS/NEW_OUTPUT --report-sha REPORT_SHA --completion-sha COMPLETION_SHA --timing-sha TIMING_SHA --receipt /ABS/NEW_RECEIPT.json
```

Fixture test: `.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_e1_same_target_host_cuda_20261029_v1.py`. P1/scientific gates remain false; known roots are not independent confirmation.
