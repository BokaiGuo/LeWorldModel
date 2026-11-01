# Four fixed-root gradient-MPC physical witnesses

Main-authored separate consumer. No changes to the gradient planner, SRA consumer,
old seals or existing physical reports. It consumes only successful gradient CUDA
reports from the four fixed already-used roots and the pre-CUDA archived exporter.
All40 candidate/39 update trajectories and exact independent winner replay are
validated by the planner's array-only exporter before action plans are prepared.
No extra optimization, root replacement, action resampling, training or success
threshold change. P1/P2/scientific efficacy stay false.

Prepare is non-neural and does not instantiate/reset/step/render a simulator.
It freezes source/tests/doc/environment code, exact planner reports/journals,
root/goal/normalization provenance and ten-command selected-witness hashes.
Run rechecks all inputs and reconstructs original physical roots with identical
warmup RGB/history/goal. Existing canonical-agent RGB guard is unchanged; no blind
continuation/retry if observation is unavailable. Every partial remains recorded.
Complete trajectory requires ten commands, same order, exact source-selected
actions. Reset/warmup are not part of selected-path step counts and are not free
end-to-end planning costs.

All physical branches are durably written before optional model loading. Complete
ten-step endpoints are encoded on host CUDA only, one per root; partial endpoints
retain physical evidence but cannot stand in for fixed-horizon encodings. Prediction
is forbidden in the encoder-only phase. Unknown upstream/backward/FLOP/walltime
costs are not zero. The 336 inherited predicts match sampling counts, but additional
156 autograd.grad scalar-objective VJPs/156 updates prevent an equal-total-budget
claim. These are not official GRASP or a paper reproduction.

Successful full report is calibration_recorded. Guard-only partial report is
guarded_calibration_recorded, not a task success; genuine integrity/encoding errors
write integrity_fail and exit nonzero. Source/model freeze and all branch slots
remain checked after execution. Outputs must be exclusive new directories.

From le-wm-main (only the final command allocates models; main host CUDA):

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p2_gradient_mpc_physical_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p2_gradient_mpc_physical_20261027_v1 prepare --planning results/p2_full20_gradient_mpc_20261027_v1_prepared01 --output results/p2_gradient_mpc_physical_20261027_v1_prepared01
SDL_VIDEODRIVER=dummy PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p2_gradient_mpc_physical_20261027_v1 run --prepared results/p2_gradient_mpc_physical_20261027_v1_prepared01 --output results/p2_gradient_mpc_physical_20261027_v1_cuda_main_v1 --encode-terminals
```

No CPU neural fallback. Fixed pure-array fixtures test plans/receipt rejection,
not task efficacy; only actual report/trace/RGB/encoding receipts prove execution.
