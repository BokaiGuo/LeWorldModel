# P2 predict frontier: executable CUDA protocol v1

Implementation-thread predeclared under authorized implementation; pending main
review, not user-picked algorithm parameters. This is a new experiment, not an
edit/reseal of the old paired v2. No new outcomes used for proposal choice.

## Fixed object and dispatch

Reuse original four already-used roots, exact root/context/RGB/goal0/Q/action
normalization/model source bindings from paired v2 preparation. Same open-loop
H2 (10 raw commands), whole20 pre-tanh L2 cap sqrt(10)*.75 and raw tanh box.
No physical state or new observations enter the planner. This is not a perpacket
ball, not an identical proposal distribution, not an independent cohort.

Fresh shared compilation MUST execute: nominal H2 two predicts plus Jacobian
forward H2 two predicts, joint192x20, no detach/recompile at prefix. Two CPU
SVDs and one bounded ridge solve (.002) inherited unchanged. Archived banks
are NOT reused as free compilation credit. Fresh40 generation is preserved,
expanded using the frozen bridge recipe: PCG64(20261027+root index), 10 normalized
Gaussian9 mixtures of inherited top9 modes, original2 radii x signs, all40 kept.
Legacy40 has no RNG seed. The prefix is identical within this fresh execution;
degenerate/repeated action slots are not redrawn or selected after outcomes.

Joint evaluates all80 real H2 candidates (160 predicts). Conditional composition
uses original40 prefixes, each prefix one predict, suffix nominal and Jacobian
forward two predicts, final suffix one predict (160). Each winner is independently
segmented-H2 and whole-H2 replayed (4). Standalone each = shared4+160+4=168.
Paired unique332/root,1328/four roots, same action-encoder entry totals. Shared
work attributed to both standalone arms, never sum those attributions. No
padding, no cached candidate results, no extra replay for nested first40 minimum.
Stable first original nonlinear terminal-Q minimum, all120 candidate scores kept.

AD remains unmatched: standalone1 vs41 Jacobians/root,192 vs7872 mathematical
scalar-output VJP directions. Paired164 Jacobians/31488 directions over four
roots. Actual SVD2 plus positive-cap suffix count/root; zero-cap still charges
nominal and AD forward. Solves1 plus positive-cap suffix count. These are NOT
backward engine API counts or FLOPs. Cached RGB/context/Q preparation costs are
inherited unknown, not zero. CPU generation/selection/compile and synchronized
root wall times retained; source verification/export adds saved-data CPU work.

## Guards, files, export

NEW runner calls only old mathematical/transition/compiler primitives, not old
40-slot runner/exporter. Ordered predispatch guards on BOTH model.predict and
action_encoder.forward enforce332/root each, exact scope, batch1/context3,
CUDA metadata and original latent/action dimensions. Visual encoder forbidden.
Attempt fsync precedes dispatch; terminal/error fsync follows. First failure
poisons root, stops remaining arms/roots; no refund/retry or failed-run export.
Partial row/journal remains and report cannot claim completion.

prepare verifies inherited seals/context/input hashes without outcomes/model
execution; seals inherited source closure plus bridge and NEW source/tests/doc.
verify checks current bytes against archives and retained external prepared SHA.
Execute must use a separate NEW output and external prepared SHA. Startup disk
preflight requires300MiB free (100MiB artifact allowance +200MiB reserve); no
download/deletion. Conservative allowance is a preflight estimate, not a measured
disk quota; stop if competing processes consume reserve before launch.

selected_actions is implemented BEFORE sealing: all four stable roots/eight arms,
all80/40 slots, fresh saved-Jacobian bank regeneration, local suffix solves,
all witnesses/context/domain/normalization/provenance, raw10 commands, original Q,
stable minimum, segmented/whole replay contexts, exact actual entry/operation
journals, ledger paths, row/bank/journal/report hashes. Saved-data checking is
not another neural replay. Export validation costs8 root SVD plus0..160 suffix
SVD,4 root solves plus0..160 suffix solves and480 witness/metric evaluations,
separate from inference budget. Neither old physical endpoints nor old40
exporter can substitute new winners. A new physical consumer is downstream work.

P1/P2 admission always false. This quality-versus-model-call frontier makes NO
total-resource/FLOP matching or physical efficacy claim. Main alone launches
CUDA after review. Tests are NumPy/mock and source/preparation checks, not efficacy.

## Commands (from le-wm-main)

```sh
.venv/bin/python -m pytest -q tests/test_p2_composition_predict_frontier_20261027_v1.py tests/test_p2_composition_predict_frontier_cuda_20261027_v1.py
.venv/bin/python -m experiments.spectral_reachability.p2_composition_predict_frontier_cuda_20261027_v1 --prepare --output results/p2_composition_predict_frontier_cuda_20261027_v1_prepared01
# Retain the returned canonical prepared_sha256 outside the prepared directory.
.venv/bin/python -m experiments.spectral_reachability.p2_composition_predict_frontier_cuda_20261027_v1 --verify --prepared results/p2_composition_predict_frontier_cuda_20261027_v1_prepared01 --prepared-sha256 SHA
# MAIN ONLY; not run by implementation thread:
.venv/bin/python -m experiments.spectral_reachability.p2_composition_predict_frontier_cuda_20261027_v1 --execute-cuda --prepared results/p2_composition_predict_frontier_cuda_20261027_v1_prepared01 --prepared-sha256 SHA --output results/p2_composition_predict_frontier_cuda_20261027_v1_main01
# Saved-data only, AFTER main's completed CUDA execution:
.venv/bin/python -m experiments.spectral_reachability.p2_composition_predict_frontier_cuda_20261027_v1 --export-selected --prepared results/p2_composition_predict_frontier_cuda_20261027_v1_main01 --output results/p2_composition_predict_frontier_cuda_20261027_v1_selected_main01.json
```
