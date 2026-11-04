# Reacher predicted-successor CEM: physical development follow-up

This is a development diagnostic, not PC2/PC3 admission, independent
confirmation, or a controlled-budget comparison. The candidate bank comprises
source action packets, not compiled spectral programs. No extra network is
trained and qpos is used only for evaluator-side distances.

The immutable model-only source report was bound by SHA256. The follow-up
checks checkpoint, action statistics and selector hashes, reconstructed root
and goal RGB hashes, history index, bank receipt and selected parameter bytes.
Each arm executes one five-step packet, re-encodes its own observed successor
and actual action history, and replans a two-packet suffix with the same CEM
parameters and seed. For the root with no eligible selected proposal the
predeclared fallback is the neutral-parameter candidate; that root is retained.
The nominal zero action has numerical residuals below 4e-8 in this run.

| Seed | Arm | First-packet qpos distance | Final qpos distance |
|---|---|---:|---:|
| 20263160 | selected, index 6 | 0.999733 | 0.550038 |
| 20263160 | zero, index 0 | 1.010031 | 0.546408 |
| 20263160 | same-bank immediate greedy, index 9 | 0.826512 | 0.540761 |
| 20263161 | selected fallback = zero | 1.245444 | 0.598310 |
| 20263161 | zero | 1.245444 | 0.598310 |
| 20263161 | same-bank immediate greedy, index 8 | 0.771386 | 0.786956 |

Initial qpos distances were 1.003500 and 1.182600 respectively. The first
selected prefix slightly improves this evaluator distance versus the initial
state and also outperforms the neutral prefix immediately, so it cannot be
presented as a physical non-improvement witness under that metric. The second
root's apparent selected advantage over greedy belongs to the zero fallback;
it is not evidence for a selected reconfiguration.

The execution identities verify replayed root, goal and successor image
hashes, and equal fallback executions reproduce the same observed hashes and
metrics. These checks do not authenticate the complete historical source
chain or independently prove all numerical interfaces correct. The saved
same-bank greedy is determined by the model-only report, not physical outcome.
The selector required 1,056 model-predict samples per root in its source
report. Each physical arm reports 32 suffix rollout evaluations; the full
cost of encoding, physical steps, hardware time, shared selection and auxiliary
zero-prefix replay is not certified. A common upper limit is not an equal-cost
comparison.

The panel has no common-suffix intervention, no task-success assessment, no
independent roots or strong baselines, and no statistical inference. It cannot
establish a change in future controllability. Next work must compile genuine
dynamics-based candidates and predefine common-suffix physical interventions,
then assess independently replanned suffixes and strong baselines within a
complete resource contract.

Artifacts:

- `le-wm-main/results/reacher_p1_successor_cem_development_20261102_v1/report.json`
- `le-wm-main/results/reacher_p1_successor_cem_physical_development_20261102_v1/report.json`
- `le-wm-main/experiments/spectral_reachability/run_reacher_successor_cem_physical_development.py`

Tests: `test_reacher_successor_cem_physical_development.py` and
`test_reacher_p1_predicted_successor_cem.py`, 13 passed.
