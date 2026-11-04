# Atlas-style matched-call exact-dynamics development protocol

This is a new development ablation on previously exposed goals, not independent
confirmation or a rerun of the full v2 Atlas beam search. Compare three
four-prefix banks: shared zero and local progress plus two independently
seeded uniform random actions, the two signs of the first nonzero local
task-weighted spectral direction, or two ordinary fixed steering actions with
the shared progress velocity and opposite maximum turns. Every prefix gets the
same one-packet relinearized progress suffix and a full-chain direct replay;
select minimum nonlinear terminal task cost with stable first-index tie-break.
All three arms compute the shared root geometry even where it is not needed to
generate their extra candidates. That makes backend calls equal, but does not
equalize actual CPU time or establish a general compute-equivalent comparison.

Freeze indices 0–7 of the already exposed 64-goal unicycle panel and random
seeds 42, 43, 44. Three seeds are repeated observations of each goal, not
independent tasks; deterministic spectral and ordinary arms repeat identically.
Every run is capped at 28 rollout calls and 5 analytic Jacobian calls. Count
attempted and returned calls and action steps; split/full replay equality and
raw action feasibility are required. Root progress uses four one-step rollout
calls and one Jacobian; each of four prefixes uses one prefix rollout, four
one-step suffix rollouts, one Jacobian and one two-step replay. Consequently
28 rollout calls contain 32 action steps, and five Jacobians contain five steps.
Terminal error is Euclidean position error, while selection uses the existing
position-only mean squared cost over three coordinates.

This design isolates the two extra candidates within a fixed, small
Atlas-style search, not a replacement for full Atlas, a random-search frontier,
or an optimized ordinary lookahead baseline. It does not establish a visual,
neural, physical, causal-mechanism or new generalization claim. The real total
compute cost of SVD, candidate generation, allocation, and wall time is not
matched. Report all outcomes, including negative outcomes, without gate changes.

Run from `le-wm-main`:

```
python -m pytest -q tests/test_pc_atlas_matched_exact_dev_20261103_exclusive.py
python -m experiments.spectral_reachability.pc_atlas_matched_exact_dev_20261103_exclusive --output results/pc_atlas_matched_exact_dev_20261103_exclusive.json
```
