# Planning Compiler: dev304 objective and candidate coverage

## Contract and evidence

The initial run fixed seed 301000004 before its outcome, but history-wide
exclusion and checkpoint training membership are not established. This is a
development root, not independent confirmation. Two CEM variants shared
the same feasible-action guard, frozen model, RGB inputs, random seed and
model budget (5 iterations, 64 candidates, 5 perturbation variants; 3,200
predictor samples per arm). They differ in their initial candidate bank.
These are guarded experimental variants, not an authenticated reproduction
of the complete original LeWM planner.

The first run propagated two model packets but scored the end of the first.
This was the declared experiment, not evidence of an accidental wiring bug;
however, its objective cannot select based on how the second packet changes
the terminal prediction. A separate post-outcome revision scores the second
packet. It uses the same development root and is explicitly exploratory.
Both execute only the first five actions. Neither is a continuous RGB
feedback experiment or an evaluation of actual ten-step program success.

## Saved results

| Objective | Original vs structured bank | Five-step native success | Final reward | Object position error | Object angle error |
| --- | --- | --- | ---: | ---: | ---: |
| First-packet latent | Exactly the same selected 10-step actions | 0/2 | -324.6692 | 215.7553 | 3.1122 |
| Second-packet latent | Exactly the same selected 10-step actions | 0/2 | -313.6935 | 217.5536 | 3.1191 |

Errors are evaluation-only simulator quantities; they are not planner inputs.
The revised objective changes the selected action sequence, but both candidate
banks within each experiment converge to the same sequence. The reward,
position and angle changes do not all favor the same objective; the two
post-hoc outcomes cannot establish an objective or planner advantage.

The structured bank changes first-iteration scores and increases feasible
candidates from 24 to 34. The saved score arrays for iterations 1–4 match
exactly in both experiments. Although elite indices and complete banks were
not saved during the run, deterministic replay of the frozen generator and
saved scores reconstructs identical first-iteration elite indices and action
arrays for both arms in both experiments. The altered initial candidates
therefore did not enter the reconstructed elite set; later searches coincide.
This reconstruction relies on the recorded scores and matching RNG/code; it
does not independently recompute the neural scores. The structured bank uses
RGB contact direction, not a demonstrated dynamics-derived spectral program.

Both saved verifiers check locked contracts and source hashes, the frozen
checkpoint before/after, blind image/action input, RGB hashes and pixels,
prediction-call ledger, action bytes, warmup snapshots, per-step execution,
and explicit non-admission. Each verifies 20 frames. This is a bounded check,
not a certification of checkpoint training provenance, all deployment
normalization, wall time or full end-to-end resource equality.

Artifacts: `results/p1_dev304_coherent_saved_20261102_DEV01.json`,
`results/p1_dev304_terminal_saved_20261102_DEV01.json` and
`results/p1_dev304_bank_replay_20261102_DEV01.json`.

## Implications

For a subsequent prospective development protocol, save the full first
candidate bank and elite indices and use the full-horizon objective with
explicitly budgeted costs. Test candidate diversity with controlled
alternatives and compare actual subsequent RGB-feedback execution. Do not
select roots or successful tails using these outcomes. A physical mechanism
intervention, independently sourced roots and strong budget-matched baselines
remain necessary; neither the legacy P1 gate nor the new PC gates pass here.
