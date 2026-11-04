# Dev304 same-action model/physical comparison

This post-outcome analysis uses all six fixed branches of the known-root
common-tail experiment. Each branch starts from its own actual cp5 RGB
history and action history, then predicts the exact ten executed tail
actions. We compare an open two-packet prediction with one that replaces the
first predicted successor by the actual RGB encoding before predicting the
second packet. The real intermediate RGB is used for diagnosis only.

The CUDA run saved the full 192-dimensional predicted and observed vectors,
rechecked the frozen model, and recorded 31 encoder/projector calls and 18
action-encoder/predictor calls. The reported squared errors and goal costs
below are independently calculated from those saved float32 vectors using
float64 arithmetic. The saved analysis binds six row shards, the physical
report, actual action hashes, RGB frames and the source contract.

| Prefix | Tail | Open step-10 MSE | Refreshed step-10 MSE | Actual latent goal cost | Object position error |
| --- | --- | ---: | ---: | ---: | ---: |
| single | single | 0.16878 | 0.02256 | 1.42896 | 207.536 |
| single | two-population | 0.29123 | 0.17953 | 1.69541 | 180.112 |
| single | zero | 0.03242 | 0.01514 | 1.89035 | 217.346 |
| two-population | single | 0.04877 | 0.01450 | 1.74546 | 232.670 |
| two-population | two-population | 0.13234 | 0.10395 | 1.57182 | 205.792 |
| two-population | zero | 0.02787 | 0.01214 | 1.84257 | 234.879 |

Refresh reduces step-10 latent prediction MSE in all six branches, with
descriptive means 0.11690 and 0.05797 respectively. Within each prefix,
all three pairwise rankings of terminal goal distance agree between model
prediction and real RGB encoding (6/6 for open and 6/6 for refreshed).
Real latent and physical object-position rankings agree in 5/6 pairs.
Specifically, after the single-population prefix, the single tail has lower
real latent goal cost but worse final object position than the two-population
tail. The single tail also has better angle error, so this is a
metric-dependent tradeoff, not proof that either action wins the full task.

These are six branches and six pairwise comparisons from one known root,
not six independent samples. Lower prediction MSE does not prove improved
control or generalize the absence of ranking reversals. RGB/action-history
and checkpoint bindings were checked, but checkpoint training-data and
normalizer provenance and complete compute costs remain uncertified. No P1
or Planning Compiler gate passes on this evidence.

Artifacts:

- `results/p1_dev304_common10_prediction_20261102_RUN_DEV01/report.json`
- `results/p1_dev304_common10_prediction_saved_20261102_DEV01.json`
