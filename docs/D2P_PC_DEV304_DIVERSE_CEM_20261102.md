# Dev304: budget-matched two-population candidate search

## Development contract

This post-outcome development experiment repeats the known dev304 root;
it is not a holdout or an independent confirmation. Both arms use the same
frozen checkpoint, goal, action limits, RGB-only selection inputs, five
iterations of 64 candidate plans with five perturbation variants, and the
two-packet latent terminal score. Each has 10 predictor calls and 3,200
predictor samples. It does not establish equal end-to-end wall-clock costs.

The baseline uses one CEM population. The experimental arm allocates 32
candidates to each of two independently updated populations, placing the
RGB contact-direction seeds in one of them on iteration zero. Four elites
per population maintain separate distributions. Crucially the final plan is
the globally lowest robust-score feasible candidate over all populations and
iterations; the experimental arm is not forced to pick a structured seed.
The contact-direction seeds are geometric, not spectral or model-derived.
The fixed five-step execution is not full RGB-feedback replanning.

## Results

Both arms completed the same five-step physical horizon without native success.
The initial state and execution inputs were bound to the same reset.

| Arm | Selected model score | Final native reward | Object position error | Object angle error |
| --- | ---: | ---: | ---: | ---: |
| Single population | 1.70544 | -313.69347 | 217.55357 | 3.11914 |
| Two populations | 1.85368 | -340.30741 | 234.87911 | 2.91596 |

These are one known development root, not two independent tasks. The two
population method produced a different model-selected plan, but did not
improve native reward or object position on this run; its angle error was
lower. No result establishes a reconfiguration mechanism or a fair baseline
advantage. Its chosen candidate was iteration 4, index 36, i.e. in the second
population, but this alone does not identify a physical causal mechanism.

The saved verifier checks the complete recorded bank, candidate feasibility,
per-population elite rankings, global winner, action hashes, frozen model,
model-call ledger, replay checkpoints, physical transitions and 20 frame
hashes. It checks recorded model scores for internal consistency; it does not
independently rerun the model or prove model predictions match physics.
The baseline selected action and outcome match the earlier terminal-objective
development run. Source training membership and complete historical reset
exclusion remain unknown.

Artifacts:

- `results/p1_dev304_diverse_cem_20261102_PLAN_DEV01/report.json`
- `results/p1_dev304_diverse_cem_20261102_EXEC_DEV01/report.json`
- `results/p1_dev304_diverse_saved_20261102_DEV01.json`

The result supports continuing to investigate model ranking, scoring horizon
and actual feedback, rather than claiming that candidate collapse alone was
the cause of the original failure. Physical oracle, same-prefix mechanisms,
independent roots, equal end-to-end budgets and stronger planners remain open.

## Observed cp5 RGB feedback follow-up

For each prefix, a fixed single-population CEM was run from its own observed
cp5 RGB, the two preceding real RGB frames, the common goal image and the
actual action history. It used the same optimizer seed and 3,200 predictor
samples per arm. Plans were made before their physical continuation; each
prefix was replayed against its saved state and RGB snapshot and the selected
first five tail actions were then executed. No simulator state entered the
feedback planner. This is a post-outcome development extension on the same
root, with one feedback opportunity, not a prospective independent study.

| Prefix source | After feedback tail: native reward | Object position error | Object angle error | Native success |
| --- | ---: | ---: | ---: | --- |
| Single population | -304.24317 | 207.31349 | 2.82594 | No |
| Two populations | -310.98076 | 209.66211 | 3.10527 | No |

Both executed ten actions in total. These are two trajectories from one root,
not two independent samples. The tail actions differ between prefixes, so this
is a development performance comparison, not a shared-tail intervention or
proof of changed control ability. The verifier binds the feedback contract,
plan, model hash, model-call ledger, actions, replayed prefix and 18 frames;
it does not certify full upstream costs or independently rerun neural scores.
The physical runner's source was not included in the pre-run feedback
contract, a provenance limitation to close before confirmatory use.

Saved follow-up: `results/p1_dev304_diverse_feedback_saved_20261102_DEV01.json`.
