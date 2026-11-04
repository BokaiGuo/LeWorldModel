# CEM32-selected, neutral and same-bank greedy: live development comparison

After opening the full-bank CEM32 model-only selection, execute all three
declared prefix roles on both already-exposed Reacher roots: selected,
neutral index 0, and same-bank immediate-goal greedy. Preserve the model's
chosen index exactly, including no-selection as a failure; do not pick a
replacement using physical outcomes. All six arms execute their five-step
prefix, reobserve RGB and action history, run the same 32-sample, three-step,
four-elite CEM with `root + 5000`, and execute their ten-step suffix.

The existing executor also performs a five-step neutral-prefix physical
diagnostic per arm, and uses simulator state only for reset/metrics. Track
all 20 physical step intents/returns and model candidates (96 per tail).
Report initial, cp5 and final joint distance, all arm failures and complete
selection costs: 1,248 CEM rollout evaluations and 2,522 model-predict samples
per root for full-bank selection, plus the per-arm feedback replanning.
Encoding, timing and complete candidate-generation cost remain uncertified;
this is not an equal-total-budget strong-baseline experiment.

The two roots, candidate bank and historical outcomes were already exposed.
This is exploratory; no inference of independent generalization, P1 mechanism
or superiority may follow merely from the six-arm result.
