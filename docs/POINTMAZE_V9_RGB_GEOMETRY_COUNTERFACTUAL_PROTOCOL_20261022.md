# PointMaze V9: RGB Geometry Counterfactual Repair Protocol

## Trigger

V8 used fresh roots and a robust two-fold candidate-bank design. Its pre-fit
headroom gate passed, but the frozen-latent recurrent repair did not beat the
action-only control on its independent outcome roots. This rules out candidate
collapse as the primary remaining explanation for that panel; it does not
prove that RGB observations lack relevant state information.

The matched PointMaze builder already stores a causal eight-frame
`source_history_visual_state` series derived from rendered RGB observations.
V6--V8 did not expose this series to their learned repair models.

## Question

Can a predeclared RGB-geometry history restore action-effect ranking where the
frozen latent history fails, without using outcome labels to choose the feature
family or policy?

## Fresh data and isolation

- New root IDs: `[11320, 11832)`.
- Base data: the immutable V8 extension; V8 context is included in the prior
  identity ledger, so roots and candidate identities are excluded from V9.
- Root partition: 128 train, 192 calibration, 64 outcome roots.
- Candidate design: fixed, shared, in-support 16-sequence bank selected by the
  V8 worst-fold robust goal-headroom criterion before model fitting.
- Outcome labels remain sealed until a checkpoint, feature family, temperature,
  gate, and authorization decision are frozen.

## Predeclared model comparison

The selection split compares the same pulse-FiLM effect architecture over
three feature families:

1. `source_history_model`: frozen latent history plus strictly pre-candidate
   action blocks (V8-compatible control);
2. `source_history_visual_state`: causal renderer-derived RGB geometry history
   plus the same action blocks; and
3. `hybrid_latent_visual_state`: their concatenation plus the same action
   blocks.

The action-only, shuffled-history, and privileged-state arms remain
diagnostics. Only the chosen visual family may become active, and only if its
disjoint authorization bootstrap lower confidence bound over action-only is
strictly positive.

## Interpretation rules

- A valid V9 gain for RGB geometry localizes the V8 failure to the frozen
  latent interface; it is not evidence that frozen latent LeWM alone solves
  counterfactual planning.
- A valid hybrid gain beyond RGB geometry would show incremental value of the
  learned latent under this renderer and action support.
- If no visual family is authorized while the privileged state arm remains
  non-positive, the remaining bottleneck is finite-horizon action-effect
  transfer rather than candidate-bank balance or missing visual coordinates.
- This is an offline renderer-backed PointMaze test, not live RGB deployment,
  cross-environment transfer, or a universal world-model result.

## Execution

Run only from the GPU host terminal:

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
./experiments/decision_benchmark/run_pointmaze_counterfactual_contrast_v9_rgb_geometry_host.sh
```
