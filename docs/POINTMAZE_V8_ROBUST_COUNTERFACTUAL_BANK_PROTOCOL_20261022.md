# PointMaze V8: robust counterfactual-bank confirmation

## Why V8 exists

The sealed V7 CUDA experiment established that the action-conditioned model
has a nonzero history-action interaction, but it did not establish transferable
repair.  Its independent outcome regret was `0.01247` for recurrent ranking
versus `0.01189` for action-only ranking, and the independent authorization
gate rejected release.  The source objective is therefore not enough to make a
planning claim.

V7 used `state_interaction_pairs`: a fixed candidate bank selected for a
single design-root set by high terminal-effect interaction and low mean effect.
That objective can leave a candidate that is globally attractive for the
declared goal on new roots.  A model can then learn a real interaction while
the task rank remains dominated by an action prior.

The sealed V7 forensic audit supports this as one real bottleneck: candidate
`8` won `19/32` action-only outcome roots, and only `4/16` candidates ever won
an outcome root.  The audit also prevents an overclaim: the privileged-state
arm did **not** have a positive lower paired interval over action-only.  Thus
the bank is concentrated, but the evidence does not say that fixing the bank
alone will recover visual decision transfer.

## Predeclared V8 intervention

V8 changes the *candidate-bank design* before fitting or opening new outcome
labels.  It does not tune on V7 outcome roots.

- Generate 512 new PointMaze roots from the immutable V23 base archive using
  seed `202610479`; only roots `10808..11319` are eligible.
- Use 64 disjoint design roots and split them deterministically into two folds.
- Select the 16 shared in-support candidate actions greedily by the worst
  fold's declared-goal winner entropy, winner diversity, best-second gap, and
  candidate cost spread; state-dependent terminal-effect interaction is only a
  later tie-break.
- Before any model fitting, require **both** design folds to have normalized
  winner entropy at least `0.50`, at least three distinct winners, at least
  `50%` of roots with a best-second gap of `0.003`, and a 10th-percentile
  candidate cost spread of at least `0.05`.  Failure is a candidate-bank
  diagnosis, not a license to tune the model.
- Fit 128 source roots; partition 192 calibration roots into 96 selection and
  96 authorization roots; retain 64 untouched outcome roots.
- Freeze the contrast weight, width, temperature, and history gate before the
  first outcome simulation.  The existing full grid remains explicit, while
  the larger authorization set reduces the selection-versus-certification
  variance exposed by V7.
- Require the same GPU contract as V7: a direct CUDA Python process, cuDNN
  disabled only for the incompatible fused RNN path, and a manually-unrolled
  `GRUCell` preflight.  There is no CPU learned-model fallback.

The panel remains an offline renderer-backed PointMaze experiment.  It cannot
by itself support a claim of live RGB deployment or universal world-model
transfer.

## Success and failure conditions

V8 has a positive repair result only if all of the following hold on the new
outcome roots:

1. the frozen policy is authorized on the independent authorization roots;
2. its paired outcome regret improves over action-only with a predeclared
   paired uncertainty interval excluding zero in the favorable direction;
3. the visual arm improves over both action-only and cyclic-history shuffle;
4. the privileged-state diagnostic does not reveal that any apparent gain is
   only a state-readout artifact.

If the candidate-bank headroom survives both design folds but the visual arm
still fails these checks, the next conclusion is not to add another memory
module.  It is evidence that the remaining limitation is visual
action-to-effect transfer rather than fixed-bank action-prior collapse.

## Host-GPU entry point

Run only from a host terminal where the CUDA preflight passes:

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
./.venv/bin/python -m experiments.decision_benchmark.run_pointmaze_counterfactual_contrast_v8_robust_cuda
```

The driver refuses completed output paths, binds all prior context identities,
and is resumable only from a sealed build or frozen-selection state with no
outcome label file.
