# Reacher planned-chunk exploratory pair: RUN02

The host CUDA run completed both arms with the predeclared seed, model,
normalizer, thresholds and CEM configuration. Both receipts report matching
root RGB, goal RGB and warmup-action hashes; the full planned 15-step raw
candidate and normalized tokens are exactly equal. Full physical root state was
not independently authenticated. Each planned candidate contained 30 nonzero
action elements.

The fixed arm executed five candidate steps and made no extension predictions.
The dynamic arm accepted both five-step extensions and executed 15 steps,
with eight additional extension predictor calls. Both arms had six planning
predictor calls for 48 samples. Each returned all scheduled environment steps;
no termination or truncation was recorded. The observed per-step rewards were
zero, but these are not a success-rate measure or a task-quality comparison.
The arms have different physical horizons and different metered work.

RUN01 was blocked by the sandbox's unavailable NVIDIA device nodes before
model loading or physical action; its failure receipts remain intact. RUN02 is
a single known-root descriptive probe, not an independent task cohort,
matched-compute comparison, safety certification, P3 admission, or P1 pass.
