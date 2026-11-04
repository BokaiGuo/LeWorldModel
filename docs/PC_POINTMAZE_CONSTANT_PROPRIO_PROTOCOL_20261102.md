# PointMaze constant-proprioceptive-input diagnostic

Reuse the four previously opened archive-start roots 1880–1883, frame-0 RGB,
frame-10 goal RGB, fixed eight-action bank and saved 64 two-packet physical
endpoints from the sealed archive alignment report. The released model needs
proprioception, which in this dataset copies simulator state. Replace both
root and goal proprio inputs with the **same fixed dataset normalization
mean** for every root, independent of episode state and outcomes. The
predictor may propagate its own predicted proprioceptive state; no actual
per-episode proprioception or simulator state is supplied to the model.

Score the same 64 candidate pairs and compare model selection to saved
position-only physical evaluation, as well as to the original
RGB-plus-proprio model score. Bind checkpoint, source report, dataset action
statistics and images by hash. Retain all scores and fixed-bank argmins.
Freeze model parameters and perform neural inference only on host CUDA.
Physical endpoints are read only after model scoring and used only for
offline evaluation. Dataset normalization constants are global assets whose
training provenance remains unknown.

This is an exploratory missing-sensor/distribution-shift diagnostic, not
proof of full RGB-only online planning, simulator replay equivalence,
independent generalization or P1/PC admission. Do not infer model dynamics
error alone from latent-versus-physical mismatch. If zeroing per-root proprio
changes action rankings, do not assume which input caused the physical error.
