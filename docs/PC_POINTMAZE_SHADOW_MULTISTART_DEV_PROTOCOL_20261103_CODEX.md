# Additional-compute CEM control for exposed shadow roots

This new development protocol supplements, and does not alter, the completed
paired spectral run. Use the same two exposed roots, saved 12-action prefix,
checkpoint, five action/re-observation cycles, RGB-only contexts, CEM
optimizer, action bounds and durable bridge contract. Each root is reset
independently; the control uses only its own observed RGB.

At each decision run **two** ordinary 3×16 CEM searches from the same zero
prior and same context. The first uses the original CEM noise. The second
uses independently generated CPU float32 noise with seed equal to the first
eight little-endian bytes of SHA-256 of the original noise bytes prefixed by
`pointmaze-cem-multistart-v1\0`, masked to 63 bits. Score both independently,
choose the lower final model objective, breaking ties toward the first. The
extra computation is spent on actual candidate search rather than discarded
Jacobian work. Save both CEM traces and scores, both chosen plans, the seed,
actual predictions and decision time before dispatch. Execute only the
selected first action.

Each decision therefore uses 100 prediction samples in eight batches and no
Jacobian/SVD, versus the existing spectral arm's 50 samples/four batches
plus one response forward, two reverse vectors and one SVD. These are
different resource types; this is an intentionally stronger-sample budget
point, **not a certified equal wall-time, FLOP, energy or total budget
comparison**. Record actual decision time, all model-call counts, frame and
action hashes, failures and censored states. Replaying saved outcomes is
evaluation-only. Existing spectral results and their historical source code
remain unchanged; the new baseline uses a fresh output directory.

This is an exposed-root follow-up, not a scientific confirmation, a proof of
spectral advantage, or a test of reconfiguration-prefix discovery.
