# First-pool prediction / actual encoding / physical alignment

This retrospective diagnostic uses the 40 exposed tasks and the first-round
3-by-3 prefix/probe bank from the sealed RGB feedback and physical replay.
Reconstruct the same initial rendering, goal image, raw 25-step actions and
physical endpoint images. Replay all 360 full chains and compare every
distance and success flag against the existing physical record. The replay
continues after success only to match the sealed 25-step endpoint.

Using the same frozen RGB checkpoint and action normalizer, re-score the nine
chains per task and verify each score against its sealed model score within
`1e-3` absolute tolerance. Save predicted endpoint feature vectors, actual
endpoint RGB encodings and target encodings. At each matching root, prefix,
probe, compare hold-relative effect signs for (a) model-predicted goal score,
(b) actual encoded endpoint goal score, and (c) actual physical endpoint
distance. A sign is zero within `1e-6` in its own units; report ties and
task-level sign counts. Also report feature mean-squared error and absolute
score error, without equating latent score units to physical distance.

This is a same-backend, exposed-data diagnostic. Actual endpoint images and
physical state are evaluation-only and never enter action selection. It does
not independently prove a model bug, the reliability of the full planning
distribution, hold semantics outside tested states, or any gate.
