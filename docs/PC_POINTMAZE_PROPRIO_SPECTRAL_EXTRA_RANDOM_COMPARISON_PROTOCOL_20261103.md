# Spectral versus extra-random CEM: retrospective cost stress test

Compare the already executed predicted-proprio-scored CEM trajectories on
all exposed roots 1888–1895: the 48-initial-sample spectral initialization
arm in `a1_pointmaze_proprio_spectral_cem_20261103_DEV01` and the
64-initial-sample random arm in
`a1_pointmaze_proprio_score_feedback_cem_20261103_DEV01`. Both arms use
the same saved goal RGB, initial state encoding, score, first three noise
batches and 48-sample real-RGB feedback tail; random also uses the fourth
saved initial noise batch. Independently validate both saved proposal
traces against the source noise and compare their saved actions, outcome
distances, observed-node timings and ledger fields. Verify source hashes,
goal/root identity, model identity, action reconstruction, saved endpoints
and optimizer counters before emitting a comparison.

Report separately initial predictive sample counts, two JVPs and one SVD
for spectral, encoding/planning and structure times, 48 tail samples for
both, outcomes and offline `<0.5` threshold. The random extra batch is a
pre-existing development condition, not a justified conversion of
JVP/SVD cost and not an equal-total-resource comparison. The arms were run
in separate executions; GPU warmup, hardware concurrency, target encoding,
rendering, timings and low-level forward/backward operations are not fully
controlled. Preserve ties and all roots. This retrospective comparison
cannot establish independent generalization, reconfiguration, P1 passage,
or a certified budget-matched baseline. No new model inference or physical
execution takes place.
