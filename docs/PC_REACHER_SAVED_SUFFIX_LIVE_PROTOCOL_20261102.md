# Saved suffix real-feedback development verification

Before execution, repeat the guarded spectral compiler on the two previously
used Reacher development roots, source-bound to
`results/reacher_p1_successor_cem_development_20261102_v1/report.json`.
The compiler now records each proposal's CEM suffix parameters. Compare its
bank, selected indices and costs with the old guarded-compiler report; any
drift is reported, not silently treated as an identical plan.

Run the saved-suffix physical feedback path for selected, zero and same-bank
greedy on both known roots. Bind to the archived physical report and demand
exact initial prefix and actual successor RGB/action-history identity. Verify
each saved imagined suffix by one additional two-token model rollout before
physical dispatch, then execute the existing carryover CEM rule with the
actual successor RGB and actions. Retain all actions and physical results.
This is a replay and engineering check, not a fresh sample or a test of
mechanism, statistical significance, equal total compute or P1.

The first phase uses host CUDA and the frozen model. Additional verification
inference, upstream compilation, model/observation encoding, physical steps
and wall time must not be excluded from any future resource comparison.
Existing selected and zero outcomes remain authoritative; no results or
gate criteria may be changed.
