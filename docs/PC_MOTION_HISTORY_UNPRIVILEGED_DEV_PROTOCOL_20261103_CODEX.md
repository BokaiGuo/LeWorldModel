# Common moving-history unprivileged RGB planning, development protocol

This is a new development protocol on the four previously exposed geometry
starts. It does not alter prior experiments or claim independent roots. Every
arm begins with the same 15 raw actions: five `(0.2,0)`, five `(0,0.2)`,
then five `(-0.2,0)`. Execute in the official MuJoCo 2.1.2 shadow RGB
bridge, one high-level action at a time; log every returned real RGB frame,
raw action, monotonic send/return timestamp and action index. The comparison
state is after action 15. No candidate or outcome can change this warmup.

Encode observed frames 5, 10, 15, deriving velocity at each from its real
immediate predecessor. Bind actions 5–9 and 10–14 (zero-based, five raw
actions each), chronologically normalized with the frozen dataset action
mean/std. Use window three and three future five-action packets. The
checkpoint configuration and code support this sequence-shaped input;
checkpoint training provenance is not independently confirmed.

Generate 11 prefixes *inside the runner* using the existing
`build_fixed_bank(2026110321 + root_index)`: zero, eight fixed directions,
two seeded random prefixes. No privileged direct or physical-outcome input.
Record generator code hash, seed, exact output, input RGB hashes, checkpoint,
normalizer, goal image, action history, model parameter hash and runner hash.
These are not yet spectral/model-generated proposals; the experiment tests
an unprivileged baseline path, not D2P advantage.

For each prefix, optimize a 10-action suffix with identical fixed CEM work:
two starts, three 16-sample generations, four elites, plus two final
evaluations per start, 100 full-program model predictions. Per root use
common random numbers with seeds `2026110340 + root_index`. Bound raw
actions to `[-1,1]`; use the same projected goal cost for every candidate.
Select minimum predicted full-chain cost with stable ties; record whether
the selected predicted prefix fails to improve the initial proxy and its
terminal score beats zero. Otherwise abstain from mechanism qualification.
Never select a different chain to satisfy the qualification. Save every
evaluation and complete decision before any selected program is dispatched.

After sealing all decisions, replay each warmup from the same reset and
verify every RGB frame matches the sealed history. Then execute the selected
five-action prefix and ten-action optimized suffix with the same bridge,
record all actual RGB and actions. Later native evaluator physical replay
is a separate post-decision stage. Model inference and execution are
open-loop after selection; this is not a feedback-replanning performance
result. A separate same-suffix intervention, per-prefix real-RGB
replanning, strong baselines and full cost audit remain required before
mechanism or fairness claims.

History diagnosis stopping rule: perform this one fixed moving-history
comparison on the complete 11-candidate bank and these four roots, using
same saved full action chains. Compare the supported real-history window
against the earlier single-frame approach and report cost differences and
physical ranking after physical evaluation. The single-frame mode is an
ablation, not a certified training-correct interface. Do not add or tune
history configurations based on which candidate wins. Repeated ranking
reversals in this fixed test are a measured model/evaluation limitation,
not a reason to keep searching context variants.
