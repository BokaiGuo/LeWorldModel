# Affine-score nonprogress lookahead: exposed-root development

Use the previously sealed nine-prefix spectral/zero banks and obstacle roots
8343, 8345, 8349 and 8351. This is a new exploratory selector on already
exposed outcomes, not an independent confirmation. Use the frozen checkpoint,
RGB-derived initial and goal states, and the analytic affine-inverse position
score from the recorded affine diagnostic. No simulator state or physical
outcomes may enter scoring or selection.

For each prefix, use the same seeded AP-compatible CEM optimizer and 50
two-packet model samples to optimize one five-action suffix. The first
packet is fixed to that bank prefix; the second contains five individually
bounded actions. Record the first-packet affine score and both zero-prior
and optimized terminal scores for every prefix. Define model nonprogress
as first-packet affine score greater than or equal to the current RGB-derived
affine score, with no tolerance; zero is a control, not a selected prefix.
The model greedy control is the lowest first-packet affine score, with ties
resolved by bank index. Among eligible nonzero prefixes choose the one with
the lowest optimized two-packet affine score only if it is strictly better
than both zero's and greedy's optimized scores; otherwise report no choice.
Use stable bank-index tie resolution. Do not impose a physical nonprogress
criterion after observing outcomes.

Execute the selected prefix if any, plus the greedy and zero controls.
For each arm use its own actual cp4/cp5 RGB to reconstruct causal velocity,
then optimize a new one-packet suffix with the same affine score, optimizer,
50-sample budget and root-specific common random seed. Execute the five
planned suffix actions, for ten total steps. The physical goal distance,
cp5 status and terminal position are offline evaluation only. Save each
branch's complete actions, RGB history, per-batch candidates and scores,
selection decision, model hashes and actual planning calls. Verify the
prefix RGB against the sealed bank and physical endpoint against the
independent audited position backend. No selection on a root is a valid
outcome and must not be filled with a different candidate.

The selection costs 450 two-packet samples per root plus the sealed bank's
construction and, per executed branch, 50 one-packet feedback samples;
shared suffix budget does not make total method cost equal to greedy or
zero. The two-packet planning horizon is shorter than the desired full
feedback task and the comparison does not isolate spectral from generic
lookahead. Report this, the model score's calibration/normalizer provenance,
and lack of independent confirmation. Do not alter old results or gates.
