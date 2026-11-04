# Predicted versus real-RGB successor CEM: development diagnosis

The fixed protocol was executed on host CUDA for two previously exposed
roots and prefixes 0 and 12, with three paired seeds and two CEM budgets.
No physical actions or model training occurred. The source journal, model
parameter hash and all 24 ordered pair records passed saved-only checks.
The four original-budget/original-seed predicted tails and their recorded
model scores were reproduced exactly from saved contexts.

| CEM budget | Paired contexts | Maximum parameter difference > 1e-5 |
| --- | ---: | ---: |
| 16 samples × 2 iterations | 12 | 0 |
| 32 samples × 3 iterations | 12 | 6 |

The `1e-5` threshold is a post-run descriptive floating-point tolerance,
not a preregistered significance criterion. The larger differences are in
unconstrained CEM parameters, not measured physical outcomes. The paired
action-history discrepancies were at most `1.87e-9` in normalized values.
The full-context predicted-versus-observed latent MSE ranged from
`0.001456` to `0.002075` across the four root/prefix contexts.

The result shows that, for these exposed cases and fixed paired random seeds,
the real-RGB context can alter the selected tail when the CEM setting changes.
Both population size and iterations changed, so their separate effects
cannot be identified. The result neither proves that feedback improves
physical control nor supplies an independent root, mechanism intervention,
fair strong-baseline comparison, or P1 admission. The new tails have not been
executed physically.

## Discarded v1 and frame-binding correction

The first sensitivity run mistakenly passed already root-anchored saved RGB
frames through `_prepare`, which shifts frames before appending the root.
That repeated the shift. Its records are preserved at
`reacher_cem_context_sensitivity_RUN_20261102_v1` but must not be interpreted
as the intended experiment. V2 reconstructs `_prepare` arguments so its
output exactly matches the three saved frames; a synthetic frame-identity
test checks this. V2's exact reproduction of four original predicted
suffixes and costs provides an additional real-model check, not independent
physical replay.

V2 plan SHA-256:
`4ec03555ad0506f8a45861e83bb6b9563a902485d307e62771ea9204dba8296a`.
V2 journal SHA-256:
`77cdf094f5af43a4fa714b6281db868da3fff5acb3856df8b9efdeb684eb3590`.
The saved consumer validates hashes, journal chain, roster, CEM accounting,
history and original-prediction replay. Fourteen targeted tests passed,
including frame alignment, record tampering and previous physical-result
consumer tests.
