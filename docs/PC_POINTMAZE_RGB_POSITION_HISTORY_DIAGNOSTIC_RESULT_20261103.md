# RGB-position and actual-history prediction: fixed-tail diagnostic

The prewritten `PC_POINTMAZE_RGB_POSITION_HISTORY_DIAGNOSTIC_PROTOCOL_20261103.md`
ran using the frozen model on host CUDA. It evaluates three previously
executed tails for each of eight fixed prefixes from four exposed roots.
The two scoring paths use the same RGB-derived position and zero velocity;
the history path additionally uses cp0 RGB and the actual executed prefix
action packet. No new physical execution was performed in this diagnostic.

| Root | Prefix source | Single selected index | History selected index | Mean single MSE | Mean history MSE | Single-selected replay distance | History-selected replay distance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1884 | random | 1 | 1 | 0.01842 | 0.05022 | 0.8711 | 0.8711 |
| 1884 | spectral | 1 | 1 | 0.02115 | 0.02175 | 1.3450 | 1.3450 |
| 1885 | random | 1 | 1 | 0.08547 | 0.08405 | 0.0954 | 0.0954 |
| 1885 | spectral | 1 | 2 | 0.11823 | 0.09870 | 1.8319 | 2.4790 |
| 1886 | random | 1 | 1 | 0.08860 | 0.05507 | 0.3911 | 0.3911 |
| 1886 | spectral | 1 | 1 | 0.07059 | 0.05911 | 0.4209 | 0.4209 |
| 1887 | random | 1 | 1 | 0.07148 | 0.06124 | 1.6678 | 1.6678 |
| 1887 | spectral | 1 | 2 | 0.02754 | 0.06189 | 1.6885 | 2.2023 |

Indices refer to the source's ordered zero-state, RGB-position/zero-velocity,
and RGB-position/causal-velocity tails; duplicates are retained. MSE is
averaged over the three tails against their real cp10 RGB latents, over the
full visual tensor. Selection scores use the separate cp5 task mask and RGB
goal. Distances are position-backend replay evaluations of those already
executed tails, not native success. Two history selections changed, and both
have worse replay distance than the corresponding single-frame selection.
Five of eight group-average MSEs improve, three worsen; lower MSE does not
guarantee better action selection. The 24 candidate instances share prefixes
and roots and some actions duplicate; they are not independent trials.

This is an exploratory test of one two-frame token-binding hypothesis, not a
certification of the checkpoint's training-time history contract. Its
single-frame scoring reproduces the RGB-position selected tail as expected
on this adaptive, already executed bank, but the three-tail bank does not
stand in for a CEM search or an unbiased candidate distribution. Do not infer
that history is intrinsically harmful or that model theory is disproven.
Neither scores nor evaluation-only outcomes were fed back into a new
physical selector. Full action-history integration, equal-budget planning,
independent roots, native success and P1 remain unresolved.

Artifacts:

- JSON `le-wm-main/results/a1_pointmaze_rgb_position_history_diagnostic_20261103_DEV01.json`:
  SHA-256 `e14033e10cc0c17a1b7362e723c953f67fe71a833272086df365fece26a1388e`
- NPZ `le-wm-main/results/a1_pointmaze_rgb_position_history_diagnostic_20261103_DEV01.npz`:
  SHA-256 `523253b81c8c99567a7e669f0a50cbca3748bcdf036a49e9576b04fdfad7a474`
- `.venv/bin/pytest -q tests/test_a1_pointmaze_rgb_position_history_diagnostic_20261103.py`:
  **3 passed**. Tests cover action-token order using a fake predictor and
  saved source bindings, MSE arithmetic, and shapes; they do not independently
  rerun the frozen model or physical trajectories.
