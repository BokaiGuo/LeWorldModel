# Residual prefix intervention and same-action model audit

These results concern one previously used dev303 PushT development root.
They do not modify the original P1 admission or establish generalization.

## Common suffix intervention

The sealed 20-arm contract crosses four prefixes (spectral selected, zero,
greedy and random) with four previously fixed cp5 latent-planned tails and
one zero tail. No physical success filtering was used to choose actions.
Every branch completed the fixed horizon. An independent saved-data pass
verified the contract, ordered roster, all action traces, and 360 PNG and
pixel hashes.

For the selected-generated tail, selected versus greedy prefixes yielded
terminal object position errors 276.055 versus 239.046, and angle errors
2.247 versus 2.833 radians. Relative to each prefix's own zero suffix,
the changes were +39.417 versus +2.385 for position and -0.804 versus
-0.221 for angle. Thus the prefix changes the response of this particular
action, with opposing position and angle effects; it does not prove a
uniformly better task response, increased independently planned capacity,
or native success. Greedy and selected prefixes already repositioned the
pusher, so object immobility alone cannot certify no immediate improvement
under a whole-task metric.

Contract SHA256:
`8fb3a5215a0e427f86f4c9d96466cd74b42626215443e07e46601d776fc0384f`.
Physical report SHA256:
`1794d41d1f107964b4e033d0a164ab705c4de2f1e3e1b00a80046e8e8ac5d9d8`.
Analysis SHA256:
`a9bff4e6f1e17e19c5cc1c174009892a9ad94cb3fac36fe8f966b9de4645058c`.
Saved verification SHA256:
`5fae27f990a3f5054d95cd894fb839949a65928f61eb271395d3426979608571`.

## Same-action prediction audit

On the eight already sealed feedback trajectories, the frozen CUDA model
was evaluated using the actual executed actions and the exact corresponding
RGB frames. Refreshing the first predicted packet with its observed RGB
encoding reduced second-packet latent prediction MSE in 7/8 arms; in the
greedy-latent arm it increased from 0.1283 to 0.1621. This is a post-outcome
diagnostic on dependent arms, not an independently selected policy or a
guarantee that more feedback improves control.

For selected versus greedy with latent scoring, terminal predicted goal
distances were 1.662 versus 1.722, while actual RGB latent distances were
1.772 versus 1.855: both ranked selected better. Yet physical reward ranked
greedy better (-291.535 versus -369.291). That pair therefore exposes a
task-metric mismatch even where predicted and observed latent ranking agree.
It does not exclude additional action normalization, model or timing errors.
The diagnostic report includes the full five latent vectors per arm and is
sealed with SHA256
`2671ffc19e41ccd91aaf3f62f84d239f390d789dc2a2031f4946966e8f1d3452`.

For each scoring group, a descriptive pairwise comparison was also made
between the four executed arms, using unweighted terminal latent MSE in
both model prediction and actual RGB encoding, and native reward as recorded:

| Planning group | Model vs actual latent inversions | Actual latent vs reward inversions |
|---|---:|---:|
| latent | 0/6 | 4/6 |
| goal-tangent | 4/6 | 4/6 |

These are dependent trajectory pairs from one root, not independent samples
or an estimate of population error. The goal-tangent group uses *unweighted*
latent MSE for this post-hoc cross-group comparison, not its selection metric.
Ties were not observed in these values. This comparison also does not isolate
reward's velocity component or establish which physical objective to optimize.

The next distinct investigation is time-aligned candidate-oracle and
task-metric assessment, together with common-budget baseline execution and
fresh-root provenance. Existing results and thresholds remain unchanged.
