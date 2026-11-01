# dev303 dynamics-derived prefixes and RGB goal tangent metric

## Scope

This is a new development protocol. It does not change or supersede historical
gates. It uses no fitted readout, critic, adapter, or simulator state for planning.
The original proposal still requires same-chain mechanism evidence, independent
confirmation and fair strong baselines. None is established by this run.

## Implementation and model run

Four existing RGB boundary approach anchors were differentiated through the
frozen model. A goal-RGB tangent metric was constructed from the original goal
and ten fixed image augmentations using the frozen encoder. The leading right
singular direction of each weighted prefix Jacobian generated both signed
perturbations with fixed infinity-norm radius 0.25.

Four of eight spectral perturbations passed the unchanged development RGB motion
clearance check. Four were rejected and retained in the report. The accepted
bank contains four geometry anchors, four spectral prefixes, zero, direct greedy,
and random. This guard is an approximate development motion model, not certified
physical safety or contact.

For each prefix, a symmetric finite-amplitude response of a zero ten-step suffix
was measured in the frozen model. Damped least squares generated four bounded
suffix scales. Every resulting action chain was evaluated by nonlinear rollout.
The latent and RGB tangent metrics each optimized/scored a suffix for every
prefix. This is not an isolated score-only ablation and not a strong-optimizer
comparison.

The model selected geometry_3 under latent scoring and spectral_0_-1 under
goal-tangent scoring. Counts recorded by module hooks: 11 image encodes, 103
action encoder/predictor calls, 1,621 predictor samples; explicit arithmetic
counted 768 VJPs and 26 SVDs. Full cost certification remains false.

Model report:
`le-wm-main/results/p1_dev303_dynamics_candidates_20261102_main_RUN01/report.json`

SHA-256: `0293d12b2cbc5fd474a8fce89af5d57cab3ec07ef1c98d488b4d39416c434474`

## Physical results

All 11 candidates under both metrics were executed, for 22 branches. Each
completed five prefix and ten suffix actions, with no native termination.
Saved analysis verified the exact action sequences and 396 decoded frame
references.

| Preselected program/control | Metric | Terminal native reward |
| --- | --- | ---: |
| geometry_3, model-selected | latent | -380.5976 |
| spectral_0_-1, model-selected | goal tangent | -300.2440 |
| zero prefix | goal tangent | -364.2902 |
| direct greedy prefix | goal tangent | -310.9089 |
| random prefix | goal tangent | -376.8958 |

The selected tangent program's object terminal pose matches the non-pushing
branches. Its better native reward does not demonstrate useful object response,
nor non-immediate task progress: native task components and prefix progress
must not be conflated with object-only progress.

Physical report:
`le-wm-main/results/p1_dev303_dynamics_physical_20261102_main_RUN01/report.json`

SHA-256: `30a6524894147107d3fda7b8d4f1178322aa8b76d792bb5278d20ef1bff2b44e`

Saved analysis:
`le-wm-main/results/p1_dev303_dynamics_saved_analysis_20261102_main_01.json`

SHA-256: `811b3970cd4bfa24b19e1df5c525eefe47cdcf9dfeb2f15b965258458ee41b4d`

## Common suffix intervention

After the development panel, a separate contract fixed four prefixes: the
model-selected tangent prefix, zero, direct greedy, and random. Each received
the same five suffixes: all four corresponding model-planned tangent suffixes
and a zero suffix. No suffix was filtered by physical success.

All 20 branches completed. For selected, zero and random, all suffixes produced
the same terminal object pose as that prefix's own zero suffix. Greedy exhibited
small object responses. These results do not establish useful selected-prefix
object controllability. They concern this fixed suffix family, not every
possible suffix or every task.

Physical report:
`le-wm-main/results/p1_dev303_dynamics_common_tail_20261102_main_RUN01/report.json`

SHA-256: `2abcd1ad5726db3297b08a83c40bd92a6c64e91c788b7912f4f735aeb0b1b030`

Analysis:
`le-wm-main/results/p1_dev303_dynamics_common_tail_20261102_main_ANALYSIS01.json`

SHA-256: `457f9b96f1cef1327531e5876f5c8a22af37e42864693b474e50293f02767461`

## Next required experiment

Replace open-loop zero-suffix linearization with the same stronger nonlinear
suffix optimizer at each prefix's actual cp5 RGB/action history. Keep latent
and tangent arms and common initialization. A separate worker is implementing
that eight-arm CEM interface; no completed result is claimed here.

The leading prefix spectrum measures immediate model action response, not
automatically future task capability. The new code makes proposals genuinely
model-dependent, but the mechanism remains unproven. Candidate development must
ultimately respond to useful successor-state action responses.

P1 remains false. No independent roots, complete resource certification, or
strong-baseline advantage have been established.
