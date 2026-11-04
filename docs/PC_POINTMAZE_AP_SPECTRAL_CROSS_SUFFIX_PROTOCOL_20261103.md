# AP-compatible CEM prefix/suffix factorial: development diagnostic

Use all eight exposed roots and the sealed actions from
`pc_pointmaze_ap_spectral_cem_20261103_DEV01`. For each root cross the
random, spectral and zero first-five-action prefixes with the random,
spectral and zero last-five-action suffixes. The result is 72 fixed
ten-action physical sequences; the nonzero suffixes were optimized for
their original prefixes, not for every counterfactual prefix. Thus this
matrix measures response to common actions, not the best achievable or
equally replanned outcome from each post-prefix state.

Record initial, step-5 and step-10 physical positions and distances to the
same goal, plus action-dependent displacement and distance change relative
to the same prefix's zero suffix. Preserve absolute terminal results and
per-root pairings. Verify all original random/random and spectral/spectral
sequences, terminal RGB, and endpoints against the sealed source; verify
all endpoints with the second physical backend. No model inference, root
selection, optimization, or privileged simulator state is used in a
planner. Root state is used for resetting and outcome evaluation only.

The source outcomes were already observed before this diagnostic was
designed; roots and actions are development data. Interventions are not
independent trials and cannot establish confirmatory P1, equal-compute
advantage, a proof of non-greedy behavior or independently optimized
replanning advantage.
