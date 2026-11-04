# Model-only spectral-prefix and common-suffix development result

On each of two previously exposed cp12 RGB contexts, the frozen model
evaluated eight prefix programs (zero, one-packet greedy, four signed
Jacobian directions and two random programs). Each prefix received the
same two-start CEM suffix budget: 100 scored candidate chains in eight
batches. Terminal scores use the complete nonlinear two-packet rollout.

| Context | Initial score | Model-selected prefix | Prefix score | Terminal score | Reconfiguration |
| --- | ---: | --- | ---: | ---: | --- |
| r8343_a0_p2 | 1.271078 | greedy | 0.772118 | 0.016358 | abstain |
| r8345_a0_p2 | 2.151743 | spectral +0.5 | 1.662480 | 0.930167 | abstain |

In both cases the selected prefix is predicted to improve the proxy score
immediately. The second root's spectral program has a lower predicted final
score than its zero (1.523301) and greedy (0.982411) controls, but fails
the predeclared non-immediate-progress condition. Neither root has a
qualified alternative under the frozen model-only rule. A candidate that is
eligible but not actually selected cannot explain the selected chain.

This experiment establishes the implemented selection and abstention path,
not a physical outcome or an independent test. Projected RGB/model costs are
not true physical distances; the two exposed contexts were already used for
development. The suffix optimizer has equal samples across prefix candidates,
but prefix generation, gradient/SVD, greedy generation and other baselines
have not been equalized under a total resource policy. Each context used
800 scored two-packet suffix chains plus a separate 50-sample greedy search,
eight first-packet prefix scores, and one response forward/two reverse vectors/
one SVD. The checkpoint hash was unchanged.

The [protocol](PC_POINTMAZE_SPECTRAL_PREFIX_COMMON_SUFFIX_DEV_PROTOCOL_20261103_CODEX.md),
[runner](../le-wm-main/experiments/pc_spectral_prefix_common_suffix_dev_20261103_codex.py),
and [complete receipt](../le-wm-main/results/pc_spectral_prefix_common_suffix_dev_20261103_CODEX.json)
are separately saved. Two selection tests passed, including an eligible but
unselected counterexample and ties. No new action was dispatched. Further
work needs same-action model/physical checks, generated alternatives with
genuinely useful non-immediate-progress responses, and the real common-suffix
and independently replanned interventions. This result cannot establish P1.
