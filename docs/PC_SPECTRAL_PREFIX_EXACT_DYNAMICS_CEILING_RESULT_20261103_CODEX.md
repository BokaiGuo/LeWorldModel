# Fixed prefix bank: privileged exact-dynamics suffix diagnostic

Each of the 16 already exposed prefixes was followed by the same two-start,
100-evaluation CEM suffix optimizer using MuJoCo 2.2 position distance as a
privileged objective. The full reset and 17-action history were replayed for
each suffix candidate. Saved cp12, cp17 and original cp22 endpoints matched
the earlier physical evaluator. No physical information entered the RGB
planner or its already sealed decision.

| Context | Physically non-improving prefix | Initial → prefix-end distance | Optimized terminal | Zero / greedy optimized terminal |
| --- | --- | ---: | ---: | ---: |
| r8343_a0_p2 | spectral +0.5 | 1.260430 → 1.318696 | 1.328111 | 0.635680 / 0.006905 |
| r8345_a0_p2 | spectral −0.5 | 2.131309 → 2.220519 | 2.170049 | 1.527867 / 1.027486 |

These were the only two bank prefixes whose final prefix position distance
did not improve relative to cp12. Neither outperformed both controls under
this fixed optimizer and budget. The selected spectral +0.5 prefix on r8345
did outperform both controls after this privileged suffix search
(0.996289 versus 1.027486 greedy), but it also made immediate position
progress, so this is not a non-greedy mechanism witness.

The physical distances at cp12 and after each of the five selected-prefix
actions were:

- r8343 greedy: 1.260430, 1.197052, 1.115412, 1.008080, 0.882978, 0.760597.
- r8345 spectral +0.5: 2.131309, 2.078197, 2.001291, 1.906871,
  1.802399, 1.696242.

Both decrease at every recorded step. Thus the five-step selected prefixes
do not contain an observed early *distance worsening* hidden before later
progress. This does not exclude other forms of state reconfiguration, and
does not change the historically sealed five-step eligibility rule.

This finite-search diagnostic cannot prove that no more capable suffix
exists, that the task requires no reconfiguration, or that the same result
would hold on new roots. Optimized exact-dynamics suffixes are not available
to the deployed planner and were not physically dispatched. The result
argues against repeatedly tuning this particular two-root cp12 bank into a
mechanism demonstration; it does not establish P1 failure in general.

The [protocol](PC_SPECTRAL_PREFIX_EXACT_DYNAMICS_CEILING_PROTOCOL_20261103_CODEX.md)
and [full receipt](../le-wm-main/results/pc_spectral_prefix_exact_ceiling_20261103_CODEX.json)
include all 1,600 suffix evaluations, candidate traces and per-step
distances. Five targeted receipt/selector tests passed. The experiment-code
workflow fixed the paired suffix budget before evaluating outcomes.
