# Q-consistent prefixes: common-suffix physical development result

The fixed protocol in `PC_REACHER_Q_COMMON_SUFFIX_PROTOCOL_20261102.md`
completed all 24 branches on two outcome-exposed roots. All branches
reproduced their previously saved prefix distance within 1e-8, completed
15 steps without stop, and all six own-tail terminal distances exactly
reproduced the prior physical report. All recorded step rewards were zero.
The frozen action sources and full physical trajectories are preserved in
`le-wm-main/results/reacher_pc_q_common_suffix_20261102_v1/report.json`,
SHA-256 `33b5b5a3bf53f2fb4fb3b24e0e630156d179e287d9edfe17e6e37c7b773823c0`.

All numbers below are evaluator-only qpos distances; lower is better.
The initial distance is shared by all arms within a root.

| Root | Arm | Initial | After prefix | Passive terminal | Selected-tail terminal | Greedy-tail terminal | Zero-tail terminal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20263160 | selected | 1.003500 | 0.984010 | 0.982176 | 0.604209 | 0.604209 | 0.460790 |
| 20263160 | zero | 1.003500 | 1.010031 | 1.008969 | 0.640825 | 0.640825 | 0.496129 |
| 20263160 | Q-greedy | 1.003500 | 0.982156 | 0.975343 | 0.617791 | 0.617791 | 0.472279 |
| 20263161 | selected | 1.182600 | 1.211151 | 1.209745 | 0.654526 | 0.636314 | 0.864397 |
| 20263161 | zero | 1.182600 | 1.245444 | 1.246073 | 0.697509 | 0.640773 | 0.906710 |
| 20263161 | Q-greedy | 1.182600 | 1.148837 | 1.139721 | 0.661658 | 0.539775 | 0.857527 |

Selected immediately improves qpos distance on the first root by 0.019490,
but worsens it on the second by 0.028550. This is a metric-specific
observation, not a blanket admission of a non-greedy mechanism.

Subtracting each arm's own passive terminal, selected's changes on root
20263161 are -0.555219, -0.573431, -0.345348 for the selected, greedy and
zero tails respectively. Zero's corresponding changes are -0.548565,
-0.605300, -0.339363; greedy's are -0.478063, -0.599946, -0.282194.
Thus selected's passive-adjusted effect is better than zero for two tails,
but worse for one; it is better than greedy for two, but worse for one.
Absolute terminal distance also does not consistently favor selected over
greedy. These small, deterministic, known-root contrasts do not establish
an independently useful capability advantage or P1 passage.

On root 20263160 the selected and greedy tails have different hashes but
maximum absolute action difference only 1.1921e-7 and identical displayed
terminal distances; they must not be treated as distinct independent
evidence. The three planned tails were already physically observed before
this protocol and cannot support independent statistical inference.
Common-tail outcomes and the previous independently optimized suffix
comparison are complementary; neither establishes equal total compute,
native success or independent confirmation.
