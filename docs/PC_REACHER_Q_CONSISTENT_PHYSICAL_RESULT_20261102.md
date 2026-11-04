# Q-consistent real-feedback suffix: development result

The v2 engineering-repair run completed on host CUDA on two previously
exposed Reacher roots. It executed the sealed selected, zero and same-bank
Q-immediate-greedy prefixes, refreshed each arm's own RGB and action history,
and used the same 16-sample × 2-iteration Q-CEM to plan its own ten-step
suffix. This is not a common-suffix intervention or an independent test.

The following distances are evaluator-only qpos distances; smaller is better.
They are not a substitute for the native success metric.

| root | arm | prefix index | after prefix | final | suffix progress |
| --- | --- | ---: | ---: | ---: | ---: |
| 20263160 | selected | 3 | 0.984010 | 0.604209 | 0.379801 |
| 20263160 | zero | 0 | 1.010031 | 0.496129 | 0.513901 |
| 20263160 | Q-greedy | 4 | 0.982156 | 0.617791 | 0.364365 |
| 20263161 | selected | 3 | 1.211151 | 0.654526 | 0.556625 |
| 20263161 | zero | 0 | 1.245444 | 0.906710 | 0.338734 |
| 20263161 | Q-greedy | 4 | 1.148837 | 0.539775 | 0.609063 |

Selected does not beat both comparators on either root: zero is better on
the first, Q-greedy on the second. All 90 recorded step rewards are zero;
this run does not separately certify the native success evaluator. The
report did not save initial qpos distance before each prefix, so it cannot
establish the physical non-immediate-progress condition. The different
optimized suffixes also cannot establish a same-action capability change.

All six arms report 15 physical step events, 10 returned suffix steps, no
termination or truncation, and 32 suffix candidate evaluations; saved prefix
action bytes match the recorded advanced-action SHA for every arm. The
frozen model parameter hashes match. This does not certify equal total
compute: upstream generation, goal-Q VJPs, encoding and setup are excluded.
The protocol does not measure wall time.

The failed v1 and its interface fix are retained separately in
`PC_REACHER_Q_PHYSICAL_RESTORE_REPAIR_20261102.md`. The v2 report SHA-256 is
`0671a015c0b850ab41fc170817199c2658b5a57e5c5418de4042ffc32dd2474b`.
P1 remains unproven; the two roots are development observations, not
independent samples sufficient for statistical inference.
