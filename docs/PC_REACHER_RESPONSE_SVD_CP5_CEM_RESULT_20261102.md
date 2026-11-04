# Response-SVD all-prefix real-RGB feedback: development result

All 26 predeclared arms completed on two already-exposed Reacher roots. For
each root, all 13 saved prefixes (including parameter-zero) were executed,
then a suffix was independently optimized from the actual cp5 RGB and action
history and physically executed. Each arm used 16 CEM candidates for two
iterations. The saved journal consumer verified all arm identities, prefix
and zero-control action bytes, successor/goal RGB and history, 20 recorded
physical step intents and returns per arm, and 32 CEM candidate rollouts.
None of the recorded steps terminated or truncated. It did not independently
replay the simulator or certify all underlying model costs.

The summary below is exhaustive over the 12 non-neutral prefixes at each
root; it is descriptive and was calculated after opening the outcomes.
Distances are wrapped joint distances, lower is better. Parameter-zero is
zero in the model's action parameter chart, not necessarily zero raw action.

| Root | Parameter-zero cp5 | Parameter-zero final | Prefixes with lower final distance | Prefixes no better at cp5 yet lower at final | Prefixes with exactly the same optimized suffix as zero |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20263160 | 1.010031 | 0.546408 | 6/12 | 5/12 | 5/12 |
| 20263161 | 1.245444 | 0.598310 | 6/12 | 0/12 | 6/12 |

For root 20263160, prefixes 7, 8, 9, 11 and 12 were no closer than the
parameter-zero prefix at cp5 and finished closer after their own CEM
replanning. Prefixes 7–9 used exactly the same raw suffix as zero. On root
20263161 none of the prefixes simultaneously met those two descriptive
conditions. This is not a predeclared method-selection result: the frozen
strict model selector returned null on both roots, and the previous
same-suffix panel did not establish the full physical conjunction. Equal CEM
candidate counts do not imply equal total computation, and these two shared
roots do not establish statistical or training-data independence. The
result does not prove that reconfiguration caused stronger controllability,
that any model-selected method beat greedy/strong baselines, or that P1 passed.

Journal step counts include five prefix steps, five separately restored
parameter-zero control steps, and ten suffix steps for each arm; only the
prefix and suffix form the 15-step action plan. No training or CPU neural
inference was performed.

- Plan SHA-256:
  `f1087157d9cb4f6d4efdfc9900c74fed66a28f33169835e61093033c79077014`.
- Journal SHA-256:
  `e9c933f8d78abc9787198e324240afdb1186475fa08189a2787e39a3b9480f43`.
- Saved consumer output SHA-256:
  `20c412e0bfd3d7cc0710005af62892ec9ab805b76087c988917e2917b9f44207`.
