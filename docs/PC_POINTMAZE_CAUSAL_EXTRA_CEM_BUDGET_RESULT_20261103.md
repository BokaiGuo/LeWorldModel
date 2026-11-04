# Causal RGB planner: extra-random-search development result

The fixed protocol in
`PC_POINTMAZE_CAUSAL_EXTRA_CEM_BUDGET_PROTOCOL_20261103.md` executed both
planning arms on all eight preselected archive roots, 1888–1895. All 16
terminal RGB images and positions agree with separate physical replay.
Both methods used the same fixed model, task mask, feedback opportunity and
tail budget; random CEM used four rather than three initial generations.

| Root | Random-extra endpoint distance | Spectral endpoint distance |
| --- | ---: | ---: |
| 1888 | 0.0606 | 0.0315 |
| 1889 | 1.1686 | 1.1648 |
| 1890 | 0.2993 | 0.3435 |
| 1891 | 0.0433 | 0.0856 |
| 1892 | 0.0271 | 0.0168 |
| 1893 | 1.3201 | 1.1763 |
| 1894 | 0.1887 | 0.0832 |
| 1895 | 0.2082 | 1.2181 |

Under the strict offline step-10 distance `<0.5`, random-extra satisfies
6/8 and spectral 5/8. Spectral has lower endpoint distance on five roots
and higher on three. Both methods miss the threshold on roots 1889 and 1893;
on root 1895 only random-extra meets it. This negative case is retained.
Distance differences do not prove a spectral planning advantage or
reconfiguration mechanism, and the endpoint check is not certified online
native success.

Each random arm uses 64 initial and 48 tail samples. Each spectral arm uses
48 initial and 48 tail samples, plus two JVPs and one SVD. One extra random
batch is a **stress-test choice**, not a verified resource equivalence:
JVP/SVD and sample calls have different costs, warm-up affects timings, and
no common hard wall-clock or complete low-level compute cap was enforced.
The extra random batch can change CEM distribution and plan selection, not
merely provide an additional candidate on the same fixed bank.

These roots are not independent confirmation: root IDs appear in prior
project planning/tests, complete historical exposure and checkpoint
training membership are unknown, and this run was designed after seeing
earlier development results. The next step is to diagnose the shared failures
and root-1895 method difference with fixed-action/goal and interface checks,
then freeze an equal-resource contract and independently sourced roots.
Do not relabel this test as a P1 pass.

Artifacts:

- JSON `le-wm-main/results/a1_pointmaze_causal_extra_cem_budget_20261103_DEV01.json`,
  SHA-256 `9c8e00d5a5d1f1401108bbcfcd5dfda12658a9f7d49468189131a7db461bcb8c`.
- NPZ `le-wm-main/results/a1_pointmaze_causal_extra_cem_budget_20261103_DEV01.npz`,
  SHA-256 `a3bc25232e716953681293313a398248c97d764d7a8422447449c023d6094bc7`.
- Saved CEM proposal trace, RGB-state input, action, endpoint, and sample
  ledger test: **1 passed**; it does not rerun physics or neural inference.
- 16 ten-step executions plus 16 verification replays; random initial
  512 samples, spectral initial 384 samples, 768 tail samples, 16 JVPs and
  eight SVDs across this run.
