# Three-prefix own planning and common H10: actual results

## Execution

- Own suffix planning: both known roots × selected/zero/library-greedy, six host
  CUDA runs, each completed 20 physical steps. Native success 0/6; no row error.
  Reports: `le-wm-main/results/p1_three_prefix_own_host_20261101_main_root{0,1}_{selected,zero,greedy}_RUN_MAIN01/report.json`.
- Common H10: both roots × three prefixes × nine fixed suffixes, 54 physical
  branches, completed without top-level execution errors. All prefixes replayed
  their saved cp5 state, dynamics and RGB exactly. No neural inference in the
  saved-action physical intervention.
- Saved paired analysis verifies equal b0 candidate banks and scores across
  the three roles, exact prefix selection rules, and executed action prefixes.
  It is not a complete independent reimplementation of the physical simulator.
- New component tests: 3 passed (analysis metrics and own-prefix selection).

## Main finding: no selected-versus-greedy separation

Both roots have exactly identical selected and greedy first-five action arrays
AND identical full twenty-step executed arrays. They must not be counted as
independent successful mechanisms or evidence of advantage over greedy.

Immediate error changes, selected and greedy (negative means improvement):

| Root | Pusher XY | Object XY | Wrapped object angle |
|---|---:|---:|---:|
| 0 | -78.26956 | -52.89041 | -0.311802 |
| 1 | -15.47867 | -0.18324 | -0.001757 |

These are descriptive separated physical quantities, not a replacement gate.
In particular the current selected prefixes are not established non-immediate-
progress mechanisms. Root1 zero-prefix also moves passively: object XY error
changes -0.42781, so selected-versus-initial alone is insufficient attribution.

## Shared suffix outcomes

Root0 selected and greedy each terminate with candidate2 at step15 and
candidate3 at step14; zero has no terminating branch. Root1 selected and greedy
each terminate with candidate4 at step15 and candidate7 at step7; zero terminates
with candidate7 at step12. No branch is marked truncated.

These are correlated branches of TWO known roots, not independent samples.
The suffix library comes from the selected cp5 model context, so is asymmetric.
Terminating observations are retained separately; missing fixed-horizon endpoints
are not filled. The original physical runner labels terminated endpoints null,
including step15; the analysis conservatively retains that policy.

The six independently replanned trajectories have no native success, despite
successful fixed suffixes existing in the intervention bank. This narrows the
next investigation to candidate selection, objective horizon and feedback
replacement; it does not prove which one is causal, nor prove all model
prediction or task-scoring issues resolved.

## Next correction and claim boundary

Before any independent confirmation, a revised method must produce an actual
selected/greedy distinction without using physical outcomes to select candidates.
Check non-linear full-plan scores and saved-tail replacement under real RGB,
charge all scoring, and compare identical feedback opportunities.

Do not choose the successful intervention suffixes after seeing their outcomes
and relabel them model-selected successes. Retain this negative mechanism result.
P1 remains closed; source independence, training provenance and fair strong
baseline confirmation remain incomplete. Existing full-plan carry tests are a
separate method variant, not the own-suffix method tested here.
