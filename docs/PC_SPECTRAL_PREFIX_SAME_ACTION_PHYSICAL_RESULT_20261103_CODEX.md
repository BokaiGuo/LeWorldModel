# Sealed spectral-prefix bank: same-action physical follow-up

All 16 previously sealed prefix/suffix chains completed in the shadow
environment, with no action selection based on their outcomes. Each branch
replayed the same 12-action history before its own five-action prefix and
five-action suffix. All history images matched their sealed hashes. The
separate artifact audit verified 368 paired bridge attempts/returns (16
resets and 352 actions), RGB/action snapshot bindings and 16 trajectories.

| Exposed context | Model-selected chain | Initial position distance | After prefix | After full chain | Same-bank zero / greedy terminal |
| --- | --- | ---: | ---: | ---: | ---: |
| r8343_a0_p2 | greedy | 1.260430 | 0.760597 | 0.042210 | 0.705787 / 0.042210 |
| r8345_a0_p2 | spectral +0.5 | 2.131309 | 1.696242 | 0.981215 | 1.567720 / 1.041835 |

Distances come from the established evaluator-only MuJoCo 2.2 replay of the
exact saved actions, not from privileged state recorded by the original
MuJoCo 2.1.2 bridge. The two selected full chains happen to have the lowest
physical terminal distance within their respective fixed eight-chain banks.
The physical oracle is hindsight, not an admissible selector. Both selected
prefixes immediately improve position distance; in particular, the spectral
prefix on r8345 reduces distance by about 0.435. Thus the selected spectral
chain does **not** establish a non-immediate-progress reconfiguration.

Pairwise ordering disagreements among the eight chains were:

| Context | Time | Model vs actual RGB encoding | Actual RGB encoding vs physical distance |
| --- | --- | ---: | ---: |
| r8343_a0_p2 | after prefix | 0/28 | 0/28 |
| r8343_a0_p2 | after full chain | 1/28 | 0/28 |
| r8345_a0_p2 | after prefix | 1/28 | 0/28 |
| r8345_a0_p2 | after full chain | 0/28 | 0/28 |

All comparisons had 28 non-tied pairs. Pairs share a root and are not
independent. These data show substantially aligned ordering for this fixed
small bank, not a general resolution of model or scoring errors. The actual
RGB score includes an estimated velocity; physical labels come from a
separate replay backend. No native success or early-stopping condition was
measured; an endpoint distance should not be relabeled as a success rate.

The [protocol](PC_SPECTRAL_PREFIX_SAME_ACTION_PHYSICAL_PROTOCOL_20261103_CODEX.md),
[full receipt](../le-wm-main/results/pc_spectral_prefix_same_action_physical_20261103_CODEX/result.json),
and [artifact audit](../le-wm-main/results/pc_spectral_prefix_same_action_physical_20261103_CODEX/audit.json)
retain all slot-level scores, trajectories and replay endpoints. Three
targeted tests passed. These are exposed development roots; no common-suffix
causal intervention, per-prefix actual-feedback replanning comparison,
matched-total-compute strong baseline, or independent confirmation has been
established. P1 remains unpassed. The `experiment-code` workflow was used to
freeze the all-slot execution and preserve the outcome boundary.
