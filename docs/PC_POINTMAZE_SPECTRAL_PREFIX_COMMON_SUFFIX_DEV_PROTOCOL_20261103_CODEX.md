# Spectral prefix with common multistart suffix: offline development

Use the two already exposed, saved decision-0 shadow RGB contexts from
`pc_pointmaze_shadow_closed_loop_dev_20261103_CODEX.json`, frozen model and
goal. No physical simulator state or outcome enters generation or selection.
This is an offline planning diagnostic, not a physical or independent result.

At each context form eight five-step prefix programs: zero; a greedy program
obtained from the existing 50-sample one-packet CEM; the four signed
task-Jacobian spectral proposals from the existing component; and two random
raw action packets drawn uniformly in [-0.5,0.5], with root-index-derived
fixed seeds. Spectrum uses the same raw/normalized chain and amplitudes as
the sealed spectral component. Verify action bounds and log every candidate.

For each prefix use the identical pair of CEM noise tensors and identical
two-start, 100-sample suffix optimizer to score the two-packet nonlinear
model rollout. Compute prefix score at the actual first-packet time, not at
the end of the second packet. Save full action witnesses, candidate scores,
costs, optimizer traces, seeds, and actual counts. The projected objective
is the existing frozen RGB-derived position proxy, not true position.

For ordinary planning pick the best terminal candidate, stable ties by bank
order. A *model-only qualified reconfiguration* requires that this same
selected prefix be spectral or random, its predicted first-packet score is
not lower than the initial score, and its predicted terminal score is strictly
lower than both zero and greedy with their independently optimized suffixes.
Otherwise report abstention. Do not relabel a distinct eligible-but-unselected
candidate as the executed mechanism. No candidate is physically dispatched
under this protocol. Both contexts and the scoring rule were previously used
in development. This is not a matched-total-budget baseline or admission.
