# Fixed-bank coverage, retention and continuation development protocol

Freeze before new outcomes. Reuse all eight exposed tasks and exact physics
from PC_WALL_RESPONSE_SIDE. This diagnostic is not RGB, independent evidence,
spectral contribution, or an equal-total-compute method comparison.

Horizon remains 16, prefix length 6. Bank: zero, followed in turn-major order
by 16 total turns (-pi + i*pi/8), each with speeds .15 and .3. Execute the
turn over two turn-only actions followed by four equal translations. Total
33 prefixes. No wall geometry, waypoint, goal or successful saved actions
enter this state-blind bank.

Fixed suffix family: zero plus ten turn-and-move suffixes, using total turns
(-pi,-pi/2,0,pi/2,pi), each with speeds .15 and .3. Each suffix is two equal
turn-only actions followed by eight equal translations. These 11 suffixes
are shared across every prefix and every task.

Freeze three retention lists of eight indices BEFORE optimized suffixes:
greedy ranks prefix distance; lookahead ranks minimum full-chain distance
over the fixed suffix family; random samples eight without replacement,
seed 2026111600+task. Ties use bank order. Lookahead is nonlinear model
lookahead, not SVD or a demonstrated control-capability metric.

Then independently optimize suffix for ALL prefixes, solely to diagnose
coverage and pruning. Same two-dimensional suffix parameterization (total
turn/pi and signed speed/.3), bounds [-1,1]. CEM uses two starts (0,0),
four rounds each, 128 or 256 total suffix candidates per prefix, std .5,
best eighth elites (at least two), std floor .02. Paired noise seed
2026111700+task is the same across all prefixes. All lists use the cached
identical suffix outcome for each prefix. Candidate outcomes cannot change
retention. Final arm selects the best complete chain ONLY within its list.
Global bank best is diagnostic only and must not be credited to an arm.
Use the fixed suffix-family best as incumbent, charged as 11 additional
evaluations per retained prefix for ALL arms (not an uncharged warm start).

Record per-budget selected indices, actions, prefix/terminal distances,
all-bank successful-prefix set (<.05), retained successful-prefix set, and
common-suffix effects relative to own zero suffix. Compare the selected
optimized suffix at selected/zero/same-bank immediate-greedy prefixes, and
each prefix's separately planned suffix. A limited witness requires selected
prefix distance >= initial-1e-12 and strict advantages in active response
and independently planned endpoint over BOTH controls. No claim of an
optimal suffix or real-physical mechanism.

Accounting: implementation always replays 16-command chains, including
prefix-only padded chains. Prefix screening costs 33 rollouts, lookahead
adds 33*11. Each arm pays 8*(11+optimization budget) continuation evaluations;
lookahead's retained 11-probe costs are reused, not double charged.
Report unique rollout counts for an arm's hypothetical execution separately
from the full diagnostic census and intervention replay. Unequal screening
overhead means this is not a fair total-resource performance result.
Persist sources, retained lists, full census, selected witnesses, traces,
budgets and separately labelled diagnostic executions.

If none succeed, preserve failures; do not tune the bank/threshold or use
oracle tails to select. Subsequent changes require a new development protocol.
