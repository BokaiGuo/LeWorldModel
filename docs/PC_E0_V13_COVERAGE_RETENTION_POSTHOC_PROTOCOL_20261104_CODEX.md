# v13 E0 fixed post-hoc coverage and retention analysis

After the fixed v13 panel stopped, analyze only the saved 32 traces and the
saved v12/v13 layer records. Do not generate, optimize, or reselect any
candidate. Verify source audit and hashes. Independently replay every saved
action batch and validate saved endpoints and scores.

For each arm, root, and horizon, count scored samples, wall crossings
(`x > 0` at any action step), terminal crossings, blocked trajectories and
the minimum terminal distance among crossings. Full-horizon successes use
the original `< 0.05` condition; partial-horizon distances are descriptive
and are not task outcomes. Record whether the selected full chain crossed.

For each response arm and depth, separately replay saved candidate chains,
check their saved feature and score, and count crossing candidates before and
after the recorded pruning indices. Record the root-level number with at
least one crossing before and after retention. These counts include duplicates
and do not identify useful prefixes: a crossing prefix need not support a
successful suffix, and a useful prefix need not itself cross. They cannot
determine the quality of pruned candidates under a common optimizer.

Report developer-panel results as post-hoc diagnostics, not a new experiment,
independent replication, a causal retention test, or evidence for gate passage.
