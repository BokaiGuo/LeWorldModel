# v13 E0 post-hoc coverage and retention diagnosis

All previously scored actions were independently replayed after the panel
stopped, with saved endpoints and scores checked. Counts below are repeated
scored trajectories, not independent samples.

| Arm | Full chains | Ever x > 0 | Roots with crossings | Best crossing distance | Successful chains |
| --- | ---: | ---: | ---: | ---: | ---: |
| v13 global + response | 3,874 | 18 | 8/8 | 0.7687 | 0 |
| v12 response | 3,874 | 3 | 3/8 | 1.1130 | 0 |
| Matched global + CEM | 5,120 | 27 | 8/8 | 0.6520 | 0 |
| Three-start CEM | 5,120 | 38 | 6/8 | 0.7913 | 0 |

None of the 32 selected chains crossed. In the recorded response search,
none of the 80 first-layer or 316 second-layer candidate prefixes crossed.
The third layer generated one crossing full chain in each response arm,
and neither retained it. All scored full-chain incumbents had already been
checked by the source audit: no successful full chain was discarded at final
selection. Crossing alone is not success, and a useful early prefix need not
cross. These counts do not establish whether pruning discarded a prefix with
a good optimized continuation.

The eight paired v13 and v12 response-layer records are exactly identical.
The implementation explains why: v13 chooses a global full-chain carry,
but v12 starts its beam at the empty prefix and builds response directions
around its fixed nominal actions. Carry changes the full-chain incumbent and
possibly refinement initialization; it does not seed recursive nominals or
beam prefixes. The resulting claim is limited to this code and panel:
the broad bank increased the number of scored crossing chains, but was not
used to explore the recursive response geometry around those global chains.
This is a testable design limitation, not proof that it caused all failures.

The predeclared analysis is in
`PC_E0_V13_COVERAGE_RETENTION_POSTHOC_PROTOCOL_20261104_CODEX.md`; source
hashes, per-root counts and recorded retention are in
`le-wm-main/results/pc_e0_v13_global_proposals_dev_20261104_codex/coverage_retention_posthoc.json`.
This is post-hoc same-environment consistency analysis, not independent
replication, a common-suffix retention experiment, or PC2/PC3 evidence.
