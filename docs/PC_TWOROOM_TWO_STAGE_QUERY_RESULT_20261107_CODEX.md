# TwoRoom two-stage query reproduction

Development-only reproduction on 40 previously exposed roots, using the fixed
RGB-only candidate generation, three retention rules, and frozen model.
Protocol: `PC_TWOROOM_TWO_STAGE_QUERY_PROTOCOL_20261107_CODEX.md`.

The implementation actually scores 65 A chains first, then eight B chains
for each arm, without reading other B scores. Per isolated arm this is 73
chains / 365 model packets; the combined three-arm run shares A and scores
89 chains / 445 packets per root without deduplicating B. These counts exclude
other costs. Stage timings sum to approximately 2.52 seconds for all 40
roots and all three arms on this run; this excludes loading, environment,
full-task feedback and other overhead, and is not a quality–cost curve.

Compared with the earlier 117-chain full matrix, the maximum A cost difference
was 0.000214. Maximum B differences were 0.000214 (lookahead), 0.000488
(descriptor), and 0.003602 (shuffled). All three arms reproduced both
retention and selected indices on 39/40 roots. On seed 1888821 the old A
minimum for prefix 11 was 13.25631905, versus 13.25632477 for several
competitors; in the new batching they tie at 13.25632477. Stable ranking
consequently selected prefix 1 rather than 11. These two prefixes are
elementwise identical; this root has only eight distinct prefixes out of 13.
Other retention slots also changed, so index equality is not established.

The selected chain in each arm succeeded on 40/40 roots according to the
previously recorded physical replay. This is **reused replay**, not a new
execution: the comparison checked root, all action arrays, model and normalizer
fingerprints, candidate-construction code, and old plan/replay hashes. The
new decision was sealed before the old plan and physical outcomes were read.

Two targeted tests passed: actual query scope/count and invalid action
rejection. No advantage over ordinary lookahead was found; the pool remains
saturated, and this does not show unseen-tail prediction, a mechanism,
independent generalization or RGB closed-loop benefit. No gate was changed.
Artifacts: `results/pc_tworoom_two_stage_query_plan_20261107_codex.json` and
`results/pc_tworoom_two_stage_query_comparison_20261107_codex.json`.
