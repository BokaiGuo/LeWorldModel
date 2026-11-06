# Saved v13 retention versus common suffix search

The sealed prospective v13 mechanism panel already contained one
119-sample, four-knot CEM lookahead search for each of the ten first-layer
response prefixes on each of 12 roots. This post-hoc analysis matched those
prefixes and saved chains to the compiler's actual layer and compared the
best recorded suffix score in each subset. It did not execute new search,
prediction, or simulation.

| First-layer subset | Roots losing strictly to all ten | Mean final-distance gap | Roots with recorded success |
| --- | ---: | ---: | ---: |
| All ten candidates | 0/12 by definition | 0 | 0/12 |
| Four actually retained | 2/12 | 0.000055 | 0/12 |
| Four lowest immediate scores | 5/12 | 0.000358 | 0/12 |

Strict gaps for the actual retained set occurred on roots 4 and 9, at
0.000140 and 0.000521. All ten-candidate best distances were between
0.5002 and 0.5031; the success threshold is the source protocol's 0.05.
The zero-prefix continuation is recorded separately in the raw output and
was excluded from these ten-candidate comparisons.

This result supports a narrower diagnosis than “the retention rule
discarded a successful continuation”: no such continuation was found by
this finite optimizer in the examined first-layer bank. It does not prove
that none exists, that a more capable optimizer cannot find one, or that
all full-method search failure comes from suffix optimization. The v13
final selected plan can be refined beyond the original layer candidates;
the limited root-9 same-chain mechanism witness is therefore not
contradicted. This is not a fair end-to-end performance comparison:
optimizing every prefix was extra diagnostic work.

The source plan and intervention audits and hashes were checked. The
candidate action order, retained indices, saved optimized-chain prefixes,
and 119 proposals per candidate were validated. Twelve exposed development
roots, one optimizer configuration and one stream do not provide
independent confirmation; PC2 and PC3 are unchanged.

Protocol: `PC_V13_SAVED_RETENTION_SUFFIX_DIAGNOSTIC_PROTOCOL_20261104_CODEX.md`.
Script: `le-wm-main/experiments/pc_v13_saved_retention_suffix_20261104_codex.py`.
Raw rows and input hashes:
`le-wm-main/results/pc_v13_saved_retention_suffix_20261104_codex.json`.
