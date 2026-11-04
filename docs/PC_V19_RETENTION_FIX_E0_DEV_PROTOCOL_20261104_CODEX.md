# v19 retention fix: exact-Wall development protocol

The v18 implementation reserved the runner-up reachability candidate if
the best was already in an immediate slot. v19 fixes only that retention
rule: when the best is already retained, remaining slots are assigned by
the existing diversity rule. The v18 results and code remain sealed.

Compare v19, v18, v17 and four-start shared-bank CEM on eight new tasks:
start `[-0.5,y,theta]` for `y` in `{-0.24,0.24}` and `theta` in
`{-1.08,1.08}`; goal `[0.5,gy]` for `gy` in `{-0.18,0.18}`. Check exact
tuples against enumerated recent panels; broader provenance is unknown.
Use horizon 15, identical global bank, legal actions, final nonlinear
score, and 4,095-token cap. All response and successor-probe calls count.

Record complete traces, token and per-call ledgers, actual retention,
selected chains and score. Audit common bank, all endpoints with alternate
same-environment geometry, budget and complete-chain selection. Report
success `<0.05`, distances and probe counts. Equal tokens do not imply
equal total costs.

Stop after one attempt for each arm on each of the eight tasks. No retries,
root substitutions or threshold changes. This is a bounded development
test of the correction, not a PC2 mechanism or independent PC3 result.
