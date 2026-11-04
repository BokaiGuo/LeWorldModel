# v13 prospective same-chain mechanism development protocol

## Scope and frozen plan

Evaluate the unchanged v13 global-proposal compiler on 12 exact-Wall tasks:
the Cartesian product of start `[-0.5,y,theta]`, `y ∈ {-0.34,0.06,0.36}`,
`theta ∈ {-1.1,1.4}`, and goal `[0.5,gy]`, `gy ∈ {-0.31,0.27}`.
Check exact disjointness against the explicitly referenced v5, v10,
waypoint, v11–v14 panels. Broader history, checkpoint, and normalizer
provenance are not established; this is not independent confirmation.

Each task receives one v13 plan, 15 actions, 1920 five-step prediction
tokens, fixed nominals and `seed=2029131000+root`. The generator and all
plans are sealed before interventions. The same-environment second geometry
implementation audits every scored action, time/score and final
complete-chain incumbent. Failures, abstentions and all attempted roots are
preserved. The planned selection must not consume intervention results.

## Controls and intervention

Use the final selected chain *after* refinement and abstention. Define the
same-bank control set as the zero prefix (present in the global bank) plus
every actual v13 first-layer response candidate. Rank greedy by the
five-step terminal task distance; for lookahead apply the same 119-sample,
four-knot CEM to every bank prefix with the same `2027111400+root` stream,
and select its best final chain. Bank ties use first occurrence. This is
additional diagnostic work, not a free formal-planning baseline.

Use the original v5 mechanism intervention implementation and thresholds:
five prefixes selected / zero / greedy / lookahead / one fixed random bridge;
six shared tails including the selected tail; three 480-sample equal-budget
replanning streams per prefix; 0.01 and 0.05 finite-difference successor
responses at the same clipped selected-tail nominal, with branch-stability
checks. The original fixed seeds and nonprogress, matrix-gap, goal-gradient,
and strict relative-benefit predicates are unchanged. Normalize only the
selected candidate-index metadata in the local-response adapter, as in v2;
the underlying actions, scores and response values are unaltered.

Save absolute distances and zero-tail-relative changes for every common
tail; all independent replanning action batches and scores; response
matrices, goal gradients and branch events; and all source/diagnostic hashes.
Report both the base intersection against zero and greedy and the full
intersection against all four controls, component failures, all 12 roots
and failures. This is a predeclared development intervention, not a PC2
pass by itself: applicability, preserved original contract, selection
validity and provenance must still be assessed. It does not test PC3.

After the 12 plans and one prescribed intervention per completed plan,
stop this panel. Do not replace failures, extend roots, adjust rules or
choose favorable seeds. Audit deterministic reconstruction separately and
report it as same-code consistency, not independent replication.
