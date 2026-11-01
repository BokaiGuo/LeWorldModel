# Horizon-consistent Q result

All four arms completed20 physical actions with native success0/4. All16 nodes
passed horizon-config, no-winner-refinement, full-plan/action, feedback and exact
node ledger/model-hook checks. Ten implementation regression tests passed.

| Arm | Final jointXY | Planner predict calls | Planner VJP rows |
| --- | --- | --- | --- |
| structured | 31.09926 | 434 | 960 |
| native | 42.71007 | 427 | 192 |
| random | 155.65672 | 427 | 192 |
| immediate_progress | 50.78159 | 491 | 192 |

Counts are planner-node counts, NOT whole-system totals; RGB encoding, physical
operations and goal Q construction are separate. No fair-total-cost advantage.

The previous implementation optimized H3 then compared truncated H2/H1 tails.
The new variant aligns Jacobian/optimization/final replay with H3/H2/H1/H1
at boundaries0/5/10/15. This removed the code-level mismatch and reduced forward
calls, but did not produce success on this developmental task. Thus the mismatch
is not established as a sufficient explanation of failure.

All first-five-action prefixes remain unchanged and improve jointXY. Consequently
this repair did not create a qualified non-greedy mechanism.
Full16-slot comparison (four algorithm variants) is in
`le-wm-main/results/p1_return_variants_analysis_20261101_main_MAIN02.json`.
The next diagnostic executes all eight frozen structured candidates, to separate
candidate coverage from model ranking, without selecting actions by true outcomes.
P1 remains closed.
