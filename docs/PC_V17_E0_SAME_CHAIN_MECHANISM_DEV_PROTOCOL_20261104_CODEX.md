# v17 score-augmented response: exact-Wall same-chain mechanism

This prospective development panel studies the same v17 core used in the
new Reacher feedback experiment, with exact-Wall task coordinates as the
chart and the nonnegative exact terminal distance as the legal planner
score. Set score scale 1 and score target 0. This is not a learned-model
or independently confirmed result. No witness trajectory is a planner
input.

Use eight tasks: start `[-0.5,y,theta]`, `y` in `{-0.32,0.32}`, `theta`
in `{-1.25,1.25}`, goal `[0.5,gy]`, `gy` in `{-0.25,0.25}`. Check
exact tuples against the explicitly enumerated prior v5–v16 and fixed-bank
panels; broader history independence is unknown. Each plan uses 15
actions, a 1,920-token cap, the existing two nominals, and seed
`2029201000+root`. Save every action, prediction and selection; audit
against alternate same-environment geometry before opening interventions.

After all plans are sealed, use the original v13 prospective mechanism
intervention implementation and thresholds unchanged: selected after
refinement/abstention, zero, same-bank greedy, ordinary lookahead and one
fixed random bridge; six common tails including the selected tail; three
480-sample replan streams per prefix; successor-response finite
differences at 0.01 and 0.05 with branch checks. Record absolute and
zero-tail-relative distances, all action batches, local response and
branch outcomes. Keep the base zero/greedy and full intersection distinct;
do not reinterpret failures by removing branch conditions. The control
search is diagnostic extra cost, not a matched end-to-end PC3 comparison.

All eight attempted plans and their interventions form the complete panel;
no retries, replaced roots, expansion, or threshold changes. Audit saved
intervention reconstruction separately. Any evidence is limited to this
task family, given finite optimization and unknown provenance. PC2 is not
automatically passed, and PC3/PC4/PC5 require separate evidence.
