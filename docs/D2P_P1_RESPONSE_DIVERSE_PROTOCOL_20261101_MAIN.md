# Response-diverse initialization: fixed exploration

Follow the first-feedback-carry experiment, changing only the structured initializer.
Original eight DLS candidates are scalar multiples of one direction before gradient
updates. Saved root0/root1 initial banks have second singular values about 4.5e-8
and 6.5e-9. This is a search coverage limitation, not proof of mechanism failure.

New fixed eight candidates: zero; original normalized damped least-squares direction;
positive and negative first three right-singular directions of the same projected
H3 Jacobian, each with peak raw amplitude 0.5. One SVD, same eight common optimization
slots, same metered Jacobian. No additional neural calls for initialization.
Canonicalize each singular vector's sign before including both signs.

All four arms, both known roots, unchanged optimizer, actual feedback, node ledger
and first-feedback carry. Existing feedback mixture still retains four structured
initializers and four preseeded Gaussian ones; this is recorded, not an all-spectral
feedback claim. No forced non-greedy selection, no simulator labels, no outcome-picked
directions. Bind the new initializer source into structured contracts.

Success does not establish P1: inspect actual initial task changes, shared-action
interventions and own-planning results before any independent confirmation. This
development comparison cannot support independent generalization or equal-cost
superiority. Both negative and positive outcomes must be retained.
