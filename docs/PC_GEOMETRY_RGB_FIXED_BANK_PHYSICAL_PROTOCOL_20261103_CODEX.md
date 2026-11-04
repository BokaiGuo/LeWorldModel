# Post-decision native physical evaluation

Only after the separate RGB model-decision receipt is sealed, replay all 48
saved model candidate chains, including every model-optimized suffix and
the same initial zero action used to obtain a second RGB frame. Use official
MuJoCo 2.1.2 persistent execution and the same four setup states. Record all
states, distances, contacts and the selected candidate per root, without
changing actions or choosing a new candidate. Evaluate terminal position
distance and prefix-end change relative to the post-warmup state. The
first zero action is explicitly checked against the zero-velocity reset;
if it changes state, prior zero-reset outcomes are not paired.

Model ranking against these physical outcomes is an exposed-development
diagnostic: previous privileged outcomes were already inspected and the
action-only bank includes a privileged direct prefix. It cannot be claimed
as a prospective or independent evaluation. Report each model-selected
prefix's passive/active results only from its matching fixed common-suffix
intervention and report its model-optimized terminal separately. Do not
claim the common-suffix result applies to the model-optimized suffix.
