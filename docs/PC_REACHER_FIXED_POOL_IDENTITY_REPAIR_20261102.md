# Fixed-pool tail identity repair

The previously completed six-arm fixed-pool experiment already held the
CEM candidate population unchanged and evaluated the old tail separately.
Its report explicitly documented that three `selected_old_tail=true`
labels referred to parameter arrays exactly identical to the final CEM
selection: singleton and batch evaluation produced slightly different
float32 scores for the same plan. Repeating those known-root physical
trajectories would not add a new outcome.

The fixed-pool selector now detects exact parameter identity before deciding
whether a tail replaced the CEM plan. An identical tail retains the CEM
selection even when the separate score differs; the receipt records
`tail_equals_cem_parameters` and both original scores. This changes the
identity label in that case, not the action, candidate population or
number of model calls. It does not introduce a numerical tolerance for
different actions or declare two near-equal programs equivalent.

Ten fixed-pool, carryover and saved-witness tests passed, including a
regression case with identical actions and deliberately different batch and
singleton scores. The old report remains intact. No new neural inference,
physical episode, fairness result or P1 confirmation was produced.
