# Fixed-bank affine lookahead physical oracle diagnostic

Replay all nine sealed prefixes and their own already optimized open-loop
five-action suffixes on each of the four exposed obstacle roots from
`PC_POINTMAZE_AFFINE_NONPROGRESS_LOOKAHEAD_PROTOCOL_20261103.md`. These
suffixes are the model-optimized normalized outputs saved *before* this
physical diagnostic, not new candidates or feedback results. Convert them
back to bounded physical actions with the recorded normalizer, cast to
float32 exactly as in the physical planner, then execute prefix and suffix
with the audited PointMaze position backend.

Report each prefix's predicted first and terminal scores, physical start,
cp5 and cp10 goal-position distances, and physical nonprogress relative to
the start. Separately report whether the model's predicted nonprogress
eligibility and physical nonprogress coincide. Compare eligible candidates
to the zero and model-greedy candidates' *same-bank own optimized suffix*
physical endpoints. An offline best candidate is an oracle ceiling and may
never be fed back into the selector or reported as a planning outcome.

Bind model-selection report and arrays, sealed bank and arrays, obstacle
states and physical backend receipt. Compare candidate-0 prefix cp5 and
other overlapping greedy/zero cp5 with the previously executed feedback
branches; tail actions differ because feedback replans. Keep all rows and
model-selection failures. This does not rerun the model, prove independently
optimized control capability, grant additional feedback, or establish
independent or equal-budget advantage.
