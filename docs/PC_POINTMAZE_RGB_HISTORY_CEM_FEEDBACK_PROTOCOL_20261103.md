# RGB-position feedback with actual-action-history CEM: development protocol

For the four already exposed roots, take both already selected RGB-position
initial packets from `a1_pointmaze_rgb_state_full_planning_20261103_DEV01`.
For each fixed packet, compare single-frame cp5 feedback with actual-history
cp0/cp5 feedback. Reset the native environment for each branch; execute the
same five high-level actions and verify cp5 RGB against the sealed source.
Both arms use RGB-derived position and zero velocity, the identical goal,
cp5 mask, task cost and sealed three-generation 16-sample tail CEM noise.
In the history arm, encode both actual RGBs, bind the normalized actual
executed prefix and each candidate tail with the validated action-history
unroll, using context window two. In the single arm, use the existing wrapper.
Each arm executes its own selected five-action tail in the native RGB
environment; seal action traces, RGB hashes and model/sample/time ledger.
Require the single-frame tail and resulting RGB to reproduce the source.
The offline position backend is evaluation only and is read after choices.

Report every root and prefix, including unchanged actions and negative
results. The fixed prefixes originate in previous development and are not
independent; RGB goal and camera metadata are unchanged; checkpoint
training provenance is unknown. Identical candidate counts do not establish
equal total compute: two-frame prediction and extra initial encoding cost
must be recorded. This is not a non-greedy mechanism intervention,
native-success benchmark, certified training-history contract or P1 gate.
