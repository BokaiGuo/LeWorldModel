# AP-CEM vs final-goal CEM: real-RGB development protocol

Development only, fixed before the run: Reacher roots 20263160 and 20263161,
which have been used previously; target delta (0.65,-0.65). Frozen original
Reacher checkpoint and previously used action normalizer. From the same
seeded warmup/root and goal, run two arms: observed-target AP-CEM and
final-goal CEM. Both use 16 candidates, two iterations, four elites,
identical per-root/per-decision search seeds, two decisions and five actions
per decision. The goal remains fixed; both observe actual RGB and actual
executed action history between blocks. This is not the paper's full CEM
budget, official query set, official memory, or an independent test.

AP memory is the first 31 recorded images from Reacher episodes 0, 2 and 3;
each record is bounded within its episode. These archive episodes and the
live reset seeds are not asserted to be training-independent. The dynamic
retrieval horizon is `max(5,25-executed_actions)`; after each block the same
memory is queried again. The baseline uses the same predictor and CEM but
scores against the actual rendered final goal. The goal offset 25 is a fixed
retrieval-control input here, not a claim that the synthetic rendered goal is
exactly 25 recorded actions away.

Execution uses normalized action bounds derived from the environment and
inverse checkpoint normalization, then checks physical clipping. A recorded
environment state is used only by the wrapper to construct the goal and for
post-hoc evaluation; it never enters either planner's candidate selection.
Record root/goal/successor image hashes, actions and per-step task distances,
termination/truncation, actual predict counts, encode frame count and elapsed
time. Runs with any exception or premature stop remain explicit failures,
not completed-horizon scientific outcomes. Encoding/index construction and
planning costs are reported separately; full compute fairness is not claimed.

No success, mechanism, independent-generalization, official AP replication,
or P1 admission claim follows from this engineering pilot.
