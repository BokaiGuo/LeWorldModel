# Fixed-candidate physical decomposition on three exposed failures

Development diagnosis fixed after observing the preceding result: roots
1889 and 1893 (both methods fail the endpoint threshold) and 1895 (random
passes and spectral fails). For each of their two arms, replay every saved
initial CEM candidate as its **full ten-step open-loop plan**. Separately
replay every saved tail CEM candidate appended to that arm's **actually
selected five-step prefix**. Flatten all CEM generations in saved order;
retain ties and duplicate candidates. No candidates are selected or generated
from physical outcomes. The original tail candidate bank is adaptive to the
real cp5 observed for that arm.

Before outcome computation, verify original source/report/array hashes;
replay the saved CEM traces with their actual innovations and require the
same selected initial plan, tail and score. Require the physical endpoint
of the selected tail candidate to exactly match the sealed ten-step
endpoint position. Use the audited position-only backend and source reset
float32 conversion. Evaluate position distance to the original frame-10
target, candidate minimum and number strictly below 0.5, the original
model-scored choice, and pairwise model-score versus physical-distance
reversals. Record all candidates, scores and endpoints. The full-plan
oracle and fixed-prefix-tail oracle address different questions.

These are post-outcome oracles with simulator state unavailable to a
deployable planner. Outcomes cannot be used as controller inputs or
independent P1 evidence. If no successful tail exists, it proves only
insufficient coverage in the saved tail bank at that fixed prefix; it does
not prove that prefix has no usable continuation. Report ties and dependent
candidate pairs. No training or model scoring is needed for replay.
