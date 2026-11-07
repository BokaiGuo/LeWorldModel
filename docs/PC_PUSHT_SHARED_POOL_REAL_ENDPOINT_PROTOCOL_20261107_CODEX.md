# Shared-prefix pool actual endpoint encoding audit

On each of the four exposed PushT development tasks, rerun all 301 frozen
shared-prefix chains from their recorded initial state for 25 environment
steps. Verify action fingerprints, legal bounds, initial snapshot, five packet
states, first-success time and final success against earlier physical replay.
Encode the actual step-25 rendered image with the same frozen LeWM and compare
its goal cost with sealed single-frame and three-frame predicted endpoint costs.
Report per-task correlations, success/failure pair ordering, best successful
rank, and the success of each score's argmin. Keep "ever successful" separate
from final success. Write each task independently so interrupted runs resume
without overwriting prior artifacts.

Actual endpoint images and physical truth are evaluation-only oracle data:
they cannot be used for online action selection. This is a post-hoc diagnostic
on exposed tasks and fixed candidate pools, not independent evidence or a
closed-loop planner. Include all candidates and all failure denominators.
