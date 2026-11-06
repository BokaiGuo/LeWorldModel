# v22 native PushT constructed witness: development protocol

On fixed seed 2026110451, after the existing ten-action warmup, construct
one 15-action witness using the fixed privileged object-to-agent policy
implemented in the frozen witness runner. Replay the same actions once
from the identical reset and compare all RGB, proprioception and physical
states at 26 boundaries. A goal is nontrivial if object displacement
exceeds 20 pixels or angular change exceeds pi/9. Record the result once,
including failure; do not replace the seed or search a different policy.

The terminal RGB image is a candidate goal for a separate feedback
development diagnostic only if this witness is legal, nontrivial and
replays exactly. Physical states and witness actions are evaluator-only.
This is a feasibility construction, not a planner result and not proof of
training, normalizer or development independence.
