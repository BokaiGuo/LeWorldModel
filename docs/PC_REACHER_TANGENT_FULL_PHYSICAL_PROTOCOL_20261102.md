# Fixed tangent bank: full common-tail physical development intervention

Before any physical execution, bind the complete model-only tangent report
SHA-256 `d85a12443483e2268a0849ad72b6afccf3b68804a536df06bb0f780620f41589`
and its preflight. Both roots and all model outputs have already been exposed;
neither root has a model-selected eligible prefix. This protocol executes
all seven saved prefixes against all five saved common tails, 35 branches
per root, regardless of their scores or any physical outcomes.

Each branch restores the same reconstructed root snapshot and verifies root
and goal RGB. The saved float32 raw actions must have the expected shape,
remain inside the environment bounds, and agree with the saved normalized
actions and frozen action normalizer. Record initial qpos distance, every
subsequent physical state, reward and stop, and preserve incomplete branches
without imputing future values. Branches are not allowed to feed state or
outcomes back to the compiler.

Report all candidates and tails, absolute terminal distance and each tail's
change relative to tail index 0 under the same prefix. Tail index 0 is the
neutral *parameter* tail mapped into physical actions; do not assume it is
bitwise zero torque. Compare against prefix index 0, and separately report
the initial-to-prefix distance change. If an exception or early stop occurs,
retain its trajectory and denominator; do not select a replacement.

This is a known-root, outcome-informed debugging and mechanism diagnostic,
not a selected-policy outcome, a native success evaluation, independent
confirmation, same-budget planning comparison or P1 admission. The model
selector remains `null`; a promising post-hoc candidate cannot be relabeled
as selected. Fixed Q scalar scores and qpos task costs are different metrics.
