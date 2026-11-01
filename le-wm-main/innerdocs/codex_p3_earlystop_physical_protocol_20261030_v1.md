# P3 early-stop physical replay preparation

Fix all 12 branches: budget points 20 and 30, both known roots and all three
arms, with a fresh environment per branch even for identical commands. The
selected commands are exactly the saved winner of each independently consumed
CUDA slot; include the actual and unused model allocation, root stop reason,
separate axis caps and actual calls in the source-bound physical prepare.
Do not reinterpret distinct maximum allocations as distinct executed sets.

Rebuild each CUDA saved consumer from its journal in a clean process before
preparing, bind source hashes, root and task, and bind the float32 bytes of
all ten actions. Existing environment adapter checks actual native action
bytes for each returned step and records state, RGB and native stop. Native
termination/truncation immediately stops that branch; missing later actions
are censored, never padded or counted as a completed endpoint. Other failures
are retained and fail closed. The two point outputs have six fixed branches
each; a partial report is not successful consumption. No inference, training,
root selection or retries occur during physical replay.

This is a post-observation, known-root exploratory replay. Separate physical
resource counts and model call axes; no equal-compute, P3 advantage, or
independent-confirmation claim. Dynamic dependency closure is not certified.
