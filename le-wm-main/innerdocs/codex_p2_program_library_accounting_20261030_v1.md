# P2 program-library fixed source and resource axes

The prepare reconstructs the existing pinned four-case cp15 input and hashes
the checkpoint, model configuration, inherited source inventory and the new
kernel, CUDA adapter, structural saved consumer, journal and lock code.
This is a saved identity check: no model load or neural inference. The imported
source inventory is not a complete dynamic import closure.

Per case, caps are 70 packet, action encoder, predict, predictor and projection
calls (2 Jacobian-forward packets plus 34 per arm), one Jacobian request,
one SVD, and 34 scores. Packet and nested model entries are separate axes,
not additive compute. The meter records synchronous attempt/return/failure
with nesting and refuses retries. A real caller must supply a clock that
synchronizes CUDA, and wrap the actual model methods and score/SVD calls.
Failed durability must prevent dispatch.

This is not a runnable CUDA driver or a saved consumer of actual execution.
It does not claim a lock is held, model loaded, budget met or physics performed.
The future supervisor must hold the existing inode lock until the CUDA child
exits, preserving child failure and unscheduled slots. The true full dynamic
source closure, energy, FLOPs, model loading cost and backward engine calls
remain unknown. Known roots remain exploratory; scientific gates stay closed.
