# AP-CEM core development: source and evidence boundary

The paper is Liu et al., *Aim Short to Reach Far: Your Frozen World Model
Can Plan Better Than You Think*, arXiv:2609.30036v4, 30 September 2026.
The full paper was retrieved through the arXiv MCP; Section 3 and Appendix
A describe the implementation below. An attempted clone of the candidate
repository failed both inside and outside the sandbox due to the host's
configured proxy; its contents and exact revision were not verified.
Consequently this is an independent paper-based implementation, not a
verified official code port.

## Implemented

- `pc_ap_observation_memory.py`: episode-bound observation-only records,
  adaptive `max(5,H-t)` retrieval, standardized source/endpoint/displacement
  key, 1e-4 standard-deviation floor and five-step successor target.
- `pc_ap_cem.py`: single-block CEM with 300 proposals, 30 iterations,
  30 elites, initial normalized mean zero and std 1/3, minimum std 1e-5,
  clipped prediction inputs and unclipped elite updates. It compares
  clipped final and initial means, using 9,002 predicted blocks in 31
  batches at default settings. A decision adapter connects retrieved target
  to a caller-supplied prediction callback.
- Episode IDs supplied for exclusion must not appear in the memory. This is
  a fail-closed identity check, not proof that supplied IDs or dataset
  provenance are genuine. No action memory is exposed by this interface.

Nine synthetic unit tests pass. They check the adaptive target, episode
boundaries, leakage rejection, invalid inputs, the default prediction
budget, clipping, the target-to-CEM connection and block/target time
alignment. They use simple numerical
callbacks, not a neural model, physical environment or GPU.

## Not yet demonstrated

No official memory/query manifest, checkpoint identity, image encoding,
action denormalization, physical action execution, closed-loop feedback,
memory-index cost, wall time or published-number reproduction is certified.
The predictor callback is deliberately caller-provided: neural use requires
an explicitly checked host-CUDA frozen model and instrumentation. Different
random number generators and implementation details may prevent bitwise
agreement with the authors even after integration. The paper reports memory
from training trajectories and excludes each evaluation episode; matching
that guarantee requires a verified provenance manifest, not just string IDs.

Next integration is to bind a certified existing LeWM CUDA adapter and
observation-only episode data, then run an exclusively saved development
smoke with actual RGB refresh and complete cost accounting. Published AP
values remain author-reported and must not be mixed with controlled results.
Neither these tests nor this implementation establish P1 or paper closure.

## Recorded-RGB CUDA development smoke

`run_pc_ap_reacher_cuda_smoke.py` now binds the decision to the frozen Reacher
checkpoint and its causal history. It feeds already-clipped normalized
five-action blocks directly into the predictor: the existing Reacher
parameterized planner applies tanh and would otherwise transform these
actions a second time. The adapter requires CUDA and records attempted
prediction batches and blocks.

With a fixed 16-candidate, two-iteration budget, the CUDA run used recorded
query episode 1 and 31 frames each from memory episodes 0, 2 and 3. It
retrieved episode 0 at step 5, planned 34 predicted blocks in three batches,
and preserved the frozen parameter hash. V2 reads the Reacher action bounds
directly from its environment; V1 and V2 produced identical decisions.
Reports are saved under
`le-wm-main/results/pc_ap_reacher_cuda_recorded_smoke_20261102_v{1,2}/report.json`.

This checks one decision on archived frames, not execution or re-observation.
The tiny memory is not the official AP memory, episode-to-training membership
is unknown, and encoding/retrieval costs are not yet certified. The 9,002
block paper budget was not run here. A real feedback-loop experiment remains
necessary.
