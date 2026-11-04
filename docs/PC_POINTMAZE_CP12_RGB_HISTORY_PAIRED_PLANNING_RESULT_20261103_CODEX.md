# CP12 regenerated-RGB single versus three-frame paired planning

The prewritten regenerated-RGB protocol and implementation lock ran all 32
fixed exposed prefixes on the host NVIDIA renderer. The new RGB context
was sealed before optimization; both arms used the same context at cp12,
with three-frame additionally using the actual cp2/cp7 observations and
outgoing action history. All 64 CEM decisions were sealed before physical
suffix execution. Each execution independently re-rendered and checked
each of 13 prefix frames against that new context before its five commands.
All 832 execution-prefix frame checks passed; the 416 context frames also
matched the old saved frames exactly on this run. This does not authenticate
the old renderer's historical identity.

| Root | Single mean distance | Three-frame mean distance | Three lower / equal / higher | Single / three endpoint success |
| ---: | ---: | ---: | ---: | ---: |
| 8343 | 0.833248 | 0.838249 | 2 / 3 / 3 | 0 / 0 |
| 8345 | 1.557001 | 1.557734 | 1 / 2 / 5 | 0 / 0 |
| 8349 | 0.664156 | 0.663565 | 5 / 0 / 3 | 2 / 2 |
| 8351 | 1.672087 | 1.676990 | 2 / 2 / 4 | 0 / 0 |
| All 32 | 1.181623 | 1.184134 | 10 / 7 / 15 | 2 / 2 |

Distances are from a separate MuJoCo 2.2 position-backend replay to the
fixed position goal, not measurements directly returned by the native RGB
executor. Success here is the wrapper's strict final distance `<0.5`:
all paired success indicators match; it is not a native early-stopping
outcome or general success rate. The mean three-minus-single distance is
`+0.002511`; seven selected tails are exactly identical between arms.
Outcomes share four previously exposed roots and selected/adaptive prefixes,
so 32 branches are not independent trials and neither this near-tie nor
the descriptive counts establish a general control effect.

Each arm used 50 samples on each branch, matching seed/noise, zero raw
prior, bounds and optimizer; updates use scored projected elites. Total
candidate samples were 1,600 per arm, 256 scoring batches. Shared context
and goal encoding used 97 image/observation encodes; first-branch wrapper
versus explicit one-frame parity required 36 additional prediction samples.
The one-frame parity passed at both batch sizes 16 and 2. Model parameter
hashes before and after matched. Mean optimizer wall time per branch was
0.0862 s single versus 0.4538 s three-frame (synchronized CUDA timing,
excluding context/goal encoding and physical commands). The same candidate
budget is **not equal total compute**.

Budget: 384 native RGB prefix commands to construct context, then 768
replayed prefix and 320 tail commands for 64 executions; 1,472 native
commands total. The separate endpoint backend replayed 1,088 commands.
Context/plan/execution elapsed times were 12.95/21.02/25.16 seconds.
Event journals contain matching attempt/return counts; the saved-output
test independently checks random noise, candidate projection, elite updates,
tail bindings, score-based winner, position arithmetic and journal totals.
Focused tests: **8 passed**; they do not rerun CUDA or simulator calls.

The one-frame model path has a real wrapper parity check; the three-frame
action ordering has synthetic batch tests and the earlier fixed-action CUDA
diagnostic, not a separate independently implemented three-frame predictor.
The checkpoint's original training provenance, RGB-derived velocity and
render timing, legacy versus position-backend equivalence, and complete
transitive dependency closure are not certified. The GL identity was probed
in a same-environment subprocess, not inside the bridge. Scores use RGB
goal and causal RGB-derived state; simulator reset state is supplied only to
the environment, and position outcomes were opened only after decisions.

Receipts (each binds its same-stem NPZ):

- Context SHA-256 `00e7daa21352c53f168471f790591265cfc1342192191f033731ad8b7fa0880e`.
- Decisions SHA-256 `fedb164fdbc7d1f80421eea09510b9d27bd6aa2f601ab575d54547de28003e18`.
- Execution SHA-256 `b1477f951ea7d65339b7a5648e85000a81a6070ec5c32382a99953bcf7e45fa4`.
- Evaluation SHA-256 `697b0a41c98a469e5960bdb829ff072c28f1f1752fa8809bcf0191cd3e83a186`.
