# P2 per-prefix feedback with common multi-start budget: development design

This is a tested planning kernel and draft execution contract, not a sealed
physical experiment or a claim of completed feedback. The old physical
executor runs an already fixed 25-action sequence and cannot pause, retain
and return actual cp5 RGB for 32 candidate/arm trials. A new authenticated
split-execution adapter, host launcher, saved consumer and pre-execution
source/plan seal are required before dispatch.

For each of the two exposed development roots (existing cases 0 and 2), both
arms receive the same 16 fixed float32 10×2 initial actions in the same order.
Use the same projected-gradient implementation, eight updates and nine
scores. Stage one evaluates a two-packet model objective: conditional masks
the first-packet gradient and joint updates all 20 coordinates. Both incur
two transition calls per score and one explicit gradient call per update.
The joint method is not historical SRA. Initial candidates are shared, but
the eventual candidate prefix can differ between arms.

Each candidate/arm then pays for a fresh physical reconstruction and executes
its own selected five-action prefix from the identical root and guarded
common history. Only three actual RGB history frames and normalized public
action history enter the CUDA encoder; native state, reward, task metric and
success score do not enter the selector. Capture receipts must bind root,
task, candidate, action bytes, RGB files/pixels, time and native stop flags.
The kernel checks the shifted action history and prefix normalization; hashes
alone do not authenticate a capture or prove the context came from the RGB
encoder. The required future adapter must establish that binding.

For each actual cp5 context, both arms run the same eight-update/nine-score
one-packet suffix optimizer. The deployed first five actions are frozen.
Each arm selects its minimum cp5-model score across all 16 candidates, with
candidate index then iteration as deterministic tie-breaks. The comparison
is a preplanning intervention with *both* arms receiving actual cp5 feedback;
it is not the earlier open-loop conditional-vs-joint GN test, nor a pure test
of feedback versus no feedback.

Per root, successful paired planning requires 288 preplanning scores, 256
explicit preplanning gradients and 576 packet transitions; 32 physical
prefix captures and 32 RGB-history encoder invocations; and 288 suffix
scores, 256 explicit suffix gradients and 288 packet transitions. Counts
are for both arms together; each arm gets half. The encoder call processes
three RGB frames: the internal model token operations are unknown. Physical
execution needs 32 separate candidate-prefix trials per root, plus two fresh
winner-prefix reconstructions and two suffix executions; all warmup, common
prefix, renderer, encoder, environment, replay and stop costs must be
separately metered. Exact CPU physics steps depend on the environment and
must not be inferred from public step counts.

Before executing a winner's suffix, replay its root, warmup, common prefix
and five-action candidate prefix, compare literal actions and the saved
actual cp5 RGB/action history, and fail closed on any mismatch. The current
kernel does not implement this replay. Any candidate native early stop,
capture/encode failure, or invalid receipt aborts the whole root with its
full denominator and partial attempts; there is no candidate replacement,
outcome-based filtering, padded trajectory or winner from a partial set.
Root failures and later-not-started roots must be recorded by the future host.

Equal candidate counts, score/gradient calls, transition calls and RGB
invocations are **not equal FLOPs or equal total costs**: gradient dimensions,
actual execution paths, GPU synchronization, reset/render, inherited bank and
model load differ. Synchronized inclusive timings may overlap and cannot be
summed as GPU kernel time. Autograd internals, FLOPs, energy, and full source
closure remain unknown. These are exposed roots, not independent confirmation,
and no P2 gate or general composition claim is implied.
