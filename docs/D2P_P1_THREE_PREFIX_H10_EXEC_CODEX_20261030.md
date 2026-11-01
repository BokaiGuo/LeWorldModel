# Three-prefix H10 executor and saved verification

The deployment prepare rebuilds the previously frozen 1008-row design and
binds the new executor, consumer, censor proof, tests and this document. All
24 anchors and 42 literal ten-step probes per anchor are scheduled in order.
The run is exploratory, on known roots after outcomes were observed; P1,
freshness, independence and equal-budget certification remain false.

Each branch constructs a fresh native environment, reconstructs the original
reset and warmup, binds its original target, replays its prefix and checks
all 51 frozen source frames and snapshots before probe dispatch. Actions,
transitions, RGB frames, resource intent/returns, native stop and censor proof
are durably journaled. A proven probe-stage out-of-canvas guard rejection
retains the returned transition and first image but not an endpoint. Native
stops are never padded. Other failures stop dispatch, retain the current
shard's 42-row roster and leave later shards unstarted; a crash or storage
failure may prevent a complete report and is not a valid observation.
The saved consumer accepts only all 24 independently checked shard reports,
while preserving the 1008 denominator and null endpoints for censor/stop.

Per-shard caps are 42 constructors/closes, 84 resets, 1050 public steps,
10668 physics steps and 3402 explicit renders. Storage is capped at 512 MiB
per shard with 32 MiB final reserve, 512 MiB free-space floor and 3600 seconds
per shard; whole-run admission reserves 24 shard caps plus the floor. Actual
resource counts must be read from the saved ledger. Historical shared and
proposal costs are unknown and not treated as zero. Source checks do not
certify unrecorded host behavior or simulator solver caches.

CPU fixtures simulate environment I/O and cannot establish real physics
behavior. Before a main-thread execution, independently review code/tests,
check the frozen hashes, runtime and available disk, ensure non-interference
with other work, and use a fresh exclusive output directory. No neural calls
are made; there is no external-signature or extra authorization-token gate.
