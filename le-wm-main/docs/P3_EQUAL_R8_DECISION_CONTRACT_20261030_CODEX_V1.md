# Exploratory P3 matched R8 feedback contract

This new contract is informed by the main-thread report that four earlier
slots requested reconfiguration; it is not preregistered independent
confirmation. Old failures remain failures. The previous progress-only gate
and frozen artifacts are unchanged. P1 and scientific admission remain false.

At each of four observed feedback boundaries, compute the original H1/H3
signals, spectral diagnostic and original adaptive decision. Only an original
`continue=True`, `program_type=extension_type=reconfiguration`, width 8 and
horizon 1 or 3 is supported. A progress, composition, stopped or malformed
decision fails closed; none is reinterpreted as R or progress.

Before choosing an arm's horizon, call the existing actual
`agent_p3_r_native_orchestrator_20261029_v1.evaluate_R` for both H1 and H3.
Each group uses the spectral8 variant, eight proposals plus a separately paid
zero, actual per-candidate postcontext H1 signals and bounded suffix
evaluations. It is an exploratory R implementation, **not** an established
equivalence to an original R optimizer or a physical benefit. Evaluate both
groups even when the first has no witness, and stop without a packet unless
both have witnesses. In each group select the eligible row with minimum
`(immediate model cost, candidate index)`, as the original tree's beam rank
does for siblings. Replay each winner both whole and segmented and require
exact equality with its previously produced path. Only then route fixed H1,
original signal horizon or reverse horizon, and send exactly its first five
raw actions; re-encode returned actual RGB before the next decision.

Per *completed* decision: 106 predict API calls, 3840 VJP directions, 20
Jacobians/signals, 40 replays, 20 program compiles, 22 array SVDs, 18 solves,
36 metrics, one schedule. Axes overlap and cannot be added as FLOPs. Full six
slots/four decisions imply 2544 predict API calls, 92160 VJP directions and
960 replays, excluding feedback encoding and physical work. A failure or
no-witness stops the slot with the attempted partial bill; no padding,
pretended completed budget or replacement. Equal measured vectors apply only
to completed decisions, not failed work, equal latency or equal FLOPs.

The independent module implements real R group calls and callback
packet/encode feedback and includes a host factory that creates the existing
native CUDA `NativeTree` with the R session's own call sink, bound to the
active adapter root and guard. A synthetic RAM native fixture exercised
both actual R groups and exact 106/3840/40 counts; this is not GPU inference
or simulation evidence.

## Still required before any host execution

The seam does not provide a production driver, durable R-specific journal,
per-call immutable capsule implementation, resource/storage/wall admission or
saved consumer. The old V1 journal terminal and saved consumer do not accept
this R schema; old adapter-produced numeric chunks and R Session transcripts
need a reviewed single-owner mapping with no double-writes and explicit source
anchors. Provide real caps sufficient for much larger R transcripts, bounded
records, failure/poison/cleanup tests, and independent saved reconstruction
and cross-log authentication. Require the active adapter guard around each
native invocation, identity/precision checks, verified host resource
coordination and a fresh separate plan and output. The callback ABI alone
does not certify that GPU/physics hooks cover actual work. No execution or
training is authorized by this document.
