# P2 predict-frontier v2: R1/R2 exception evidence, not a control repair

NEW independent source/tests/protocol. Do not modify/reseal v1. Main reported
v1 successful normal execution; v2 exception handling is not an explanation
of those control outcomes, nor evidence that normal behavior improved.
No new outcomes are read by this implementation. No neural/physics run here.

## Unchanged scientific object and budgets

Same four known root identities, original contexts/goal0/Q/model/normalization,
joint H2 whole20 ball sqrt(10)*.75, tanh raw box, original damping .002.
Frozen bridge supplies genuine80 joint proposals and original40 conditional
prefixes; no outcome-based redesign, padding, retry or candidate caching.
Fresh shared nominalH2(2)+ADforwardH2(2)=4; each arm160 search+4 winner replay.
Standalone168/168, paired unique332/root1328total; same act-encoder count.
Standalone AD1 vs41/root,192 vs7872 mathematical VJP directions; paired164 J,
31488 directions. Actual CPU SVD/solve count inherited, not a FLOP match.
Immediate array transfer/serialization/fsync costs are NEW overhead, never free
or credited as saved inference. Model-load, root, compile/generation/selection
times remain reported. Upstream RGB/context/Q cost unknown, not zero.
P1/P2 admission remains false; no physical efficacy or total-compute claim.

## R1: first exception is the primary

Faults retains actual exception objects in arrival order and serializable stage,
type and message. Re-observing the same exception does not duplicate it. During
unwind, pending forward/SVD/solve failure is recorded BEFORE invoking a journal.
A broken failure/terminal journal is secondary; it cannot replace that pending
exception. Without an active exception, journaling failure stops work before any
next call; failed attempt persistence prevents dispatch. Hook restoration tries
EVERY installed hook individually. Restore failures become secondary if a
primary exists; otherwise first restore failure is raised after other restores.
Outer transaction ultimately rethrows the SAME primary object, including
KeyboardInterrupt/SystemExit, after best-effort final evidence. No silent retry.
Restoration failure means a hook may remain installed: abort experiment, do not
reuse the model. In-memory fault log is retained even when a journal is broken.

## R2: derivative arrays and independent finalization

The new primitives retain exactly the old differentiable recurrence, no prefix
detach in the shared joint Jacobian. Nominal is converted/saved/fsynced BEFORE
Jacobian begins. On Jacobian return it is converted/saved/fsynced BEFORE its
completion journal or ANY ridge/SVD solve. Each suffix does the same, including
zero-cap suffixes (nominal and AD still charged). 82 files/root: shared nominal/J
and40 suffix nominal/J pairs. Each carries root/index, root-context SHA, original
goal/Q SHA, protocol/prepared SHA; suffix adds full predicted context/prefix SHA.
Array receipts plus returned/converted/persisted stage acknowledgements survive
later solve failure. A save whose receipt journal fails can already have a durable
file; a false stage acknowledgement does NOT assert that the file is absent.
No new neural calls are introduced. Files are exclusive, fsync(file)+fsync(parent).

After exclusive output creation, one execution envelope covers execution.json,
imports/CUDA allocation/model load/initial model hash, source reads, roots,
sync, row writes/hashes and report construction. A partial row retains actual
entries, operations, array receipts, stages and fault chain. Row commit failures
do not prevent final source/model checks; aggregation uses retained in-memory
rows, not potentially missing row files. Source verification and model/frozen
verification each run independently and write separate finalization artifacts.
One failing check/write cannot skip the other. Model unavailable is failed, not
unchanged. Report is attempted even after loading/sync/check/hash errors.

On clean completion, report.json is followed by completion.json binding its byte
SHA. Any observed execution/finalization/write error prevents export and raises
the primary; an emergency finalization_failure.json is attempted. Export requires
clean fault logs, both separate passed artifacts, completion hash, original
all-slot80/40 guards/witnesses/Q/replays AND all82 immediate array identities,
values/hashes/stage receipts. The old40 exporter is never used.

Hard limits: pre-directory validation/preflight failures need not create output;
SIGKILL, process death, OOM-kill, hardware loss or exhausted/broken storage cannot
guarantee a final report/emergency receipt. If both report and emergency writes
fail, existing stage files/partial journals remain; do not infer success. Disk
preflight requires310MiB free (110MiB conservative artifact allowance plus200MiB
reserve), not a measured hard disk quota. Never download/delete to meet it.

## Commands (repository root le-wm-main)

```sh
.venv/bin/python -m pytest -q tests/test_p2_composition_predict_frontier_cuda_20261027_v2.py
.venv/bin/python -m experiments.spectral_reachability.p2_composition_predict_frontier_cuda_20261027_v2 --prepare --output results/p2_composition_predict_frontier_cuda_20261027_v2_prepared01
.venv/bin/python -m experiments.spectral_reachability.p2_composition_predict_frontier_cuda_20261027_v2 --verify --prepared results/p2_composition_predict_frontier_cuda_20261027_v2_prepared01 --prepared-sha256 SHA
# MAIN ONLY after review; not run by implementation thread:
.venv/bin/python -m experiments.spectral_reachability.p2_composition_predict_frontier_cuda_20261027_v2 --execute-cuda --prepared results/p2_composition_predict_frontier_cuda_20261027_v2_prepared01 --prepared-sha256 SHA --output results/p2_composition_predict_frontier_cuda_20261027_v2_main01
```

Pure mocks validate fault mechanics, not empirical efficacy or actual CUDA
equivalence. Main's successful v1 result remains its original sealed contract.

Implementation validation:58 new pure mock/fault tests passed; bridge/v1/v2
joint regression137 passed. This includes mocked-torch (NumPy-backed, no real
torch operation) equality of both new primitives against sealed old math,
82-array root/goal/Q/context binding rejection tests, and stage ordering before
AD/solve. Source/model checks still run after row commit or report-builder failure;
failed summary aggregation reports unknown totals, never invented zero resources.
