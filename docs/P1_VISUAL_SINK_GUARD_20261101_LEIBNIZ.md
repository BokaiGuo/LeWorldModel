# Visual Sink begin/end adapter

This exclusive adapter targets `agent_p4_moving_multiboundary_20261028_v1`
`encode_boundary` and its actual
`p1_moving_live_feedback_physical_20261027_v1.Sink`. Before the callback it
charges one source-encoder call and one sample to the prospective guard;
the encoder and projector are suboperations, not two extra encoder charges.
The guard's encoder/projector pre-forward sentinels must both fire, in the
same order as the Sink's `module_attempt` / `module_return` and meter tokens.

During the callback the adapter constructs the original Sink with a narrow
meter proxy. Its actual `begin/end` methods still reach the original meter,
but only once each for encoder and projector, in order, with matching tokens.
Direct `meter.begin/end`, unknown resources, wrong tokens, missing module
hooks, double Sink construction, source drift and resource exhaustion fail
closed. Source-encode attempts remain charged on failure; original pending
meter operations are retained. Scoped process-local patches are restored in
`finally`, and the adapter serializes its own calls. The original
`encode_boundary` calls the handler's `torch.cuda.synchronize` before each
native module return; this adapter does not substitute a CPU implementation.

For a new host with an already source-bound guard and model sentinels
installed, replace the **call site** (do not patch `moving.encode_boundary`
itself) with:

```python
from experiments.spectral_reachability import p1_visual_sink_guard_20261101_leibniz as visual_guard
z, observation = visual_guard.encode_boundary(
    guard, handler, runtime, task, layer, row, obs, body, meter, store, step)
```

The guard contract's source manifest must include the exact adapter, guard,
legacy visual module, legacy encoder module and moving-boundary module source
files and hashes. The original `meter`, `store`, and handler must be supplied.
Do not additionally wrap this call in the earlier bridge's `guarded_visual`:
that would double-reserve and break nesting/sentinel expectations. This is a
new-host call-site integration interface, not a patch to sealed hosts. It
does not grant permission to bypass the earlier host readiness gate.

The tests exercise the actual Sink implementation and legacy meter with fake
module callbacks; no CPU model inference. They test the real call signature,
budget refusal before Sink construction, direct begin bypass, forged module
event, failed/pending operation, token mismatch, and missing source binding:

```sh
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p1_visual_sink_guard_20261101_leibniz.py tests/test_p1_total_guard_host_bridge_20261101_leibniz.py tests/test_p1_prospective_total_guard_20261101_leibniz.py
```

Remaining blind spots: Python can retain aliases to original methods, call
`forward` directly, or use other processes/functional CUDA paths; scoped
patching and source hashes do not prove whole-program coverage. This adapter
does not yet wrap goal construction or source acquisition, other Sink
stages, or all host visual call sites. Constructor/close, goal/state
operations, autograd and physics coverage, precommitted full-run caps,
wall/memory accounting, and source-time commitment remain unresolved.
No CUDA or physical experiment was performed here; `coverage_verified` and
full-cost/fairness certification remain false.
