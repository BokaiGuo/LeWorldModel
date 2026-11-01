# dev303 native host bridge: partial implementation

`dev303_native_host_bridge_20261102_exclusive.py` connects the existing
full-horizon CEM callback interface to model `action_encoder`/`predict`, a
CP5 image encoder, two-packet actual action history, and step-by-step action
receipts. `native_callbacks` adapts the existing native PushT capture,
preprocessing, model encoder and `_check_return` to that interface. Its caller
must perform and authenticate reset and ten-step warmup, encode the three
source frames and goal, and persist all events; this component does not do so.
The initial RGB digest is mandatory and must come from the authenticated root.

The bridge meters outer attempted/returned calls and control steps. It does not
measure internal model calls, initial setup, synchronized CUDA time, FLOPs or
energy. Counters on exceptions indicate pending/failed calls and do not prove
that a native step had no side effect. Callers must stop after any exception.

This is **not an executable experimental release**: no complete pre-execution
contract, authenticated native launcher, source/model closure, lock management,
or independent saved-artifact verifier has been implemented. In particular, no
runtime authorization or evidence of an executed run is provided. The caller
must not use this bridge for a real experiment until those pieces are completed
and separately reviewed. CPU fixture tests exercise the callback seam only;
they do not validate CUDA, PushT, native rendering or model fidelity.

dev303 is a previously explored development root, not an independent
confirmation root. Full accounting and a matched comparison have not been
established; neither equal-budget claims nor comparisons to author-reported
values follow from this code.
