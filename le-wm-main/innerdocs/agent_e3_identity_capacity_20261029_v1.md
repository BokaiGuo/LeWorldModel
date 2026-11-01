# E3 identity binding capacity preflight

`load_bound(prepared, torch, device, store, prepare_sha)` calls the original
loader, computes the actual returned model's parameter inventory and exact
serialized binding-event length, checks the existing single-event limit and
147,465-byte identity event predispatch bound, then binds that same object.
It does not run model prediction or encoding. The 64-character placeholder
tag has the same JSON length as the generated lowercase hex tag. Original
Store limits, filesystem floor, wall clock and failure behavior remain in
force; the whole-job event check does not reserve future capacity.

This module is wired into the identity HOST candidate but has not been run
on the actual model. The existing saved MAIN prepare/report were
checked for a saved named-parameter inventory and do not supply one. Thus no
claim is made about actual checkpoint binding-event size. A binding event
that exceeds 16,385 bytes fails before its write; no field or cap is reduced.
Loading the actual model to obtain its inventory is a MAIN-only authorized
HOST action, not performed here.

Fixtures use the real Store and original loader control flow with fake model,
device and checkpoint. They compare the calculated size against the actual
event bytes, reject an oversized inventory before writing, and reject a wrong
device. No GPU allocation, neural or physical production occurred. The real
encoder two-node fixture, complete saved strict and host integration remain
unfinished; this artifact does not authorize production.

```sh
.venv/bin/python -m pytest -q tests/test_agent_e3_identity_capacity_20261029_v1.py tests/test_agent_e3_encoder_identity_20261029_v1.py
```
