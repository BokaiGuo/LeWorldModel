# Joint Vreach saved single-arm audit (intermediate)

`audit(q, arm, result, scalar_receipt, events)` validates the original scalar
native journal and its 98 forward / 16 scalar reverse resource ledger. It
checks the actual loss and gradient buffers, independently checks identity-Q
float32 loss against each saved endpoint and goal using the existing scalar
audit's reduction bound, and verifies twenty full H1 signal transcripts and
their actual 2-forward / 192-VJP resources. It then runs the unchanged joint
core against the saved callbacks and checks the complete result and method
event sequence, including the separately paid original-winner baseline and
all twenty candidate parents and action bits. No model is invoked.

This is a completed-arm audit only. It is **not** a host, authority verifier,
durable failure consumer, or proof of full model/source/precision closure.
Caller must authenticate the input task, source files, and event journal;
the signal and scalar callback streams must be in actual original order.
There is no P1 admission or physical release. The fixture uses arithmetic
native-ABI events and is not model or CUDA evidence.

```
.venv/bin/python -m pytest -q tests/test_agent_p1_joint_vreach_saved_audit_20261029_v1.py
```
