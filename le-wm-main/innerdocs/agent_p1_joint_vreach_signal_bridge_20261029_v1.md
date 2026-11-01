# Joint Vreach: saved H1 signal bridge (partial)

`from_transcript(root_task, parent_context, transcript)` reconstructs the exact
parent-local task and H1 request, runs the existing complete native-call audit,
and extracts the full nominal context from the recorded nominal boundary. It
checks the zero-action path and the response terminal. It does not execute a
model, authenticate the original root authority, or establish that the parent
context came from a particular prefix. Those are responsibilities of the host.

The original audit checks the two actual forward trajectories and all 192
individual VJP, grad, and raw-backward witnesses, including the normalization
buffers and returned Jacobian. This bridge performs no second normalization or
Jacobian conversion. The test uses an explicit numpy native-event fixture; it
is not CUDA or model evidence.

This is an intermediate component, **not** the requested production host. It
does not provide prepare/run/consume, model or precision closure, full four-slot
authority, scalar AD auditing, durable failure accounting, or a 160-candidate
saved consumer. No production readiness or P1 admission is implied.

From the repository root run:

```
.venv/bin/python -m pytest -q tests/test_agent_p1_joint_vreach_signal_bridge_20261029_v1.py
```
