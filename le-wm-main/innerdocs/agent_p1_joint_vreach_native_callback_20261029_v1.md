# Joint Vreach native callback: intermediate component

`NativeCallbacks` connects the joint core callback ABI to the original scalar
backend and the existing H1 `Components` implementation. Scalar calls open
their own observation scopes; each H1 call uses a distinct parent-local task and
observation scope, and emits its events before returning. A completed H1 call
is independently checked using the saved signal bridge, including its full
nominal boundary and actual original native IO. The per-call resource vector
includes two predict API calls, one Jacobian, and 192 VJP/grad/raw-backward
callbacks. Scalar reverse calls are separate and are not included in that 192.
The scalar label prefix only identifies AD forwards for the existing audit.

The root authority and prefix provenance, scalar objective/AD audit, durable
event transport, total caps, source/model/precision guards and four-slot
prepare/run/consume are **not supplied by this component**. Its RAM factory is
fixture-only; HOST creation requires explicit authorization, CUDA device and a
caller-provided guard. This is not production readiness or P1 evidence.

RAM checks from the repository root:

```
.venv/bin/python -m pytest -q tests/test_agent_p1_joint_vreach_native_callback_20261029_v1.py tests/test_agent_p1_joint_vreach_signal_bridge_20261029_v1.py
```
