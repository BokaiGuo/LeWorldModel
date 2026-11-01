# Fixed-prefix saved group audit

This is a non-production, single-group saved-transcript component. It replays
the frozen fixed-prefix method from all 19 native forwards and eight reverse
calls, checking native ledger, literal tensor IO, control events and ordering.
It independently checks eight scalar losses against the saved endpoint and
goal, using float32 subtraction/squaring and a float64 reduction with the same
3e-5 relative reduction bound as the existing independent scalar audit. This
does not establish gradient correctness, model parity, P1 admission, or physical
release. It does not execute a model.

It is not the requested 64-group host: strict parent authority reconstruction,
source/model/precision closure, durable failure accounting, group scheduling,
prepare/run/consume CLI and resumption semantics remain to be implemented and
reviewed. The fixture uses a synthetic linear backend and is not production
evidence. Do not run CUDA or physical experiments using this component alone.

From the repository root: `.venv/bin/python -m pytest -q tests/test_agent_p1_fixed_prefix_saved_audit_20261029_v1.py`.
