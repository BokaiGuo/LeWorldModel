# v11 interface development status

This record concerns interface implementation and synthetic tests only. No E0
or JEPA scientific panel was reopened, and no physical or GPU experiment was
run. v10 and its historical dependencies were not modified.

The v11 candidate separates model packet length from search segment length;
segments must contain whole packets. Its prediction-token counter uses the
actual submitted batch and physical length. Invalid segment/packet and carry
shapes fail closed. On a new observation, zero and a supplied remaining tail
are evaluated together and eligible as the full-chain incumbent; refinement
starts from the best full chain. The final selected chain comes from the paired
incumbent/zero-prefix comparison.

An independent JEPA v11 single-decision worker and isolated launcher now call
this candidate. The response arm checks the compiler's ledger against the
existing actual-predictor meter. They have not yet been exercised on the real
model or connected to an independently audited physical feedback collector.
The earlier v7 feedback collector is not a v11 collector and must not be used
to claim this connection without a new, preregistered protocol.

Synthetic regression command, from `le-wm-main`:

```
.venv/bin/pytest -q tests/test_pc_recursive_response_v10.py tests/test_pc_recursive_response_v11.py
```

Result: 12 passed. This includes 15/10/5-step synthetic calls, carry incumbent,
packet accounting, rejection and v10 regression. It does not validate GPU
metering, physical timestamps, paired causality, search efficacy, scoring,
feasibility, source independence, or scientific gates.

Next: establish a new bounded feedback and audit protocol with an eligible root
and explicit stopping rule; connect an exclusive-output collector and validate
actual observations, dispatched actions, carry and packet ledger end to end.
Do not reuse the stopped JEPA oracle root for new optimization. PC0–PC5 remain
unproven by this work.
