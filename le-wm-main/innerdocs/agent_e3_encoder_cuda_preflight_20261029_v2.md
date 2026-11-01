# E3 encoder preflight v2: array failure transport

The preflight remains two fixed synthetic frames only; no physical execution or scientific admission. The v1 encoder and unified transport remain unchanged. The new store writes an array intent before each encoder NPY and immediately journals its returned descriptor, or a failure with the complete descriptor when the bytes actually reached disk. A partial consumer first reconstructs the publication inventory and hash chain, then checks this intent/terminal ledger and returns the saved array inventory. It never calls a partial a successful encoder.

If a write is incomplete, or the journal cannot itself be durably written, the unreferenced file, pending operation, or category accounting mismatch makes publication consumption fail closed. No historical logs are repaired. Failures in publication itself may prevent any receipt.

Two calls, seven arrays each, imply at most 28 additional events, conservatively 458780 bytes at the existing 16385-byte event bound. The preflight checks that amount against existing physical-event and whole transport caps before loading; this check does not reserve space against subsequent writes, which remain individually guarded. Existing 512 MiB total, 64 MiB final reserve, 200 MiB floor and 1800 s wall are unchanged.

The v2 authority must pin both the v2 preflight source and the new array transport source in addition to the original closure. The saved consumer is transport and preflight evidence only: it does not rebuild full upstream strict, certify scientific success, or recompute neural parity. MAIN must review source/authority and explicitly authorize any future HOST CUDA execution; no preflight has been executed here.

Fixture command from the project root:

```
.venv/bin/python -m pytest -q tests/test_agent_e3_preflight_array_transport_20261029_v1.py tests/test_agent_e3_encoder_cuda_preflight_20261029_v1.py
```
