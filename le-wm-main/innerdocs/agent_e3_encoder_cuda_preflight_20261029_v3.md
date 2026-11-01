# E3 CUDA encoder preflight v3: chunked identity

The v2 attempt hit its single-event identity binding cap before encoder invocation. V3 writes the same identity descriptor (actual model instance, module instances, complete frozen parameter names/instances/devices, model and prepare hashes, random tag) as canonical JSON chunks of 6,000 bytes, each base64-encoded in a normal bounded event, followed by a length/count/SHA seal. No old file or original event-size limit is changed. Runtime encoder callbacks and repeated object/device checks are inherited from the original identity Binding. The saved path first verifies the original durable hash chain, all-and-only array inventory and sources, then reconstructs the binding and runs the original identity semantics with a synthetic in-memory bind event; the synthetic event is not written and is not an old receipt.

The descriptor raw bound is 16 MiB; before the first chunk, `(number_of_chunks+1)*16385` is checked against existing physical-event/whole caps and disk floor. This deliberately may reject a descriptor below 16 MiB if the unchanged 32 MiB physical-event budget cannot fit its worst-case record bound. It is a pre-check, not a reservation for subsequent guard snapshots and events. Per-event cap remains 16,385 bytes including LF, whole cap 512 MiB, final reserve 64 MiB, disk floor 200 MiB, wall 1800s. There is no automatic size expansion.

Incomplete chunk streams or errors are retained as partial when publication remains possible; torn/unowned files or failed journal publication remain fail-closed. Partial `binding_chunks_complete` means bytes and seal reconstructed only, not a verified loader/identity contract or encoder result. No partial is scientific or full strict. No past receipt is reinterpreted as v3 and no neural parity is recomputed.

MAIN first creates and verifies the identity authority using `agent_e3_identity_authority_prepare_20261029_v1` (its documentation supplies actual saved inputs). Then use a fresh absolute output path:

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m experiments.spectral_reachability.agent_e3_preflight_v3_authority_prepare_20261029_v1 prepare --source-authority /ABS/IDENTITY_AUTHORITY.json --source-sha256 IDENTITY_SHA256 --output /ABS/NEW_V3_AUTHORITY.json
.venv/bin/python -m experiments.spectral_reachability.agent_e3_preflight_v3_authority_prepare_20261029_v1 verify --source-authority /ABS/IDENTITY_AUTHORITY.json --source-sha256 IDENTITY_SHA256 --output /ABS/NEW_V3_AUTHORITY.json --output-sha256 V3_AUTHORITY_SHA256
```

Only after MAIN source review and explicit authorization, the separately gated preflight CLI is `python -m experiments.spectral_reachability.agent_e3_encoder_cuda_preflight_20261029_v3 run --authority /ABS/NEW_V3_AUTHORITY.json --authority-sha256 V3_AUTHORITY_SHA256 --output /ABS/NEW_OUTPUT_DIR --authorization MAIN_E3_ENCODER_IDENTITY_FIXTURE_CUDA_ONLY`. Its `consume` requires the same authority and explicit report/completion SHAs. This document is not an authorization and no CUDA execution occurred in this implementation.

Tests: `.venv/bin/python -m pytest -q tests/test_agent_e3_identity_chunk_transport_20261029_v1.py tests/test_agent_e3_preflight_array_transport_20261029_v1.py tests/test_agent_e3_encoder_identity_20261029_v1.py`.
