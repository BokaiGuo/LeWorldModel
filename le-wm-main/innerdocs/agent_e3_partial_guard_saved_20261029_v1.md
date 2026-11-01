# E3 partial failure guard evidence

`inspect(output, report_sha, completion_sha, authority_path, authority_sha)`
accepts only a partial publication and first invokes the existing bounded
saved inspector and authority checks. It reconstructs complete before/after
guard snapshot chunks and their hashes, validates the pinned sources, model
tensor hash, precision, loaded class identities and GPU-peak fields, and
requires any recorded cleanup outcome to lie between the guard snapshots.
An absent after snapshot remains explicitly unverified. Incomplete chunks,
bad seals, inconsistent pairs or invalid chronology are rejected.

This is saved-record verification, not evidence that fixture snapshots came
from real hardware. It does not repair an incomplete source/class census,
replay a physical episode, authenticate a failed loader as successful, or
establish full failure semantics. It always returns `full_saved_strict=False`
and `scientific_admission=False`; `guard_pair_verified` applies only to the
guard fields and source bindings checked here. Production remains subject to
MAIN review and the original gates.

Fixture-only regression from the repository root:

```sh
.venv/bin/python -m pytest -q tests/test_agent_e3_partial_guard_saved_20261029_v1.py tests/test_agent_e3_host_loader_episode_fixture_20261029_v1.py tests/test_agent_e3_native_meter_20261029_v1.py
```
