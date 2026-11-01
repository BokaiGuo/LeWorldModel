# E3 HOST loader and same-episode fixture boundary

This is a non-production fixture, not an authorization to launch E3. It uses
the original `loaded_model` control flow and real loader journal bridge with
temporary invented checkpoint files, a Hydra-created fake algebra model, and an
explicit fake deserializer. It does not load actual model weights, execute a
neural forward, allocate CUDA, or run physics. The string `cuda:0` is only a
fake model interface argument, not a CUDA allocation.

The launcher reads the real saved MAIN authority, while the fixture loader is
deliberately bound to a different synthetic preparation and tensor hash. The
same temporary journal then contains a fake-environment episode, including
both the 20-control/two-node and 12-control/early-stop branches. The fixture
independently reconstructs loader events and saved feedback, and verifies
that the real authority loader check and the whole saved consumer reject the
identity mismatch. Altering a fixture weights file before loading is rejected
before deserialization. This does not establish provenance or model equality
for the real checkpoint, nor does it validate a successful full consumer run.

The fixture also calls the original model/source guard snapshot function on
the loaded fake model before and after the fake episode, with an explicit fake
CUDA interface, and records both snapshots in the same journal. The same
loaded object is passed to the fixture numerical backend and its fixture-only
encoding marker. Tests check fake parameter/device identity and prediction
calls, and an injected encoding error checks metered fake cleanup and the
post-cleanup model hash. The marker is not the actual encoder; neither real
GPU allocation nor production model guard is asserted.

Existing guard tests separately cover saved source tampering and model guard
validation; they are not a same-episode real model guard. The fixture episode
does not prove full production cleanup semantics. Full native failure-path and
source/model lifecycle reconstruction remain incomplete, and the existing
consumer explicitly reports `full_saved_strict=False`. Census remains a pair
of snapshots, not a transient-import trace. No scientific gate is opened.

From the repository root, fixture-only regression:

```sh
.venv/bin/python -m pytest -q tests/test_agent_e3_host_loader_episode_fixture_20261029_v1.py tests/test_agent_e3_launcher_authority_fixture_20261029_v1.py tests/test_agent_e3_continuous_saved_consumer_20261029_v1.py tests/test_agent_e3_model_source_guard_20261029_v1.py tests/test_agent_e3_guard_saved_20261029_v1.py tests/test_agent_e3_native_meter_20261029_v1.py
```
