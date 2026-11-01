# E3 continuous launcher and saved inspector: work in progress

These files are not frozen and MUST NOT be used for production. They do not
provide the requested full saved strict, even when transport says complete.
The producer starts its monotonic clock before any authority read, creates one
unified Store, loads the original checkpoint through the journal bridge, calls
the authorized same-environment episode, records its return only after the
episode's post-cleanup model/source guard, and publishes failures as partial.
An over-deadline or failed publication is not a completed receipt.

The prospective run input is an externally SHA-pinned JSON object with keys
`schema`, `cell`, `control5`, `resolver`, `additional_sources`. The schema is
`agent_e3_continuous_host_launcher_20261029_v1.authority`. Each `control5`
entry `prepare`, `report`, `completion`, `receipt`, and the `resolver` entry
must have exactly absolute canonical `path` and `sha256`. `additional_sources`
is a canonical absolute path to SHA mapping. The fixed original preparation
and resolver SHA values are enforced in source. This is an interface description,
not a verified source-closure manifest or authority to run.

The inspector verifies the publication inventory, existing saved loader and
model guards, native-meter ledger, available observations and same-action
joins, and reconstructed later nodes. A 12-control-step fixture replays only
the reached node at 10; no padding or fabricated node 15. Partial failures are
reported as partial, without asserting that unverified authorities were valid.
No neural parity or physical replay is claimed.

The original metered reset lacks input recording. The new independent
`agent_e3_reset_identity_20261029_v1` adapter captures both actual reset calls,
actual raw vector dtypes and bytes, the sampling seed/recipe, selected polygon
vertices when the original sampler reads them, and the returned placement. It
uses explicit per-call wrappers, without patching the old meter or globals.
Its saved calculation rebuilds the sample in f64 from recorded inputs and
compares it byte-for-byte to both the sampled output and the second reset's
actual input. This evidence applies only to new executions; no historical
reset input is inferred. The tests include the original setup/sampler ABI with
fake environment and early stop, and reset/sample exceptions with cleanup.

The sampler wrapper additionally performs two separately journaled vertex
reads before sampling (attempt/return/failure, no physics step or render),
then requires the polygon actually read by the sampler to have the same
ordered f64 bytes. The saved inspector parses the hash-pinned official
`PushT.add_tee` source and verifies the complete two-polygon vertex sets;
it preserves the actual polygon and vertex order for sample reconstruction.
The two added vertex reads are new work and are not included in the original
native render/physics counts.

`agent_e3_loaded_module_census_20261029_v1` snapshots file-backed modules under
the project tree, including the local venv, before environment creation and
after cleanup. Each loaded module file must be presealed in the authority
bindings, rehashed, and recorded; existing model guard independently checks
every named model class and its source. An incomplete pin set fails closed.
The census is a pair of snapshots, not a complete import trace: a module loaded
and removed between snapshots and native dependencies outside the project tree
are not certified. It does not establish a full source/class execution closure.

Full native snapshot semantics, complete failure-path semantics, and an
end-to-end actual-ABI fixture for the new launcher remain unverified.
Class/module execution closure has not been demonstrated complete. No true
HOST/native end-to-end execution has been performed or implied. All scientific
gates remain false.

The temporary-disk combined fixture joins the real setup and sampler ABI,
reset/polygon evidence, fake-environment execution, fake encoder and node
callbacks, and saved feedback replay for both control20 and a control12 stop.
It does not exercise the actual native environment and is not a complete
production launcher fixture.

A separate temporary-disk fixture reads and hashes the actual saved MAIN
control5 and resolver authorities through the launcher, then exercises the
loader journal ABI using a synthetic loader receipt, deliberately fails before
environment creation, and checks partial saved consumption and authority
tampering. It does not load or deserialize the checkpoint. The physical saved
inspector independently checks the seven-vector against both recorded body
snapshots and their f64 bit seals, the original angle normalization, blue agent
pixels, canvas bounds, native success and reward, and cross-frame ordering.
The authority and synthetic loader now also share one launcher publication with
the fake-environment episode, in both full-20/two-node and control-12/one-node
early-stop fixtures. Saved loader, reset geometry and feedback are reconstructed
from that one journal. No real loader is invoked; the fake episode is not a
substitute for a complete native cleanup or model/source guard. These additions
do not establish complete native or failure-path semantics or full saved strict.
The same launcher fixture also exercises a fake-environment nested public-step
and physics-step failure, with independently reconstructed failed attempts and
a successful metered close. A forged cleanup-return without a metered close is
rejected. This does not prove every failure or cleanup path.

Fixture-only regression from the repository root:

```
.venv/bin/python -m pytest -q tests/test_agent_e3_launcher_authority_fixture_20261029_v1.py tests/test_agent_e3_loaded_module_census_20261029_v1.py tests/test_agent_e3_reset_identity_20261029_v1.py tests/test_agent_e3_continuous_host_launcher_20261029_v1.py tests/test_agent_e3_continuous_saved_consumer_20261029_v1.py tests/test_agent_e3_launcher_loader_bridge_20261029_v1.py tests/test_agent_e3_native_meter_20261029_v1.py
```

The fixture uses fake models and a fake environment. It makes no production
CUDA or physics call and does not certify their behavior.
