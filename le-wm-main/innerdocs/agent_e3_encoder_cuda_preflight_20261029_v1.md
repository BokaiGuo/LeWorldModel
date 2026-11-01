# E3 two-frame HOST CUDA identity preflight candidate

This is an engineering preflight, not E3 production, a physical rollout,
task success, scientific admission or a substitute for full saved strict.
It requires the separate explicit authorization string
`MAIN_E3_ENCODER_IDENTITY_FIXTURE_CUDA_ONLY` and an absolute SHA-pinned
existing authority whose source bindings include this source and all
dependencies required by the identity HOST. Its clock starts before the
authority read. It fails closed if the actual checkpoint, model class/source,
device, freeze, precision, capacity or transport conditions cannot be met.
There is no CPU neural fallback.

The original loader's returned model is bound and capacity-checked before
two real encoder/projector calls. The fixed RGB224 fixture frames have every
pixel 0 and 127 respectively; boundary values 10/15 are ABI labels only, not
physical time or observed frames. The same held model/module/parameter
identities are checked by the new runtime identity contract. Original model
hash/class/precision/GPU peak snapshots are taken before and after; the after
snapshot is also attempted after an encoder failure. No prediction, AD,
training or physics is authorized. Saved consumption checks transport,
source pins, loader/identity events, actual binding line size, two fixed
frames, all seven arrays and their original IO links, and before/after guards.
It does not rerun neural computation or establish cryptographic process-memory
identity. A successful consumer result is only `two_encoder_io_verified` and
has `scientific_admission=false` and `full_saved_strict=false`.

For each frame the seven payloads total 1,554,432 bytes; two total 3,108,864
bytes, at most 3,166,208 including 14 conservative 4096-byte NPY headers,
under the unchanged 8 MiB encoding-array cap. Two PNGs individually retain
the 256 KiB cap. Identity events are bounded by the unchanged 147,465-byte
preflight check, subject to its no-reservation limitation. This does not prove
the complete model/class guard snapshot journal fits the physical-events cap:
the actual class inventory and binding line size remain unmeasured. Whole
512 MiB cap, 64 MiB reserve, 200 MiB floor, 1800-second wall and all category
limits remain unchanged. Any overrun is a failure, not a cap expansion.

Partial failure is not certified as success. The original encoder can write
array files before it emits its final descriptor; if it fails there, the
existing all-and-only publication consumer may reject the partial output due
to orphaned arrays. Such a failed run requires inspection of retained files
and is not a complete saved receipt. If failure occurs before the authority
can be read, this consumer cannot authenticate it. Production use requires
MAIN review and an actual encoder fixture/full transport audit first.

No run, model, GPU or physical call was made during implementation. The
following fixture tests exercise only fixed-frame construction, source-free
guard serialization and corruption, and the existing fake-loader capacity
and identity contracts:

```sh
.venv/bin/python -m pytest -q tests/test_agent_e3_encoder_cuda_preflight_20261029_v1.py tests/test_agent_e3_identity_capacity_20261029_v1.py tests/test_agent_e3_encoder_identity_20261029_v1.py
```

After MAIN audit, the explicit API is `run(authority_path, authority_sha,
output, AUTH)` and `consume(output, report_sha, completion_sha,
authority_path, authority_sha)`; CLI has `run` and `consume` subcommands with
the same explicit fields. Paths must be canonical absolute paths. Source and
test fixtures are not evidence that CUDA execution succeeded.
