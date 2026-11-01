# P2 known-root shared-RGB pair: execution and saved audit

This is one development-only, known-root model-planning comparison. It is not
an independent confirmation, fresh-root result, physical execution, or P2
admission. The capture was reconstructed from a saved B0 native physical
receipt; its SHA establishes binding, not independent physical provenance.

The prepared contract is
`le-wm-main/results/p2_shared_capture_launch_20261102_exclusive_CONTRACT_20261102_p2continuation.json`,
SHA-256 `646ef8da17e9d4a7600e2fa86f402d2c04ef9a78aa9179adefcc3740eedb7e6f`.
It binds capture SHA-256
`0a8c4c8ff7798fa082a168f3b770e75238ce556627cf4ba56a058c305f331706`,
the frozen checkpoint, normalizer, and selected source files. It is not a
complete transitive software-environment lockfile.

The first, sandboxed attempt stopped before model load because NVIDIA device
nodes were unavailable. Its preserved report SHA-256 is
`21ffd8b5065002df8d4c40417ab9cfb357b8ecf73ccb49a56932fdce3248dc43`.
After host execution was authorized, a separate exclusive output was used:
`le-wm-main/results/p2_shared_rgb_pair_wiring_20261102_codexexclusive_RUN_20261102_p2continuation_host01`.
The completed report SHA-256 is
`d39f8e4f5f292034331b8a60b3fe0682b04d81206fd904200db2295a38e3ef2d`;
pair SHA-256 is
`3f3bdfc70bec1a32e62a7df9093d0fd90e5bd5d43e2960a23fd59e5d27244f88`.

Four shared RGB encodings returned. Each branch returned exactly the eight
contracted resource counts: packet/action encoder/predict/predictor/prediction
projection 44 each, Jacobian and SVD 5 each, and score 17. Terminal
model-predicted latent MSE was 0.8707822022 for joint_multistart and
0.7880259874 for prefix_chain; both selected candidate 12. These are model
objectives on one known source, not measured physical task outcomes or a
generalization claim. Measured resource counts do not establish equal total
FLOPs, energy, I/O, or wall time.

CPU tests: 10 passed across the new launcher, shared RGB wiring, and capture
tests. The new saved audit rechecks capture provenance to the limits of its
saved receipt, report and journal hashes, RGB and latent receipts, normalized
actions, both branch records and winners. It addresses an event-wrapper
mismatch in the earlier saved consumer without modifying that file. It returned
`verified_saved_pair_known_root_not_admitted`; P1 and P2 admission remain false.
No new physical actions were taken.

Recheck the completed output on CPU:

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p2_shared_capture_launch_20261102_exclusive audit \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/p2_shared_rgb_pair_wiring_20261102_codexexclusive_RUN_20261102_p2continuation_host01 \
  --report-sha256 d39f8e4f5f292034331b8a60b3fe0682b04d81206fd904200db2295a38e3ef2d
```
