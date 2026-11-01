# P2 shared-RGB matched-pair wiring

The existing `p2_matched_program_host_20261101_parallel.py` is the reusable
host for the fixed, previously sealed latent panel. The new
`p2_shared_rgb_pair_wiring_20261102_codexexclusive.py` provides a separate
host-CUDA entry point for a different input: one shared CP15 RGB capture
bundle and its 15 recorded action receipts. It does not execute physical
actions and does not modify the old host, panel, or gates.

The new entry point encodes RGB frames at controls 5, 10 and 15 plus the goal
once each; builds the two history packets from the last ten returned actions
using the pinned native normalizer; then executes the existing
`joint_multistart` and `prefix_chain` kernels on the identical latent context
and goal. Each branch is checked with the existing saved branch verifier before
the pair is marked complete. Both use the existing exact eight-axis caps; the
four shared encodings are recorded separately and are not free work. CUDA
branch clocks synchronize and enforce the existing 4 GiB allocated / 6 GiB
reserved limits. Failures terminate the pair without retries or replacement.

## Input and future run conditions

An external capture producer must provide an immutable JSON file and four
flat PNG files in its directory. The JSON has:

- `schema`: `p2_shared_rgb_pair_wiring_20261102_codexexclusive.capture.v1`;
  `base`: integer 0–3.
- `frames`: three objects in order with `step` 5, 10, 15 and `file`,
  `file_sha256`, `pixel_sha256`. Images must decode to uint8 RGB 224×224.
- `goal`: the same three file/hash fields, without `step`.
- `actions`: 15 ordered objects with `step` 1–15, `dispatch`, `returned`,
  `terminated`, `truncated`; action vectors must be exact finite float32
  values in [-1, 1], dispatch must equal returned, and stop flags must be
  false.

File hashes cover PNG bytes; pixel hashes cover contiguous RGB bytes. The JSON
is pinned by its external SHA-256. This is an integrity and shape contract:
the supplied receipts are **not independently authenticated as physical
observations** by this entry point. A trusted producer and a separate
source/feedback lifecycle verifier are still required before describing the
input as independently verified real feedback. A SHA is not such proof.

After the GPU is released, a separately authorized host operator can run:

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p2_shared_rgb_pair_wiring_20261102_codexexclusive \
  --capture /absolute/path/capture.json --capture-sha256 CAPTURE_SHA \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/p2_shared_rgb_pair_wiring_20261102_codexexclusive_RUN_UNIQUE \
  --authorize-host-cuda
```

Requires the existing advisory cooperative GPU lock, a working host CUDA
device, the exact pinned PushT LeWM checkpoint and normalizer source, writable
exclusive output and disk reserve. The model remains frozen; Jacobians are
computed for planning and no training occurs. The output contains a hash-chain
journal, pair and report on success, or a partial/failure report where
possible. The input PNGs and JSON remain external and must be retained.
There is not yet a separate saved-output consumer for this new entry point;
the existing branch validation runs in-process.

CPU fixture tests only have run. Actual CUDA encoding, capture provenance,
physical execution, independent-root confirmation, FLOPs, energy, I/O and
total-compute equality remain unverified. P1 and P2 admission remain false;
the comparator here is `joint_multistart`, not the separately named SRA arm.
