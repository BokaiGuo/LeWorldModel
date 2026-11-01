# P2 replay: source preflight and host handoff

This is a CPU-only source-snapshot prerequisite for the existing exploratory
launcher. It binds the original verified inventory and every regular `.py`
file found under `le-wm-main` except hidden directories, `.venv`, `results`,
`data`, `wandb`, `outputs`, and `__pycache__`. Additions, removals and byte
changes fail verification. Symlinked Python files fail; symlinked directories
are not followed. The snapshot includes both launchers and this preflight's
tests. The inherited inventory separately covers its pinned protocol,
compilation and checkpoint. Project non-Python files beyond that inventory,
excluded directories, external libraries, dynamic imports, interpreter,
drivers and environment are not comprehensively bound. This is not proof of
transitive source closure or independent preregistration. Review those gaps and
the actual runtime environment before any launch.

The wrapper rechecks the snapshot at each original launcher's inventory check,
including under its advisory lock and after work. Hash checks cannot eliminate
concurrent modification between checks, or attest code already imported before
the first check. Operator coordination and independently pinned, reviewed bytes
remain necessary; preserve the printed SHA externally. Do not alter any source
after snapshotting. Drift requires a new independently reviewed snapshot, not
an automatic reseal. Advisory locking and one-time process census are not
hardware exclusivity. No physical execution or P1/P2 admission is granted.

## Host commands

Run from the indicated checkout on the actual host. `P2_DIR` must be a new,
canonical directory with a canonical existing parent; replace the example path
and GPU UUID. Retain both printed digests and arrange independent review before
the final command. These commands do not train a model.

```bash
set -euo pipefail
cd /home/gbk/code/LeWorldModel/le-wm-main
P2_DIR=/home/gbk/code/LeWorldModel/le-wm-main/results/p2_replay_host_attempt_NEW
mkdir "$P2_DIR"
PREPARED=/home/gbk/code/LeWorldModel/le-wm-main/results/p2_composition_predict_frontier_cuda_20261027_v2_prepared01
PREPARED_SHA=1d79f5b17002334434454dfb5a94e6d918de1d603afdf84918e5ded6eb980d1b
.venv/bin/python -B -m experiments.spectral_reachability.p2_replay_source_seal_20261101 create --prepared "$PREPARED" --prepared-sha256 "$PREPARED_SHA" --output "$P2_DIR/inventory.json"
INV_SHA=$(sha256sum "$P2_DIR/inventory.json" | cut -d' ' -f1)
.venv/bin/python -B -m experiments.spectral_reachability.p2_replay_full_source_preflight_20261101_codex create --inventory "$P2_DIR/inventory.json" --inventory-sha256 "$INV_SHA" --output "$P2_DIR/snapshot.json"
SNAP_SHA=$(sha256sum "$P2_DIR/snapshot.json" | cut -d' ' -f1)
printf 'inventory_sha256=%s\nsnapshot_sha256=%s\n' "$INV_SHA" "$SNAP_SHA"
.venv/bin/python -B -m experiments.spectral_reachability.p2_replay_full_source_preflight_20261101_codex verify --snapshot "$P2_DIR/snapshot.json" --snapshot-sha256 "$SNAP_SHA"
```

After independent review and device coordination, use the externally retained
snapshot SHA (do not silently recompute it), with a physical `GPU-...` UUID:

```bash
GPU_UUID=GPU-REPLACE-WITH-PHYSICAL-UUID
.venv/bin/python -B -m experiments.spectral_reachability.p2_replay_full_source_preflight_20261101_codex execute --snapshot "$P2_DIR/snapshot.json" --snapshot-sha256 "$SNAP_SHA" --output "$P2_DIR/run" --device-uuid "$GPU_UUID" --operator-coordinated
```

The `run` directory must not exist. A failed or interrupted run is never
resumed or reused; preserve its records. The commands are a handoff, not an
authorization or evidence that CUDA or physical execution occurred.
