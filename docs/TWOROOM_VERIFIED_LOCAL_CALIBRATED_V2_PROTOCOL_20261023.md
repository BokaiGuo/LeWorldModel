# TwoRoom verified-local calibrated V2

## Why V2 exists

V2 is a fresh execution of the frozen V1 scientific question after correcting
an implementation defect in the original runner: a shared `--episodes 64`
argument could truncate an authorized 128-root outcome panel.  No V1 outcome
is used for V2 route selection, threshold selection, or claims.

## Frozen protocol

- Calibration roots: `20264224..20264287` (64).
- Sealed outcome roots: `20264288..20264415` (128).
- Three routes, candidate bank, checkpoint, CEM budget, RGB effect map,
  fallback, teacher, and exact frozen-LeWM local-fallback comparator are
  unchanged from V1.
- The selector writes `protocol_id` and the exact calibration seed panel.  The
  auditor verifies its matching protocol ID and exact outcome seed panel.

The outcome starts only after calibration chooses one route by the original
positive paired-success, four-wins, one-sided sign-test rule.  Any calibration
rejection seals the outcome roots.  The only intended claim from a passing
result is a matched TwoRoom teacher contribution above the identical verified
RGB-local fallback; it is neither an RGB-reactive comparison nor a universal
world-model claim.

## Host-only command

Run this file through the host CUDA path, never through Codex sandbox:

```bash
systemd-run --user --collect --working-directory=/home/gbk/code/LeWorldModel/le-wm-main \
  /bin/bash /home/gbk/code/LeWorldModel/le-wm-main/experiments/gdd_wm/run_tworoom_verified_local_calibrated_v2_cuda.sh
```
