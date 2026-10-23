# TwoRoom support-gated repair V1 execution

This is the executable C2 test for the support-gated repair component.  It
uses three new 160-root panels with nonoverlapping seed ranges:

- `20264896..20265055`
- `20265056..20265215`
- `20265216..20265375`

Within each panel, only train roots fit the effect models. Calibration roots
freeze the support threshold from paired true-ranking regret of full adapter
versus action-only fallback. Outcome roots are materialized only after that
threshold exists and compare three deployment policies: always full adapter,
always action-only, and support-gated adapter.

The runner is host-CUDA only:

```bash
systemd-run --user --collect --working-directory=/home/gbk/code/LeWorldModel/le-wm-main \
  /bin/bash /home/gbk/code/LeWorldModel/le-wm-main/experiments/gdd_wm/run_tworoom_support_gated_repair_v1_cuda.sh
```

The primary evidence is fresh paired terminal-distance and candidate-regret
gain of gated over both components. A positive result is scoped evidence for a
specific support-gated repair; it is not a conditional coverage guarantee.
