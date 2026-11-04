# PointMaze RGB metric: fixed-bank retrospective result

The [development protocol](pc_pointmaze_rgb_metric_protocol_20261103.md)
was written before this scoring run, but these data and a closely related
RGB geometry analysis were already exposed. This is a reproduction and
additional predicted-latent comparison, not independent confirmation.

All 624 previously saved actual terminal RGB frames across 12 banks and
three outcome-selected roots were scored. All terminal and goal marker
detections passed. Scoring uses only image marker centroids and pixel
distance; physical positions and outcomes are loaded only in the subsequent
evaluation stage. Goal and terminal images use the same top-down camera.
The source render receipt reports 624 exact position matches to the
saved physical oracle; that prior check was not rerun here.

| Within-bank comparison | Reversed / comparable | Reversal rate |
| --- | ---: | ---: |
| RGB geometry vs physical distance | 482 / 16,190 | 2.98% |
| Predicted-plan latent cost vs physical distance | 8,713 / 16,200 | 53.78% |
| Actual-terminal-RGB latent cost vs physical distance | 7,218 / 16,199 | 44.56% |
| Predicted-plan latent cost vs RGB geometry | 8,796 / 16,190 | 54.33% |
| Actual-terminal-RGB latent cost vs RGB geometry | 7,237 / 16,189 | 44.70% |

There are 10 RGB distance ties and one actual-RGB-latent tie across
within-bank pairs. Ties are excluded from comparable pairs and reported
separately. Counts are pair-weighted descriptive summaries, not independent
observations or confidence intervals. On root 1893 random-extra tail,
RGB geometry reverses 2/1,128 physical pairs, compared with 807/1,128 for
predicted latent and 769/1,128 for actual-RGB latent. The RGB argmin's
physical distance is 0.46994, versus 1.32006 and 1.24890 for the predicted
and actual-RGB latent argmins respectively.

The finding is limited to these outcome-selected, already analyzed banks.
Real future RGB is unavailable to a planner before action, so its ranking
is an offline oracle, not closed-loop performance. Camera/scene assumptions,
marker visibility, quantization, and render/integration timing remain
limitations. No model or physical simulator was rerun.

Artifacts:

- Source: `le-wm-main/experiments/pc_pointmaze_rgb_metric_20261103.py`
- RGB-only scores and provenance: `le-wm-main/results/pc_pointmaze_rgb_metric_20261103_scores.npz`
  and `le-wm-main/results/pc_pointmaze_rgb_metric_20261103_scores.json`
- Full paired results: `le-wm-main/results/pc_pointmaze_rgb_metric_20261103_evaluation.json`
- Tests: `le-wm-main/tests/test_pc_pointmaze_rgb_metric_20261103.py`

The evaluation report SHA-256 is
`c5aba4f98a232de452ff2e625506856e69478a30d1a90e3d6406ed31040dff11`.
