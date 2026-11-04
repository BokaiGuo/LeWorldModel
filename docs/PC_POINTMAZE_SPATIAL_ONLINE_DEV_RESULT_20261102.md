# Full-spatial scoring in prospective real-feedback development runs

The predeclared protocol executed 24 native RGB trajectories: four fixed
archive roots (1884–1887), three candidate families (spectral, seeded
random-direction, axis), and two selectors (pooled and full-spatial visual
MSE). Each selector chose a five-step prefix, received real step-5 RGB,
planned within the same eight-tail bank and executed five further steps.
Position endpoints were evaluated afterward by the separate position-only
replay backend; they were not used in planning.

Across the 12 root-by-family pairs, both selectors chose the same complete
action chain in eight. In the four changed pairs, full-spatial scoring gave:

| Root | Candidate family | Pooled distance | Spatial distance |
| --- | --- | ---: | ---: |
| 1884 | axis | 1.155238 | 1.155224 |
| 1884 | random | 0.011725 | 1.155224 |
| 1884 | spectral | 1.096896 | 1.056668 |
| 1887 | axis | 1.669993 | 1.809479 |

Two differences favor each selector, but the axis improvement on root
1884 is only approximately `1.37e-5` distance units and no tolerance or
statistical test was prespecified. These are 12 dependent comparisons on
four roots. In particular, the random-prefix comparison on root 1884
shows that the prior post-hoc advantage for full-spatial scoring did not
reliably transfer to this development run.

All arms used five candidate prefix slots and eight candidate tail slots.
The two scoring metrics shared the initial model predictions per family;
each separate feedback branch performed its own RGB encoding and eight
tail predictions. Spectral generation also incurred two JVPs and an SVD.
Therefore equal candidate counts do not establish equal total compute.

The runs used real RGB feedback but a single-frame context and fixed
raw-zero proprio for a checkpoint that expects proprio. The action
normalizer includes the full archive; checkpoint training membership,
complete historic root use and source provenance are not certified. A
bounded search found no mentions of these roots in selected PointMaze
notes and scripts; it is not a full historical exclusion audit. Position
distance is not native success, and the physical position replay was not
directly compared to live hidden state or separately cross-backend audited
for these 24 newly selected chains.

Three targeted tests passed across source objective arithmetic, saved RGB
hashes, selection argmins and endpoint-distance calculations. The report
stores selected prefix actions and scores, but not the full generated
prefix candidate arrays, and records the runner hash rather than hashes of
every imported helper. Those are reproducibility limitations to close
before a confirmation experiment. This is not full task-weighted D2P,
an official planner reproduction, a non-greedy mechanism test, an
equal-budget strong-baseline result, independent generalization or P1.

- Protocol: `docs/PC_POINTMAZE_SPATIAL_ONLINE_DEV_PROTOCOL_20261102.md`
- Runner: `le-wm-main/experiments/a1_pointmaze_spatial_online_dev_20261102.py`
- Result: `le-wm-main/results/a1_pointmaze_spatial_online_dev_20261102_DEV01.json`
- Result SHA-256:
  `9788884799a786714a89f8f607f63f672b9b2b2ddfeb4934c91ae154051dc9a3`
- Test: `le-wm-main/tests/test_a1_pointmaze_spatial_online_dev_20261102.py`
