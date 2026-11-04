# AP author-reported Table 1: recorded-outcome reconstruction

The pinned author's source in `external/anchored-planning-984327424ca7` was
used without modification. Its original `tools/reproduce_table1.py` was run
with a new output directory:

`le-wm-main/results/pc_ap_author_table1_recomputed_20261102_v1/`

The script read 12,288 author-released outcomes and verified all 128
task/controller/start cells against the author's `data/main_results.csv`.
The printed standard and perturbed Table 1 values were also checked against
the Table 1 extracted from the repository's `paper.pdf`. The table uses 128
paired queries per task; perturbed rates average two perturbations within
each query, and the mean weights the four tasks equally.

| Task, standard | LeWM | AP-CEM | AP-rank |
| --- | ---: | ---: | ---: |
| Cube | 15.6 | 46.9 | 73.4 |
| PushT | 7.0 | 69.5 | 66.4 |
| Reacher | 66.4 | 75.8 | 96.9 |
| TwoRoom | 17.2 | 46.9 | 98.4 |
| Mean | 26.6 | 59.8 | 83.8 |

These are author-reported success percentages, reconstructed from released
records, not new local simulator trials. AP-rank has access to recorded
actions; AP-CEM uses observation memory. The LeWM baseline has different
information access, and neither author-reported column is a paired,
equal-resource comparison to the local D2P development runs. The full
reconstructed table also includes the final-goal and learned-target controls;
the latter are trained and are not training-free D2P.

SHA-256:

- `tools/reproduce_table1.py`:
  `2098e973bbd112a81fb0671e83288064894d7125202804315a09542273362add`
- `data/main_episodes.csv`:
  `d2276c2055312a7721690e8df33f6d9f1cb75194ccd4b92addd93238dcf8888d`
- `data/main_results.csv`:
  `3d16d83f527526d57b15fb8e6eb44275fd0c4ababb54a9e2c639a023e1f0c1ad`
- Reconstructed `table1.md`:
  `ff5cae72aa0cb9d496cbc2e79b9f968383523dad9c8a68d5d1da49611e8d3d04`
- Reconstructed `table1_unrounded.csv`:
  `2dbda6a04db8b71e465c29fa9e2b12fa00058d0e08e712183ff5ca11cdd938bd`

This does not establish local official AP controller fidelity, checkpoint
identity, training-data independence, or planning-compiler P1 admission.
