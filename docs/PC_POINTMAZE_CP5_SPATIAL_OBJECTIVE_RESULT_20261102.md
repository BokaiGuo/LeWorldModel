# CP5 full-spatial versus pooled terminal objective

The fixed candidate comparison found a concrete objective mismatch with the
released JEPA-WM PointMaze L2 code. Earlier development runs averaged visual
features across spatial positions before calculating MSE. The released L2
objective instead averages per-token squared differences, optionally adding
an `alpha`-weighted proprioceptive term. In general these are different
functions; this was a deliberate local implementation choice, not evidence
that the released model itself is defective.

On all 16 exposed-root prefix states and the same eight previously executed
tails, frozen host-CUDA predictions and actual successor encodings were
recomputed. The old pooled scores matched the sealed source to
`rtol=1e-4, atol=1e-6`. The alternative scores were full-spatial visual MSE
and full-spatial visual MSE plus 0.1 times predicted proprio-token MSE.
The latter arithmetic matches the released objective and the inspected
PointMaze evaluation configuration, verified by a targeted synthetic test
against the actual objective class. It does **not** reproduce the official
planning pipeline: the model still receives a single frame and fixed raw-zero
proprio, and the candidate bank, horizon and goals differ.

| Selector | Mean regret to same-bank physical oracle | Same index as oracle |
| --- | ---: | ---: |
| Pooled predicted visual MSE | 0.17644 | 1/16 |
| Full-spatial predicted visual MSE | 0.16130 | 2/16 |
| Full-spatial + 0.1 predicted proprio MSE | 0.16130 | 2/16 |
| Actual-successor full-spatial visual MSE oracle | 0.03510 | see JSON |

Full-spatial scoring changed two selections. Both had smaller position
distance in the previously sealed physical panel: root 1882 random-prefix
regret fell from 0.31203 to 0.17904, and root 1883 zero-prefix regret fell
from 0.10916 to 0. The proprio term did not change any of these 16 selected
indices relative to full-spatial visual scoring. The physical outcomes were
already executed before this selector was computed; their lookup is a valid
same-candidate development comparison, not a prospective physical test.

The real-successor score and physical oracle use privileged future information
and cannot be deployed. All 16 branches share only four exposed roots;
means are descriptive, not independent confidence estimates. Position
distance comes from the position-only offline replay and is not native
success. Training membership and normalizer provenance remain unknown, the
normalizer includes the full archive, and the camera was calibrated on a
development root. These results do not prove an equal-budget advantage,
reconfiguration mechanism, generalization, or P1 passage.

Full predicted visual/proprio tensors and actual visual tensors were saved.
Two tests passed, checking objective arithmetic against the released class
and recomputing saved scores from tensors; they do not independently rerun
the model or authenticate unselected physical endpoint replays.

- Protocol: `docs/PC_POINTMAZE_CP5_SPATIAL_OBJECTIVE_PROTOCOL_20261102.md`
- Runner: `le-wm-main/experiments/a1_pointmaze_cp5_spatial_objective_20261102.py`
- Result: `le-wm-main/results/a1_pointmaze_cp5_spatial_objective_20261102_DEV01.json`
- Result SHA-256:
  `17ec94dca406a1107a0af265e92af51e6bd3a36f3266422dca34c30e00cf5dc0`
- Test: `le-wm-main/tests/test_a1_pointmaze_cp5_spatial_objective_20261102.py`
