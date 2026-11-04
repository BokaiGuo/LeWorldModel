# Affine-inverse score with real RGB feedback: development result

The [fixed development protocol](PC_POINTMAZE_AFFINE_FEEDBACK_CEM_PROTOCOL_20261103.md)
was executed on all eight exposed PointMaze roots with the frozen model on
host CUDA. The new arm plans its own first packet and, after five physical
steps, replans from its own real cp4/cp5 RGB observations and causal velocity
estimate. It uses the same CEM innovations, sample counts, feedback points,
action horizon and reset sources as the previously recorded visual and
proprio arms.

| Root | Affine inverse | Visual | Proprio |
| ---: | ---: | ---: | ---: |
| 1888 | 0.0102 | 0.0606 | 0.0646 |
| 1889 | 0.8205 | 1.1686 | 0.5039 |
| 1890 | 0.2356 | 0.2993 | 0.3002 |
| 1891 | 0.0347 | 0.0433 | 0.2370 |
| 1892 | 0.0998 | 0.0271 | 0.1259 |
| 1893 | 0.9040 | 1.3201 | 0.2897 |
| 1894 | 0.0774 | 0.1887 | 0.4628 |
| 1895 | 0.0936 | 0.2082 | 0.4181 |

Values are offline physical endpoint position distances; lower is better.
Affine is lower on 7/8 roots than visual and 6/8 than proprio. The strict
offline distance threshold `<0.5` is met on 6/8 roots by affine, 6/8 by
visual and 7/8 by proprio. The two affine failures are 1889 and 1893;
one should not infer a success-rate advantage from the lower distances.
The roots and prior arm results were already exposed.

Each affine run recorded 64 initial and 48 tail prediction samples, seven
prediction batches, two current-image encodings, and ten physical steps.
The target is separately encoded per root. Candidate/score traces, action
sequences and RGB histories are saved; independent legacy rendering and
position-backend replays agreed. Saved trace, action, endpoint and hash
checks pass. Goal encoding, rendering and all low-level operations are not
fully accounted for; common sample counts do not establish equal total
compute. Previous comparison outcomes were loaded with the comparison
artifact, but the planner's candidate scorer and CEM callbacks do not
consume them.

The first execution completed eight trajectories and the two replay checks,
but report writing failed because a string was supplied to a path-only hash
function. Its `DEV01.npz` remains preserved without a completed report.
After correcting the type, the `DEV02` run completed; its saved arrays are
exactly equal to `DEV01.npz`. This is an execution/reporting defect, not a
negative planning outcome.

The score uses an analytic projection of predicted tokens, not an
authenticated state decoder. Camera/scene calibration and normalizer
provenance remain constraints. This experiment does not test the incremental
value of spectral candidates or non-greedy program composition, and lacks
independent roots and the required strong-baseline/resource contract. It
does not establish P1 or PC-stage admission.

- [Full result](../le-wm-main/results/pc_pointmaze_affine_feedback_cem_20261103_DEV02.json),
  SHA-256 `d1472f08afcf995168be28d62abded6b3e1083a634991efd5db61f3a20b84a4d`.
- [Saved arrays](../le-wm-main/results/pc_pointmaze_affine_feedback_cem_20261103_DEV02.npz).
- Relevant affine, safe ranking, mixed-basis and RGB metric tests: 44 passed.
