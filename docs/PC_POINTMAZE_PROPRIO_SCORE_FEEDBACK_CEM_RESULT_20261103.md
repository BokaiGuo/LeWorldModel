# Predicted-proprio score in real RGB feedback CEM

The [fixed development protocol](PC_POINTMAZE_PROPRIO_SCORE_FEEDBACK_CEM_PROTOCOL_20261103.md)
executed 16 ten-step trajectories on host CUDA with frozen model parameters.
Each arm planned 64 initial and 48 feedback samples using the same noise
and the same CEM implementation. The proprio arm recomputed its own initial
plan and tail using its own real cp5 RGB and causal velocity; it did not
select among physically evaluated futures. The visual arm exactly reproduced
the prior saved plans, RGB histories, terminal images and endpoint states.
Both arms passed separate legacy-render RGB and position-backend replay.

| Root | Visual terminal distance | Proprio terminal distance |
| ---: | ---: | ---: |
| 1888 | 0.0606 | 0.0646 |
| 1889 | 1.1686 | 0.5039 |
| 1890 | 0.2993 | 0.3002 |
| 1891 | 0.0433 | 0.2370 |
| 1892 | 0.0271 | 0.1259 |
| 1893 | 1.3201 | 0.2897 |
| 1894 | 0.1887 | 0.4628 |
| 1895 | 0.2082 | 0.4181 |

The proprio score improves distance on two roots and worsens it on six.
Strict offline endpoint distance `<0.5` is 7/8 versus 6/8 for visual; root
1889's 0.5039 is **not** a success. These are exposed development roots, not
independent trials, and the 1893 improvement follows prior inspection of
that root. This is not native online success or proof of generalization.

Each arm recorded seven prediction batches, 112 samples, two current-image
encodings and ten physical steps; goal encoding was shared per root. Both
scores were computed for each batch for implementation symmetry. Combined
encoding/planning time was recorded for each node but goal encoding,
rendering and full low-level operations were not completely metered, so
equal sample counts are not a certified equal-total-cost comparison.
The experiment contains no spectral arm, common-suffix intervention, new
root certification or strong-baseline suite, and does not establish P1.
The goal velocity zero and RGB projection assumptions remain explicit.

- Report:
  `le-wm-main/results/a1_pointmaze_proprio_score_feedback_cem_20261103_DEV01.json`,
  SHA-256 `a07f3c73c6cbcc215899d7bbaa6ab1b30ecadf43188b7a882e926c66abdac3df`.
- Arrays: same basename `.npz`, SHA-256
  `df1e0eeddeac0e8cf79b71aae8507d5b9e8efb781e5392d707c342fe8e6a62fb`.
- Saved optimizer trace, action sequence, endpoint and source replay tests,
  plus score tests: **3 passed**. They do not rerun physics or CUDA.
