# Frozen-model spectral actions on native PointMaze

The fixed development protocol
`PC_POINTMAZE_SPECTRAL_NATIVE_A1_PROTOCOL_20261102.md` was executed on host
CUDA. Each of four exposed roots supplied a native-rendered initial RGB and
the archive frame-10 goal image. The frozen model was differentiated with
respect to a physical action repeated for five archive steps; its
384-by-2 successor Jacobian was decomposed by SVD. Spectral, seeded
one-dimensional random, and axis-aligned candidate sets each contained five
actions, and each selected candidate was executed in the native environment.

The JVP Jacobian's relative finite-difference errors at epsilon `1e-3` were
0.00046–0.00299. The independently recomputed singular values agree with
the saved values. This validates the local derivative only to the extent
measured; it does not establish the Jacobian is a reliable approximation
at the 0.3 or 0.6 action amplitudes.

| Exposed root | Spectral distance | Random-direction distance | Axis distance |
| --- | ---: | ---: | ---: |
| 1880 | 1.021208 | 0.406713 | 0.623167 |
| 1881 | 0.621229 | 0.755837 | 0.575713 |
| 1882 | 1.275611 | 1.274671 | 1.486326 |
| 1883 | 0.317707 | 0.359987 | 0.318249 |

Distances are from a separate MuJoCo-2.2 position-only replay at step 5 to
the archive frame-10 position. The native visual environment executed each
action and saved the resulting RGB hash, but these new twelve action chains
have not separately undergone the cross-backend final-state replay audit.
No native success criterion or feedback replan is measured in this run.

Spectral is better than the seeded random direction on two roots and worse
on two; it is better than the axis set on two and worse on two. This
does **not** establish a spectral advantage. Model choice can be influenced
by the unweighted latent metric and the model's physical prediction errors;
the data do not identify a unique cause for the negative cases.

All methods score five candidate actions, but spectral also pays for two
JVP calls and an SVD. Derivative wall times were approximately 0.040–0.146 s
per root versus approximately 0.008–0.009 s for each candidate scoring call.
Finite differences added four diagnostic model evaluations per root and are
not a deployment component. JVP backward operations and the full shared
encode/environment costs are not fully converted into a common compute
budget; this is not an equal-resource or strong-baseline comparison.

The experiment is an unweighted single-packet SVD, not the full task-weighted
progress/reconfiguration/composition algorithm. It uses already exposed
roots, a development-selected camera, a constant raw zero proprio input and
full-archive normalization whose training provenance is unknown. It does not
establish an independent mechanism, reconfiguration, P1, or generalization.

- Runner: `le-wm-main/experiments/a1_pointmaze_native_spectral_20261102.py`
- Result: `le-wm-main/results/a1_pointmaze_native_spectral_20261102_DEV01.json`
- Result SHA-256:
  `2c61566337ae9915807e724913143e067a46ddfa519431de804ee21b452eed4a`
- Seven targeted tests passed, including saved Jacobian, SVD, candidate
  selection and native-bridge checks. They do not establish efficacy.
