# Model-greedy prefix with real RGB tail replanning

The [fixed protocol](PC_POINTMAZE_GREEDY_PROPRIO_REPLAN_PROTOCOL_20261103.md)
executed eight new ten-step physical trajectories on exposed development
roots. The model-greedy prefix minimizes predicted goal-proprio cost at
five steps using 48 initial CEM candidates; each arm then replans with 48
tail samples from its own real cp4/cp5 RGB. No true future or simulator
state enters the selection. The checkpoint hash stayed unchanged; all
terminal RGB and positions matched separate replays.

| Root | Greedy | Zero | Random | Spectral |
| ---: | ---: | ---: | ---: | ---: |
| 1888 | 0.0425 | 0.0346 | 0.0646 | 0.0646 |
| 1889 | 1.2191 | 0.6325 | 0.5039 | 0.5039 |
| 1890 | 0.5207 | 0.2246 | 0.5669 | 0.5426 |
| 1891 | 0.2512 | 0.1775 | 0.2466 | 0.2381 |
| 1892 | 0.1312 | 0.1539 | 0.1259 | 0.1257 |
| 1893 | 0.6263 | 0.4544 | 0.2897 | 0.3196 |
| 1894 | 0.4590 | 0.3040 | 0.2739 | 0.2447 |
| 1895 | 0.3442 | 0.6544 | 0.4181 | 0.3705 |

Values are physical terminal position distances, lower is better. Spectral
is closer on five roots and farther on three; strict offline endpoint
`<0.5` holds on 6/8 spectral versus 5/8 greedy. On root 1894 the greedy
prefix also worsens position error from 0.2529 to 0.6855, while spectral
worsens it to 0.5819, so this comparison does not establish that spectral
forewent immediate progress relative to greedy.

Greedy is a model-based short-horizon baseline, not ground-truth greedy.
Initial sample counts match random/spectral, but greedy predicts one packet
per sample instead of two, and spectral additionally costs two JVPs and an
SVD. Goal/rendering and all low-level operations were not fully metered;
this is not certified equal-total-compute or a full-horizon strong baseline.
The zero/random/spectral outcomes are prior results, not new trajectories.
Exposed roots and unknown training provenance prohibit independence claims;
no causal non-greedy mechanism or P1 admission follows.

- Report:
  `le-wm-main/results/a1_pointmaze_greedy_proprio_replan_20261103_DEV01.json`,
  SHA-256 `627316fe3fcaa10e224f22346087a221e46d1f0ac2f1cb77738553551fd85e3c`.
- Arrays: same basename `.npz`, SHA-256
  `a00191a37b0616d36c842d0fe310a6e5d62055fed21a65db534f760d264e53c8`.
- Saved two-node CEM, action and outcome bindings together with zero-prefix
  tests: **2 passed**; tests do not rerun model inference or physics.
