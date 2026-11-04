# CP12 initial RGB mismatch: renderer-source audit

The same local bridge and saved reset/action inputs were independently
replayed in the restricted sandbox and on the approved host GPU. Each of
four exposed roots used its `raw`, candidate 2, sealed prefix. Both modes
repeated the initial reset once.

| Root | Sandbox exact frames 0–12 | Host exact frames 0–12 | Sandbox initial differing pixels | Sandbox initial MAE, 0–255 | Sandbox minus saved initial RGB-position (x,y) |
| ---: | ---: | ---: | ---: | ---: | --- |
| 8343 | 0/13 | 13/13 | 2867 | 0.02622 | (-0.006050, -0.001635) |
| 8345 | 0/13 | 13/13 | 2861 | 0.02596 | (+0.000556, +0.004143) |
| 8349 | 0/13 | 13/13 | 2867 | 0.02827 | (+0.004676, -0.004105) |
| 8351 | 0/13 | 13/13 | 2834 | 0.02509 | (+0.002535, +0.001014) |

For initial frames the differing pixels span bounding box x/y 29–194.
Current initial images repeated exactly within each environment. Saved-source
and bridge hashes, `mujoco_py` version 2.1.2.14, compiled extension hash
`daf43161ab3c8c281e431e8db1b412a126ca74c73498ccdf233b9b360eb3f3dd`,
and MuJoCo runtime library hash
`1ebd4a5f6126cf440dad2fb641bb54e905799eaf90b093eb04111f6ca32f0fcd`
matched across the two current executions. The OpenGL probe, run in a
separate subprocess with the bridge's runtime/library environment, reported:

- Sandbox: Mesa llvmpipe, LLVM 20.1.2, Mesa 25.2.8.
- Host: NVIDIA RTX 5060 Ti, OpenGL 4.6.0 NVIDIA 595.99.02.

This comparison identifies a reproducible environment-dependent rendering
difference, with host rendering matching all 52 sampled historical frames.
It does not independently prove which driver or renderer produced the old
frames: the historical cp12 receipt did not bind that configuration. Nor
does the sandbox pixel difference establish a physical dynamics failure. The
earlier failed exact-replay gate was conditional on sandbox software
rendering and should not be generalized to the host GPU path.

Both modes used 48 stored prefix actions, 4 extra reset/repeat checks, zero
model calls and no suffixes. Results bind the original receipt/NPZ and current
bridge; GL identity is measured in a separate process, not queried inside the
actual bridge process. The audit doesn't bind the full historical dependency
closure or establish cross-renderer latent/decision equivalence.

- Sandbox receipt SHA-256
  `5d92277ce56b23593e547964690cd1d8213bbe5aa97dfccdcc3ee773e9a0c8f3`.
- Host receipt SHA-256
  `408f0990bb957f585c1c873e0e8c272bf0e32082f1e0471b6270dc239a5f34ae`.
- Focused tests: **5 passed**, verifying comparison arithmetic and receipt
  invariants, not independently re-executing the renderer.
