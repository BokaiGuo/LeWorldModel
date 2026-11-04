# End-to-end causal RGB spectral/random development replay

The prewritten `PC_POINTMAZE_FULL_CAUSAL_SPECTRAL_DEVELOPMENT_PROTOCOL_20261103.md`
recompiled initial actions from the frozen model and ran eight ten-step RGB
trajectories on exposed roots 1884–1887. Initial RGB-position CEM actions
exactly reproduce the prior sealed initial planner. The cp5 observations,
causal-tail actions, histories and terminal RGB exactly reproduce the
previous fixed-prefix causal-tail study. Independent action replay also
matches all terminal RGBs and terminal positions.

| Root | Random CEM endpoint distance | Spectral CEM endpoint distance |
| --- | ---: | ---: |
| 1884 | 0.1573 | 0.1197 |
| 1885 | 0.0773 | 0.0431 |
| 1886 | 0.0601 | 0.1432 |
| 1887 | 0.1979 | 0.0729 |

All eight satisfy the strict offline endpoint `<0.5` check. Spectral
initialization has lower endpoint distance on three roots and higher on one.
These are the **same action chains and outcomes already reported**, so this
is engineering reproducibility of an end-to-end planning chain, not eight
new independent scientific outcomes or evidence of statistical superiority.
It is also not a native online early-termination success measurement.

Per arm, initial CEM used 48 two-packet model samples and tail CEM 48
one-packet samples. Each spectral arm additionally used two JVPs and one
SVD; the random arm did not receive compensating search budget. Structure
time and each CEM/encoding interval are recorded separately, but the
receipt does not establish equal total resources or all low-level forward
and backward operations. No spectral advantage, reconfiguration mechanism,
independent generalization, or P1 admission follows. Root source and
checkpoint training membership remain uncertified.

Artifacts:

- JSON `le-wm-main/results/a1_pointmaze_full_causal_spectral_dev_20261103_DEV01.json`,
  SHA-256 `76839b2e0ec7a5132946b144c578d9b2a1f882e6d31e7ae9d036730dc1b46379`.
- NPZ `le-wm-main/results/a1_pointmaze_full_causal_spectral_dev_20261103_DEV01.npz`,
  SHA-256 `ffae4ecd96e53528f2ee51d1e4b12806e7fa45dc783db18e33169bbb392ce886`.
- Saved initial/tail CEM trace, prior action/RGB binding, endpoint and
  structure-counter tests: **1 passed**; no model or simulator rerun.
- Eight execution trajectories and eight additional verification replays;
  total across all arms: 384 initial and 384 tail samples, 1,152 predicted
  packet transitions, eight JVP calls and four SVD calls.
