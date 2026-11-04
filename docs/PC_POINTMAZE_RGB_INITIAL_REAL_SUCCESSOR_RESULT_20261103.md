# RGB initial candidates: prediction, real successor and task decomposition

The prewritten `PC_POINTMAZE_RGB_INITIAL_REAL_SUCCESSOR_PROTOCOL_20261103.md`
replayed all 384 sealed two-packet plans using the legacy RGB renderer and
encoded all terminal RGBs using the same frozen model on host CUDA.
All 384 replayed terminal positions exactly matched the earlier position
backend; images are preserved in the result NPZ. This is a deterministic
post-outcome replay of complete open-loop plans, not a cp5-replanned
trajectory or a new independently selected policy.

| Root | Bank | Model vs real RGB reversed / comparable | Real RGB vs position reversed / comparable | Model-choice position distance | Real-RGB-choice distance | Physical oracle distance |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1884 | random | 270/1128 | 121/1128 | 0.3215 | 0.0241 | 0.0241 |
| 1884 | spectral | 238/1128 | 192/1128 | 0.3029 | 0.0419 | 0.0419 |
| 1885 | random | 528/1128 | 434/1128 | 0.2819 | 0.1991 | 0.1991 |
| 1885 | spectral | 551/1128 | 524/1128 | 0.2042 | 0.1127 | 0.1127 |
| 1886 | random | 169/1128 | 150/1128 | 0.2522 | 0.1533 | 0.1533 |
| 1886 | spectral | 173/1128 | 127/1128 | 0.0463 | 0.0463 | 0.0463 |
| 1887 | random | 723/1128 | 308/1128 | 0.7038 | 0.3111 | 0.2124 |
| 1887 | spectral | 692/1128 | 311/1128 | 0.6755 | 0.1717 | 0.1717 |

For root 1887, actual-RGB goal scoring chooses a candidate below the strict
0.5 position threshold in each bank, while the model-selected candidate in
each bank does not. The model-vs-real-RGB reversals on that root exceed
half the comparable candidate pairs. The actual-RGB selector also produces
the physical-oracle distance on seven of the eight development banks, and
improves on the model choice in seven and ties in one. **It sees the real
future and is an oracle, not a deployable controller**; this apparent
advantage does not establish an independent planning result.

Real RGB cost vs physical position also has nonzero reversals in every
bank: replacing the predicted future with the actual future does not make
the latent metric globally order-equivalent to physical distance. The
three score types use the same candidate plans, but candidate pairs overlap,
banks are adaptive, and four exposed roots are the independent units at
most. These observations support investigating both model prediction and
metric alignment; they do not prove a unique cause or justify post-hoc
selection using simulator outcomes.

Artifacts:

- JSON `le-wm-main/results/a0_pointmaze_rgb_initial_real_successor_20261103_DEV01.json`:
  SHA-256 `3331a9c74efb93f163df4799828066f0237f3a767222f3aa454b80c68e63e2fa`.
- NPZ `le-wm-main/results/a0_pointmaze_rgb_initial_real_successor_20261103_DEV01.npz`:
  SHA-256 `d657fdc56260e5a23c39bd2ed7bfdd75b736854c7dd638b11f6b385b84390ea7`.
- `.venv/bin/pytest -q tests/test_a0_pointmaze_rgb_initial_real_successor_20261103.py`:
  **1 passed**, checking saved score/bank bindings and physical position
  agreement; it does not rerun the frozen encoder or simulator.
