# CP5 action-history diagnostic and physical follow-up

The 16 previously sealed action chains were replayed with both the native
legacy visual backend and the separate position backend. All 16 step-5 and
step-10 RGB hashes match the source run, and all 16 position-backend states
match the saved offline endpoints. The two replay backends' final position and
velocity components were bitwise equal for this panel. The original online
bridge did not save its hidden state, so this is deterministic **replay**
evidence, not a direct comparison against that online state. No tolerance was
predeclared and no general backend equivalence is implied.

The existing model wrapper's `unroll` pads missing historical action features
with zero when multiple visual context frames are passed. The preceding
feedback experiment used a single frame, not a two-frame actual-history
context. A new separately implemented, exploratory interface supplies the
real step-0 and step-5 RGBs with the normalized executed prefix packet and
candidate tail packet aligned to the two context frames. It does not alter
the frozen model or overwrite the original experiment. This alignment is a
testable interface hypothesis; checkpoint training-source provenance is
still unknown. The normalizer uses the full archive and proprio remains fixed
raw zero.

On the same 16 previously executed tails, actual-history conditioning gave
lower pooled latent prediction MSE in 8 cases and higher in 8; means were
0.033019 with history versus 0.032461 with single-frame context. These are
dependent branches on four exposed roots, not independent statistical
observations. The single-frame costs were independently recomputed within
`rtol=1e-4, atol=1e-6` against the saved selector costs. The history selector
changed four of 16 tail choices.

Under a separately fixed physical protocol, **all 16** history-selected
tails were executed, including the 12 unchanged choices. All step-5 RGB
hashes and all 12 unchanged step-10 RGB hashes reproduced. For the four
changed choices, offline position distance to archive frame 10 was:

| Root | Prefix | Single-frame tail | History tail | Distance change |
| --- | --- | ---: | ---: | ---: |
| 1880 | spectral | 1.70698 | 2.08500 | +0.37803 |
| 1880 | axis | 0.66785 | 0.79033 | +0.12248 |
| 1882 | spectral | 1.08321 | 1.14913 | +0.06592 |
| 1882 | random | 1.12152 | 1.15445 | +0.03292 |

Positive change means worse position distance. The four changed choices
share two roots and are selected by a development rule; this is not an
independent estimate of general degradation. The history physical run
produced native RGB but reports position from separate offline replay;
the new changed tails have not themselves undergone a separate cross-backend
state comparison. Position distance is not native success. There is no
evidence here that adding this history interface improves real control,
nor does this negative result prove that history is intrinsically useless or
that the Planning Compiler hypothesis fails.

Eight targeted tests passed across action binding, source and array hashes,
history token alignment, saved prediction arithmetic and physical distance
arithmetic. They do not independently rerun every neural prediction. The
diagnostic checked source-ancestry hashes but did not itself rehash the
original A1 JSON before reading its initial RGB hash; a subsequent check
confirmed that JSON's SHA-256 equals the recorded source SHA. The current
artifacts remain exploratory, with no equal-total-budget, independent-root,
reconfiguration mechanism, or P1 claim.

- [Replay audit JSON](../le-wm-main/results/a1_pointmaze_cp5_replan_endpoint_audit_20261102_DEV01.json):
  `e150107a30d6a9074c377105db81a0616b4c45680b250b077ab5fe083f47b90b`
- [History diagnostic JSON](../le-wm-main/results/a1_pointmaze_cp5_history_diagnostic_20261102_DEV01.json):
  `040d627cd21c6337c244713c464ce91d2afa89d9923ca13d7f601e3d0e2b2cac`
- [Physical follow-up JSON](../le-wm-main/results/a1_pointmaze_cp5_history_physical_20261102_DEV01.json):
  `c01d4ee29642a54f8666e9e2a55485d61970634bf307e5ac4da1d5e2d1befa89`
- Protocols: `PC_POINTMAZE_CP5_HISTORY_DIAGNOSTIC_PROTOCOL_20261102.md` and
  `PC_POINTMAZE_CP5_HISTORY_PHYSICAL_PROTOCOL_20261102.md`.
