# Affine-score spectral injection: real-RGB development ablation

The [development protocol](PC_POINTMAZE_AFFINE_SPECTRAL_CEM_PROTOCOL_20261103.md)
compares a spectral-injection arm with the saved random-CEM affine-score arm
on eight already exposed PointMaze roots. Both use the same goal, frozen
checkpoint, score, initial/tail innovations, 64 initial and 48 tail model
score samples, five-step packet execution and real cp5 RGB feedback. The
new arm substitutes four initial candidates with signed top-Jacobian
directions at fixed amplitudes; all other optimizer operations are shared.

| Root | Spectral injection | Random CEM |
| ---: | ---: | ---: |
| 1888 | 0.0345 | 0.0102 |
| 1889 | 0.8350 | 0.8205 |
| 1890 | 0.2314 | 0.2356 |
| 1891 | 0.2077 | 0.0347 |
| 1892 | 0.0973 | 0.0998 |
| 1893 | 0.4420 | 0.9040 |
| 1894 | 0.0369 | 0.0774 |
| 1895 | 0.1184 | 0.0936 |

These are offline physical endpoint position distances, lower is better.
Spectral injection is lower on four roots and higher on four. With the
strict offline threshold `<0.5`, it has 7/8 and random has 6/8; these are
descriptive on previously examined roots, not a statistically independent
success advantage. Root 1893 crosses that threshold; root 1889 remains
outside it for both.

Injected proposals entered the first-generation elite set on five roots.
None was itself the final selected full initial plan. Thus the change can
act through the CEM update; this result does not show direct execution of a
spectral proposal or a reconfiguration mechanism.

The spectral arm adds two JVPs and one SVD per root on top of the matched
model-score samples. Its eight finite-difference model calls per root are
validation-only, executed after the plan was selected. Structure time,
projection time, optimizer timing and individual calls are saved. The
random baseline was not given an equivalent structure allowance; total
compute is therefore **not matched**. The arms' actual proposals differ
after injection because CEM updates differ.

For the task-projected Jacobian at zero action, centered finite differences
at step 0.001 have relative Frobenius discrepancies between 0.00010 and
0.00047 across the eight roots; at 0.01 the range is 0.00245–0.00845. This
checks the local implementation, not its accuracy at 0.3 or 0.6 action
amplitudes. The frozen model hashes agree before and after the run.

An independent legacy-process replay of both saved arms verified **all
16 trajectories' frames 0–5 and frame 10 pixel-for-pixel**, including
the cp4/cp5 frames used for feedback. The spectral run also saved its
terminal legacy render and separate position-backend endpoints. This
checks these recorded frames, not every possible reset or model input.

The new run binds `seq_lengths.pth` and records action statistics; the
earlier saved random arm did not bind that file or persist its action
statistics. Therefore the exact historical normalizer identity across the
two runs cannot be established solely from their receipts, even though
they use the same loader and current archive. A subsequent
[saved-trace CUDA replay](../le-wm-main/results/pc_pointmaze_affine_score_replay_20261103.json)
reconstructed both arms' actual initial and cp5 contexts and rescored all
112 batches (1,792 candidates) under the current bound inputs. All scores
and orderings reproduced exactly, as did the saved cp5 input vectors.
This functionally validates these recorded comparisons but cannot establish
the historical contents of the missing source file or training provenance.
Checkpoint training-data
provenance and camera/scene calibration remain unresolved. The experiments
do not establish a causal capability change, equal-budget superiority,
independent generalization, or P1/PC admission.

- [Result and full candidate traces](../le-wm-main/results/pc_pointmaze_affine_spectral_cem_20261103_DEV01.json),
  SHA-256 `2606f1b1e6cbd70a263ed8593881df8355f2de20f15483ed0caa8db4f3e50694`.
- [Spectral actions and replay arrays](../le-wm-main/results/pc_pointmaze_affine_spectral_cem_20261103_DEV01.npz).
- [Independent history replay receipt](../le-wm-main/results/pc_pointmaze_affine_history_audit_20261103.json).
- Affine, feedback, spectral, saved-trace, mixed-basis and RGB checks:
  **48 passed**.
