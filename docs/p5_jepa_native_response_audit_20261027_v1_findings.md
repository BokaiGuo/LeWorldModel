# Actual saved-array and v9 read-only findings

No neural computation or GPU jobs were run. All v8/v9 source/test/doc/manifests/reports/assets remain unchanged. Newly written reports live under le-wm-main/results/p5_jepa_native_response_audit_20261027_v1.

## v8 independently recomputed matrix audit

episode1_saved_arrays_audit_v1.json reports saved_array_arithmetic_consistent: all24 checks pass, all10 JVP columns and all20 FD summaries retained. Four authentic numeric files were read with bounded headers, allow_pickle=False and read-only mmap. Residual equals actual target minus nominal exactly; target and nominal feature hashes, all column hashes, v7 native feature/history bindings, complete source seals, frozen module hashes, original normalized action center and all recorded phase/frame/AD costs match. 16 new non-neural tests passed including the actual-array integration test.

Independent NumPy float64 recomputation differs from CUDA receipts by at most:

| Quantity | Maximum absolute discrepancy |
|---|---:|
| Gram | 7.28584e-17 |
| rhs | 3.46945e-18 |
| normalized ridge delta | 6.52256e-16 |
| Gram eigenvalues | 6.71528e-17 |
| direct-SVD raw-J singular values | 1.11244e-12 |
| linear residual MSE | 0 |

Both singular spectra and fixed .01 damped solution are consistent with all saved columns. The nominal MSE is 0.06643459207997592 from arrays; linearized solved MSE is 0.061086641639004124. This is a linear response prediction, NOT an evaluated nonlinear improvement. All columns are included; no rank/epsilon/episode screening.

Recorded FD maxima are:

| Fixed epsilon | RMSE maximum | Relative L2 maximum | Absolute error maximum |
|---|---:|---:|---:|
| .01 | 8.18575e-5 | 5.46698e-4 | 2.28834e-3 |
| .005 | 9.44719e-5 | 9.24374e-4 | 1.40929e-3 |

These summaries are algebraically consistent with saved JVP column norms and their reported cosine/FD norm identities. Smaller epsilon does NOT uniformly reduce diagnostic error; no monotonic-convergence claim is made. Raw finite-difference vectors and plus/minus successors were not saved, so the audit cannot independently recompute FD/JVP agreement. There is deliberately no FD accuracy threshold/pass verdict. Float64 CPU/GPU receipt consistency tolerances are NOT a derivative/scientific gate.

## v9 fixed-delta replay validity

episode1_v9_readonly_validity_v1.json records bounded_v9_receipts_consistent, all17 checks pass. All v9 manifest-bound files match current hashes. Its embedded v8 manifest/report binds exactly to the independently audited arrays. Delta matches the unchanged full ten-coordinate v8 solution and the independent solve; float32 model-boundary delta SHA is correct. Official encoder/native predictor/proprio state hashes match v8, strict loads and imported sources remain sealed, model before/after receipts match. Source preserves both historical packets and adds delta only to actual normalized new packet; no tanh/clamp, line search, changed damping or candidate selection. The source makes exactly three predictor calls (nominal, fixed delta, fixed delta replay), matching reported counts3 plus encoder1/proprio1. No per-call v9 trace exists, so frame/phase recount is not claimed.

| Fitting diagnostic | Target global patch MSE |
|---|---:|
| actual-packet nominal | 0.0664345920799759 |
| frozen linear prediction | 0.061086641639004124 |
| recorded fixed-delta nonlinear replay | 0.06593793513867648 |

The recorded nonlinear decrease is only 0.0004966569412994254; nonlinear minus linear is +0.004851293499672352. Thus the linear approximation substantially overstates observed-target nonlinear fitting improvement. Delta was cast from float64 solve to float32 at the declared native boundary; that small rounding is included in the comparison. No concrete source implementation bug was found in this bounded review.

Candidate/replay tensors were not saved by v9. Their SHA, MSE and exact torch.equal replay remain source-bound host-CUDA receipts, NOT a second independently reconstructed neural result. Source checks and scalar arithmetic agree; no rerun occurred. The already observed frame15 was used in ridge fitting, and episode1 was previously opened. This is not prospective planning, physical action execution/control benefit, fresh heldout science, completed D2P or P5 admission. Conditional cached weight provenance and the v6 failed official-vs-HF equivalence remain unchanged.
