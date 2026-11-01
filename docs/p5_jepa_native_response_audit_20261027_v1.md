# Independent saved-array audit of v8 native response

This new NumPy/stdlib-only audit never imports torch, v8, native predictor, encoder, dataset loaders, checkpoint loaders or model code. It reads only frozen JSON, source hashes and numeric .npy files, performs CPU NON-NEURAL array algebra, and writes exclusive new reports. No GPU jobs, training, neural fallback, actuator execution or sealed-file changes.

Preparation binds the v8 manifest byte SHA 0588f825c29b884350e97ed0dc47d8872bb539d0ca7a7faf045190eee5f3340f and every old seal plus this new source/test/doc. It does not require actual result availability or open a result before preparation. Actual audit requires the fixed actual episode1 result and four complete assets; missing or failed native result is a blocker, never substituted with fixtures. Files and source hashes are checked before and after read-only work.

## Independently recomputed

All four .npy records must have exact filename/directory, no symlink or path escape, file SHA and byte count, bounded header, exact shape/dtype, finite numeric values, expected payload length, no object dtype, no Fortran layout or trailing payload. Read header BEFORE mmap, max1024 header bytes; load allow_pickle=False/mmap_mode='r'. No arbitrary file or pickle decoding.

Using saved float32 response[98304,10], nominal and official target[1,256,384] and float64 residual, verify residual EXACTLY equals target.astype(float64)−nominal.astype(float64), actual target/nominal hashes, every ordered JVP column hash, step/x-y label, norm and max. Bind expected v7 feature receipts, full history/RGB hashes and actual normalized packet center, never insert live [-1,1]/tanh or raw rescaling.

Independently recompute G=JᵀJ/98304, rhs=Jᵀr/98304, Gram eigenvalues, direct tall-thin SVD of J, fixed .01 damped solution, linear residual MSE and equation residual. Save the complete small recomputed Gram/rhs/delta and both raw and MSE-normalized spectra. Compare CPU/GPU float64 arithmetic receipts with PREDECLARED rtol1e-8/atol1e-10, fixed in the source before reading the actual result. This is arithmetic consistency only, NOT a new FD scientific pass threshold. Reported singular values squared (raw divided by D) are checked against clipped Gram eigenvalues, avoiding unstable square-root tolerances at numerical zeros. Direct-SVD spectral values/differences are separately recorded, not used to select/rank/drop columns.

Verify native module weights match frozen v7 receipts, strict dynamics comparisons, imported-source seals, episode1/input bindings, no action execution, exact successful model-call counts (1 encoder,1 proprio,51 predictor), AD API count receipts (10 JVP,20 autograd.grad,0 direct backward API), frame totals (4/3/153), predictor input layouts and ordered nominal/JVP/FD phases. Counts are independently recounted from recorded model traces, not a new profiler/model run. Backward API zero does not mean no reverse-mode work: jvp uses double backward.

## FD scope and limitation

All20 saved summaries must remain: both fixed epsilons .01/.005 for all ten coordinates. Check finite/nonnegative norms, error RMSE/max bounds and cosine bounds, relative_l2 = sqrt(D)*RMSE/max(||J_i||,1e-12), and the norm/cosine error identity. These establish SUMMARY ALGEBRA consistency, NOT derivative correctness. Large self-consistent FD errors are explicitly accepted by tests as valid recorded diagnostics. No FD tolerance, epsilon selection, direction screening, invented pass verdict or retrospective efficacy claim is introduced.

The raw F(+epsilon e_i), F(−epsilon e_i) and FD vectors were NOT saved by v8. This audit therefore cannot independently reconstruct the finite differences or the neural JVPs, and cannot certify derivative agreement from those absent vectors. Output FD_accuracy_verdict is always not_assessed_no_threshold, with all actual diagnostics retained. Source/native invocation provenance, complete fixed traces and low scalar discrepancy are not substitutes for a saved-vector recomputation. No nonlinear solve candidate was evaluated; a lower LINEAR residual is not actual prediction or physical improvement. Single previously opened episode1 is not generalizable evidence or P5 admission.

## Commands

From /home/gbk/code/LeWorldModel/le-wm-main:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p5_jepa_native_response_audit_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p5_jepa_native_response_audit_20261027_v1 --output results/p5_jepa_native_response_audit_20261027_v1/manifest_frozen.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p5_jepa_native_response_audit_20261027_v1 --audit --manifest results/p5_jepa_native_response_audit_20261027_v1/manifest_frozen.json --output results/p5_jepa_native_response_audit_20261027_v1/episode1_saved_arrays_audit_v1.json
```

saved_array_arithmetic_consistent means ONLY the explicit source/receipt/hash/algebra checks passed. audit_inconsistent retains all numerical comparisons and exits1. Missing/corrupt/unsafe files or incomplete actual results fail before success reporting. No existing output is overwritten. Preparation deliberately binds protocol and parent inputs without reading actual results; the audit JSON separately binds the actual report and all asset SHA256. The actual result arrived during implementation and was available for the integration test and audit; this is independent arithmetic verification, not a prospectively blinded scientific acceptance test.
