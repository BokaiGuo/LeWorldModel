# Native normalized-action response prerequisite (v8)

Fixed minimal episode1 from the already-opened v5/v7 panel 1–8. This version does not run the other seven episodes, replace a bad episode, establish fresh heldout evidence, train, execute actions or admit P5. Old source/docs/reports/manifests and official native source remain unchanged. Preparation/tests alone are run by the agent; only main runs CUDA after review.

## Native data and coordinates

Replay the sealed v7 complete inputs: official PushT validation RGB frames0/5/10/15, first three as context, actual frame15 official patch target, real normalized proprio histories, all actual 0:5/5:10/10:15 action packets. Official encoder preserves 37×37 positional grid, native offset .1 and antialias false; strict cached HF safetensor mapping is unchanged and does NOT authenticate independent official checkpoint provenance. No HF model or shim executes. Predictor/proprio state hashes must equal the actual v7 receipts. Encoded full/target feature hashes must exactly equal v7 episode1; nominal actual MSE must agree within predeclared absolute 1e-6. No change to v6 equivalence gates or negative result.

Let a0 be the actual last normalized 5×2 packet, in time-major x/y order. F(delta) calls the native predictor with unchanged real visual context, unchanged proprio, first two actual historical packets and last packet a0+delta, then returns its last visual successor flattened to 98,304 coordinates. delta=0 is the actual packet—not mean action. This derivative is in ten native NORMALIZED coordinates; no [-1,1] constraint, tanh, clipping, actuator command replacement, raw-coordinate rescaling or optimization-action execution. The data-raw relation is 100*(normalized*std+mean); it is not inserted into the derivative path. Damped delta remains in these same normalized coordinates.

Parameters stay frozen/eval/all-CUDA. Real features are encoded under no_grad, NOT inference_mode, to create ordinary context tensors valid for backward. Only the ten input coordinates carry derivatives. This is one successor, not an autoregressive prefix/suffix compiler: no future observed proprio is fed into a suffix.

## Derivative and algebra protocol

Use exactly ten sequential torch.autograd.functional.jvp calls, standard basis e0..e9, strict=True/create_graph=False. This API uses DOUBLE BACKWARD, not native forward-mode; support for native predictor operations must be demonstrated by main CUDA, not assumed from tests. Native SDPA math is selected, TF32 off, same official/predictor arithmetic policy as v7. No alternative AD method is silently substituted if unsupported. Native strict loads and imported-source boundaries precede any model forward.

Save the actual patch target-minus-successor residual values, shape (1,256,384), MSE, nominal successor SHA and actual real feature/target/input/history hashes. Each JVP saves coordinate/step/axis, norm, maximum, full column SHA and primal-vs-nominal error (fixed max absolute 1e-6). Retain all ten columns on CUDA, float32 (98,304×10 = 3.75 MiB). Derive G=JᵀJ/D and b=Jᵀr/D with D=98,304 and float64 CUDA reductions. Save all 10×10 Gram entries, rhs, eigenspectrum, raw-J and MSE-normalized singular values. Fixed damping .01, solve (G+.01 I10)delta=b. Save delta as 5×2 normalized-coordinate parameters, linear residual MSE and solve residual; do not evaluate the nonlinear solved candidate or execute any action. Rank deficiency/zero columns are retained, not filtered.

For independent non-neural re-audit, successful CUDA writes a NEW exclusive directory named <output-json-stem>_assets, containing response_columns.npy float32 (98,304,10), nominal_successor.npy and official_target.npy float32 (1,256,384), and target_minus_successor.npy float64 (1,256,384). All four numeric-only arrays are saved with allow_pickle=False; JSON contains their exact paths, shapes, dtypes, byte counts and file SHA256, not full matrices/residual lists. Total numeric payload is about5.25 MiB. Reject any pre-existing artifact directory before model loading. An audit can use np.load(...,allow_pickle=False,mmap_mode='r') to reconstruct residual, all10 column hashes, Gram, rhs, damping solution and both singular spectra without any model run. Partial files from an I/O failure are never reused/overwritten; their existence causes the next invocation to fail.

For every coordinate and BOTH fixed epsilons .01 and .005, central finite differences [F(+epsilon ei)−F(−epsilon ei)]/(2 epsilon) are computed, with identical histories/context/target. Save absolute max, RMSE, relative L2 and cosine vs the JVP. These are diagnostic only—no post-hoc tolerance gate, epsilon choice, direction selection, sample replacement or claim that disagreement is acceptable. All 20 coordinate/epsilon diagnostics remain. Nonfinite/shape/runtime failures fail closed; no CPU neural or finite-difference-as-AD fallback. Unsupported second derivative/OOM is saved with exact phase and partial counts; CUDA errors that prevent even receipt collection can still abort externally.

## Exact logical model/AD costs and memory

Successful expected counts: official encoder1 (4 RGB frames); proprio encoder1 (3 context frames); predictor51 (1 nominal +10 JVP primals +40 FD plus/minus), each with3 frames =153 context frames =39,168 patch-context tokens. These are model-boundary calls, not every internal layer/kernel. Each native predictor output contains three time slices, but only the last successor enters the response. Exactly10 JVP requests/completions and20 actual autograd.grad API requests/completions (two reverse traversals per JVP); zero direct autograd.backward API calls. It is incorrect to interpret zero backward API calls as no reverse-mode work. Scoped pass-through wrappers of torch.autograd.grad/backward record actual calls and are restored in finally; no native model forward/source is modified. Forward hooks record actual successful module calls, phase, input shapes and frame count. Derivative engine work/unsupported operator cost is exposed rather than replaced by a fabricated forward-equivalent budget.

No 98,304² output identity/dense metric/Jacobian is constructed. Gram is only800 bytes float64; retaining float64 response for reduction takes7.5 MiB in addition to float32 response. A dense float32 metric would be36 GiB and is prohibited. These retained arrays do not bound double-backward activation memory. Preparation saves the arithmetic estimate; host validation requires4 GiB free scratch AFTER module load as a fixed conservative policy, NOT a proven memory upper bound. Save actual CUDA peak allocated/reserved, free/total before/after load, walltime and complete phase trace. OOM fails without CPU fallback, shrinking/retuning or episode retry. Before/after state hashes and every source/input/checkpoint seal are compared; count/frame/phase contracts gate recorded status, not a scientific effectiveness threshold.

## Commands and freeze

From /home/gbk/code/LeWorldModel/le-wm-main, non-neural:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p5_jepa_native_action_response_20261027_v8.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p5_jepa_native_action_response_20261027_v8 --output results/p5_jepa_native_action_response_20261027_v8/manifest_frozen.json
```

Pure-array ridge and analytic sin/tensor-AD tests are non-neural fixtures, not CPU models or real predictor derivative evidence. Preparation only replays real videos/arrays, restricted checkpoint meta and headers from v7; no model construction/forward/derivative. It seals new source/test/doc, old v7 frozen sources and actual report, expected module/feature receipts and installed autograd.functional implementation/version.

Only main, after review:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p5_jepa_native_action_response_20261027_v8 --validate-cuda --manifest results/p5_jepa_native_action_response_20261027_v8/manifest_frozen.json --output results/p5_jepa_native_action_response_20261027_v8/episode1_native_action_response_cuda_main_v1.json
```

native_action_response_recorded means only that this local response/solve was measured with finite outputs and the stated receipts/counts. FD errors must be read explicitly. No physical benefit, completed RGB→D2P compiler, generalizable science P5, new heldout result, or retroactive v5 validation follows.
