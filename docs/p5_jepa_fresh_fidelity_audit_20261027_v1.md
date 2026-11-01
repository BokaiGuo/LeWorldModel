# Independent fresh JEPA fidelity saved-array audit

Primary input is actual host-CUDA v11, v10 retained only as contrast. NEW audit
source/tests/doc/results only; all existing sources/reports/seals unchanged.
No neural/physics package imports, simulator calls, models/checkpoint loading,
training or optimization. NumPy/stdlib only. Existing checkpoint bytes may be
SHA-read as part of the frozen bindings but never decoded/loaded.

## Checks

Bind exact v11/v10 manifest FILE hashes and actual result FILE hashes, plus
every frozen source/data/input binding before/after the audit. Distinguish the
pretty-JSON manifest file SHA from CUDA report's compact canonical JSON digest.
Verify v11 attempted/completed counts1official encoder/1proprio/4predictor, all
four arm statuses completed, source/finalization error lists empty, frozen eval
CUDA model receipts identical, native strict-load receipts consistent. These
are source/hash-bound receipts, not independent neural reexecution.

Independently verify the nonzero physical collector's frozen manifest, journal
hash chain, intent/return pairing, terminal report digest, all32 durable frame
receipts, both16-frame repeats,302dt=.01Space.step receipts grouped by
role/reset/control frame. Check each actual command against frozen schedule,
its time/packet coordinates, previous full-public agent position/hash and
current absolute PD target. Check every state/proprio against its full-public
body snapshot without simulating or integrating physics.

Reconstruct source literal stats via AST (never importing dataset). Recompute
native float32 `pixel32=(100*u64).astype(float32)` then
`z=(pixel32/float32(100)-mean32)/std32`, all three packets and four model arms.
Preserve first two packets and context proprio. Exact-match both consumer
exports, frame0/5/10/15 RGB/normalized-proprio/packet input hashes and CUDA
manifest controls. Normalizedzero is native action mean, not rawzero.

Safe bounded `.npy` header/dtype/shape/payload/SHA, read-only mmap and
`allow_pickle=False`. Load complete4x256x384 official patch features and all
four1x256x384 saved successors. Recompute float64 global MSE over98304 coordinates
against feature frame index3 (actual factual frame15). Compare recorded MSE
and actual-minus-control arithmetic with atol1e-12/rtol1e-10 for CPU/GPU reduction
receipts ONLY. Ordering uses exact sign of reconstructed costs; tolerance
never turns a near difference into a scientific tie/pass.

Recompute v10 as contrast too. Check all five saved array-file hashes and
values, costs, deltas, input hashes/controls/model receipts against v11. Failure
retention/attempt-count fixes are not a claimed cause of the reverse ordering;
if saved outputs are identical the numerical finding is unchanged.

## Scope and execution

This is a posthoc arithmetic/provenance audit, not a prospectively frozen model
accuracy threshold. All four arms/32 factual frames retained. Saved features
are bound to actual CUDA/source receipts, not independently reproduced from
RGB by this audit. Official source with mapped cache weights remains subject
to the previous weight-provenance boundary; no official/HF equivalence claim.
Single fresh fixed-seed factual successor, no counterfactual ground truth.
Lower error for a control against this factual target does not demonstrate its
physical effectiveness. No target fitting, future control or P5 admission.
Truly prospective action compilation must be separately fixed before opening
its new physical outcomes; existing observed-target fits cannot be relabelled.

From `/home/gbk/code/LeWorldModel/le-wm-main`, original environment, no overlay:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p5_jepa_fresh_fidelity_audit_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p5_jepa_fresh_fidelity_audit_20261027_v1 prepare
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p5_jepa_fresh_fidelity_audit_20261027_v1 audit
```

Commands only print JSON. Saved freeze/report/findings are exclusive new files
under this basename. Test fixtures are numeric/I/O contracts, not model evidence.
