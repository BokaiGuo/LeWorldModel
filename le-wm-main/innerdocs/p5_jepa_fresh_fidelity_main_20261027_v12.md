# P5 fixed three-root native factual fidelity v12

This is a NEW independent consumer, not an edit/reseal of successful v11.
Only main may execute CUDA. Implementation work runs non-neural serialized-array
tests and prepare only. Constant-interface mocks never construct a torch neural
module, load weights, execute a model, or use physics; they are not CUDA evidence.
The experiment-code skill contributes source binding, immutable logging and tests;
training/target tuning/plotting workflows are intentionally outside this task.

## Fixed cohort and input coordination

Collector: p5_jepa_factual_panel_20261027_v3. Source schema was read directly
from Euclid's newly created source in the shared worktree. This session had no
send_input/agent messaging tool; no undocumented alternate collector schema is
accepted. No collector source is patched. Its non-neural verify(root,manifest)
must agree with its frozen manifest before consumer prepare succeeds.

Exactly seeds2026102702/03/04 are consumed. Seed2026102701 and v11 outcomes are
not inputs. The remaining seed identities were already declared in the source
runtime; this is not a new unseen population or efficacy admission.
All three roots must be complete paired repeats; missing, duplicate, partial,
failed, replaced or wrong-seed roots refuse prepare, not trigger reselection.
Input CLI identifies one panel directory; each seed path is explicit and its
seed-keyed report is joined against the panel index, never array position.

Actual collector layout:

~~~
results/p5_jepa_factual_panel_20261027_v3/
  manifest_frozen.json
  factual_main_v1/
    report.json
    seed_2026102702/{report.json,journal.jsonl,schedule_env_u.npy,a/,b/}
    seed_2026102703/{report.json,journal.jsonl,schedule_env_u.npy,a/,b/}
    seed_2026102704/{report.json,journal.jsonl,schedule_env_u.npy,a/,b/}
~~~

Global status must be fresh_native_nonzero_panel_recorded; every root status
fresh_native_panel_root_recorded; role_status a/b completed,16 frames per role.
Verify panel/root/manifest SHA links, full journal chain and terminal summary,
32 frame receipts/root, all five lossless files/frame and exact independent
a/b identities. Verify all15 command JSONs per role against actual journal step
intents, packet/frame indices, raw u,100*u displacement, agent-relative target
and previous full-public-state digest. Check construct/seed/reset/15step/close
intent-return order, own seed,302 substeps/root, dt=.01, native call census.
Export manifest SHA is bound to consumer_assets_durable receipts and every
exported file SHA/size is checked. Required packets/controls/proprio exports
must exactly equal the independent v11-pure-function reconstruction.
All constructor/RNG/supplementary assets are additionally bound. Refuse symlinks.

The source input validator is parameterized HERE; immutable v11.factual_inputs
has a single-seed/schema assertion and is not monkeypatched. Reuse v11.numeric,
normalized_inputs and controls unchanged; source SHA is pinned exactly.

## Native tensor contract, not tuning

Same sealed v2 fixed15-command schedule at every seed. Physical env consumed
native u64. Dataset convention is pixel32=(100*u64).astype(float32), followed
by pixel32/float32(100), then source float32 mean/std normalization. This is
NOT an extra division of native u by100. It reproduces exported3x10 packets
bit-for-bit in NumPy. Original time-major five-step xy grouping is unchanged.

Lossless frames0/5/10 supply context; frame15 supplies only the observed target.
Proprio is [agentx,agenty,agentvx,agentvy] at the three context frames. Target
state/proprio never enters predictor history or action selection. Four fixed
arms are actual, normalized_zero, negative_normalized, reverse_steps; only the
last packet changes. Zero means zero normalized action, NOT raw-zero action.
No action optimization, target selection, threshold changes or root screening.

Reuse frozen v8.load_cuda and its original official-weight mapping, strict native
dynamics/proprio loads, imported-source closure and exact expected model receipts.
Models are constructed/loaded on explicit host CUDA, frozen eval, offline cache
only. No CPU fallback or overlay. One model load serves all three roots; model
state is rechecked after EACH root and finally. SDPA MATH and TF32-off match v11.
P5_admission remains false regardless of ordering.

## Work and failure preservation

Per root: encoder1 (4 frames), proprio encoder1 (3 context frames), predictor4
(each3 context frames). Total:3/3/12 module calls and12/9/36 frame samples.
Pre-hooks track attempts; post-hooks track completed forwards, independently
per seed. Exceptions do not refund attempted calls. Load/import/source work,
disk IO and state hashing are additional overhead, not fabricated as zero.

Before execution all three rows are initialized, including not_run rows if load
fails. A later-arm failure retains earlier costs and arrays. Ordinary root
failures do not select a replacement or hide later roots. Global unusable CUDA
may cause each subsequent fixed root to fail; all remain visible without retries.
Per-root model-receipt/source/final-hook errors are retained separately from the
original error. Failed total counts cannot be labeled a successful scientific
negative or partial-cohort success. No subset summary or inferential test exists.

Original float32 arrays are exclusively saved immediately:
official_patch_features[4,256,384] before proprio, and each
<arm>_successor[1,256,384] before cost reduction. All15 arrays on success.
Store SHA256, shape, dtype and bytes. Exclusive fsynced checkpoint_NNNN.json
files retain intermediate statuses/costs/counts; report.json is never overwritten.
Preflight failure materializes a failed all-three-row report with zero neural
work. Host crash/full disk can still interrupt durable writes; no resume/retry
or claim of crash-proof execution is made.

## Commands, main host only for final step

From /home/gbk/code/LeWorldModel/le-wm-main:

~~~bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p5_jepa_fresh_fidelity_main_20261027_v12.py
~~~

After main seals and completes the exact factual panel, prepare (non-neural):

~~~bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p5_jepa_fresh_fidelity_main_20261027_v12 prepare --panel-run results/p5_jepa_factual_panel_20261027_v3/factual_main_v1 --collector-manifest results/p5_jepa_factual_panel_20261027_v3/manifest_frozen.json --manifest results/p5_jepa_fresh_fidelity_main_20261027_v12/manifest_frozen.json
~~~

Missing actual evidence MUST refuse, never create a ready manifest from fixtures.
Manifest binds own source/test/this innerdoc, immutable v11/v8 and their source
closure, collector source/protocol and all factual assets before any new CUDA.
Do not edit source bytes after successful prepare. Different actual panel/output
requires an explicit new prepare destination; previous reports are not revalidated
under changed bytes.

Only main after review executes:

~~~bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p5_jepa_fresh_fidelity_main_20261027_v12 cuda --manifest results/p5_jepa_fresh_fidelity_main_20261027_v12/manifest_frozen.json --output results/p5_jepa_fresh_fidelity_main_20261027_v12/cuda_main_v1
~~~

Keep both result trees and original v11 sealed bytes. Successful fidelity means
saved native model diagnostic only, not counterfactual physical control or P5.

