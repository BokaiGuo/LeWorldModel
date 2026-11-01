# Minimal executable conditional-composition ablation v2

This is an implementation-thread predeclaration under authorized implementation,
pending main review before CUDA. Parameters are not attributed to user choice.
P1/P2 scientific admission remain false. No old source, report, seal, gate or
physical consumer is modified. No training, downloads, CPU neural forward or
physics. All four roots are already-used engineering calibration roots.

## Executable intervention

Freshly compile the frozen joint-H2 SRA v2 bank at the original root/context:
192x20 terminal Jacobian, original Q/goal0, all20 bounded ridge with damping .002,
zero + three DLS scales + top9 modes x two radii x two signs =40 slots. Joint
logit ball radius sqrt(10)*.75, tanh to original raw [-1,1] box. This is not the
old per-packet ball. Slot order and stable first nonlinear-Q tie are inherited.

Joint arm executes each original two-packet proposal. Composition arm uses the
EXACT same first packet in each slot, predicts its complete successor context,
then computes a fresh 192x10 suffix Jacobian at zero suffix at that context.
It solves full10 bounded ridge (.002) with radius
sqrt(max(0,CAP^2 - prefix_logits dot prefix_logits)), then executes that suffix.
At zero remaining radius the only legal suffix is zero: still charge nominal
and Jacobian, but dispatch no SVD/solve. No prefixes pruned, cached, redrawn or
deduplicated. There is no gradient through prefix selection: this is a local
suffix compile at a fixed predicted state, not joint second-order optimization.

Select separately among all40 complete nonlinear terminal original-Q costs.
Each candidate saves its actual actions, full contexts, local derivative/solve,
and source-bound receipt. Joint receipts have one ten-action edge; composition
receipts have two five-action edges. Saved-context receipt validation is NOT
additional neural replay. Each winner also gets independent segmented and
whole H2 replay (four predicts), checked against search terminal context.
Any failed candidate invalidates that arm's deployable winner; failed work is
retained without refund or retry. There is no latent waypoint substitution.

## Resource contract, not full-budget matched

| Per root | Shared | Joint exclusive | Composition exclusive | Unique paired |
| --- | ---: | ---: | ---: | ---: |
| predict / action encoder | 4 | 84 | 164 | 252 |
| Jacobian API calls | 1 | 0 | 40 | 41 |
| mathematical scalar-output VJP directions | 192 | 0 | 7680 | 7872 |
| NumPy SVD | 2 | 0 | positive-radius suffix count (0..40) | 2 + actual count |
| bounded ridge solve | 1 | 0 | positive-radius suffix count | 1 + actual count |
| nonlinear selection evaluations | 0 | 40 | 40 | 80 |

Shared predicts = nominal2 + Jacobian-forward2. Joint exclusive = candidates80
+ winner replay4. Composition exclusive = prefix40 + suffix nominal40 + suffix
Jacforward40 + chosen suffix40 + winner replay4. Standalone attributions are
88 and168/root, NOT additive: summing them double-charges shared work by4.
Four roots unique total1008 predict,1008 action encoder,164 Jacobians,31488
mathematical VJP directions,320 selections. Predict and encoder context-token
samples are separately3024 each (three context tokens per entry), not a single
combined number. Positive-cap suffixes determine actual SVD/solve totals.
The operation journal records requested/completed work and failures; VJP
directions are not measured backward engine API calls. Root wall time and peak
CUDA allocation are reported. FLOPs/upstream RGB construction remain unmeasured,
not zero. Source verification and selected_actions array re-solves are additional
work disclosed separately, not hidden in the neural run's ledger.
Shared compile wall time and per-candidate Q-selection wall time are recorded.
Suffix Jacobian dispatch timing is asynchronous host timing, not CUDA kernel
elapsed time; CPU-transfer/solve timing and synchronized root end-to-end time
are separately labeled. Do not add these overlapping timings as a total.

The intervention replaces suffix proposals as well as adds conditional compute;
it does NOT establish equal proposal distributions, strict total compute pairing,
non-greedy P1 mechanism, independent-cohort effectiveness, or scientific P2.
It is a real minimal node compile/search/witness asset, not an admission checker.
Old strict P2's global provenance/budget schema is not silently reused: this new
schema preserves truthful per-root metric and per-arm bank-generation identities.
Poincare owns physics and comparisons; this module only exports selected actions.

## Commands (from /home/gbk/code/LeWorldModel/le-wm-main)

```bash
.venv/bin/python -m pytest -q tests/test_p2_composition_paired_20261027_v2.py
.venv/bin/python -m experiments.spectral_reachability.p2_composition_paired_20261027_v2 prepare --output results/p2_composition_paired_20261027_v2_prepared01
.venv/bin/python -m experiments.spectral_reachability.p2_composition_paired_20261027_v2 verify --output results/p2_composition_paired_20261027_v2_prepared01
```

Only after main review, main may launch the following; implementation thread
does not execute it:

```bash
.venv/bin/python -m experiments.spectral_reachability.p2_composition_paired_20261027_v2 cuda --output results/p2_composition_paired_20261027_v2_prepared01
```

Prepare seals full transitive local runtime/test/doc sources and input hashes;
verify rechecks current source, archived source, source/protocol args, and original
root/goal/Q contracts before allocation. Output writes are exclusive. Failed
started directories cannot be rerun/overwritten. Future changes require a NEW
prepared path. selected_actions(output) requires a complete source-bound report,
rebuilds the shared bank and every local solve, validates all80 action/receipt/
Q-cost slots per root and stable winners, and returns (root_sha,arm)-keyed arrays
plus report SHA and extra pure-array verification costs. It proves saved-data
consistency, not independent truth of derivatives or physical execution.

Prerequisite audit: docs/P2_COMPOSITION_PREREQUISITES_20261027_v1.md. That snapshot
identifies the broader scientific gaps; this new executable fills only its minimal
node-compilation/search/receipt integration gap, not P1 or full P2 prerequisites.
