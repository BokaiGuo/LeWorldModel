# Independent paired-composition physical consumer v2

Own source/test/doc only. The sealed paired planner/exporter is unchanged.
Implementation thread runs metadata/array/fake-guard tests and prepare/verify
only: no neural forward, CUDA, simulator creation, physical trajectories,
training, downloads or root/outcome selection. Main owns explicit execution.
This is physical calibration, not comparative effectiveness or P1/P2 admission.

## Input contract

Require the completed paired v2 report and original protocol canonical digest
`adea073583427f78ee5d832c7f647c1c77c66291d3db83c02b7776bd6beedfd9`.
Verify the exporter was already in the original pre-execution source archive,
with planner source SHA
`8d3f924c2bc38730e3d3eeedb24149cfb3a420857d6f8132bc95eeb01fe87379`.
Never patch the exporter, reseal its prepared directory, or use the old physical
whitelist runner. Reuse only the full20 physical helpers and durable guard/
serialization/encoding primitives; their source bytes are in the NEW source seal.

Call the original source-bound `selected_actions(planning)` and then verify all
four original root/context/goal0/Q identities and eight stable `(root_sha, arm)`
keys. Bind the exact ten float commands and complete one-edge joint / two-edge
composition winner witnesses, original cost, independent-H2 replay marker,
checkpoint/source/protocol/metric provenance and source report SHA. Full40-slot
selection validation and array reconstruction are inherited from the exporter.
Reject missing/duplicate roots or arms, fixtures, incomplete runs, mismatched
budgets/limits/model/root bindings, or edited receipts/actions before any env.

Upstream physical bank/preflight hashes and whole-stack history/goal RGB hashes
must match original planning roots. Source dependencies include the installed
environment class plus reset/step/render/eval_state implementations resolved
without gym.make/reset/step/render. Prepare binds actual report bytes once they
exist; no placeholder plans or new predictive evidence are manufactured.

## Physical contract

Eight branches, one replay each, fixed root-index order and alternating arm
order (even ARMS, odd reversed). Each branch resets to its original source/goal;
no winner reselection or retries. Validate ALL four finite seven-state/goals and
complete state+goal fingerprints before the first selected-path command. Check
the installed official evaluator on all roots upfront. Each branch rechecks the
whole-stack RGB and goal hashes after warmup; three raw `[2]` warmup commands are
normalized in float32 with ORIGINAL mean/std and repeated time-major to model
tokens, comparing the final two `[10]` tokens exactly to saved incoming context.
No normalized/raw coordinate substitutions, clipping, zero padding or shortened
path promotion. Reset/warmup/root-collection costs are outside selected counters,
not claimed zero or whole-pipeline compute matching.

Guard every selected raw primitive using the canonical RoyalBlue RGB contract.
Use fsync journals before step attempts and after returns; retain images/hashes,
all observed commands, attempts/returns, partials, early terminations, missing
observations, reset/label/encoding failures and setup-blocked branches. A step
which returned before a render failure remains charged even if no trace/image
exists for that return. Final integrity gates distinguish it from complete H2.
Materialize missing rootarms: never drop a branch or invent a terminal.

Persist all physical rows and `physical_report.json` BEFORE optional CUDA setup.
Only complete fixed-ten-step endpoints may be encoded; partial RGB is retained
separately and receives `blocked_fixed_horizon`, not a shortened-H2 task score.
With no complete endpoint, skip model/CUDA setup entirely. With eligible rows,
optional HOST CUDA loads frozen eval model solely for model.encode (192-vector),
overrides predict to raise, checks unchanged weights, saves attempted encoding
samples including failures. Encoding failure cannot overwrite prior physics.

## Inherited versus new costs

Planning four-root unique counts:1008 predict,1008 action encoder,164 Jacobian,
31488 mathematical scalar-output VJP directions,320 Q selections. Predict and
encoder context tokens3024 EACH. Actual SVD/solve counts come from report, never
assumed from predict equality. Shared compile4/root is charged once in paired
unique252/root; standalone attributions88/168 are not additive. Preserve each
row's full shared/exclusive/attributed receipt; no new total-budget-matched claim.

`selected_actions` performs extra pure-array SVD/solve verification (neural0).
Record its returned cost separately per load. Prepare/verify and run entry/final
recheck each invoke it; repeated metadata work is not silently merged into the
1008 planning ledger. Source hashing/JSON validation are also additional CPU/I/O
work, not FLOP-matched. New physics selected-path attempts/returns and optional
RGB encode attempts are reported separately. P1/P2 and physical efficacy false.

## Commands

From `/home/gbk/code/LeWorldModel/le-wm-main`:

```bash
.venv/bin/python -m pytest -q tests/test_p2_composition_paired_physical_20261027_v2.py
.venv/bin/python -m experiments.spectral_reachability.p2_composition_paired_physical_20261027_v2 prepare --planning results/p2_composition_paired_20261027_v2_prepared01 --output results/p2_composition_paired_physical_20261027_v2_prepared01
.venv/bin/python -m experiments.spectral_reachability.p2_composition_paired_physical_20261027_v2 verify --output results/p2_composition_paired_physical_20261027_v2_prepared01
```

Only main, after reviewing NEW source seals, may explicitly execute:

```bash
.venv/bin/python -m experiments.spectral_reachability.p2_composition_paired_physical_20261027_v2 run --prepared results/p2_composition_paired_physical_20261027_v2_prepared01 --output results/p2_composition_paired_physical_20261027_v2_main01
```

Append `--encode-terminals` only for HOST CUDA encoding AFTER physics is durable.
New output names are exclusive; never reuse a failed execution/preparation path.
Original paired prepared01/source/report and all earlier physical consumers are
read-only. No physical/P3 comparison statistics are implemented here.
