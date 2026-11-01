# Independent two-stage witness producer v1

Only a candidate/witness producer, NOT P1 admission. This version's algorithm
was predeclared under authorized implementation before examining new witness
benefits. The parameters below are implementation protocol choices, not a
claim that the user personally selected them. No neural/physics execution is
performed by the implementation author. Main owns actual HOST CUDA.

The existing direct core, driver, source and all sealed files remain unchanged.
Only this new source, its tests and this innerdoc are added. There is no learned
adapter/readout, simulator state in planning, threshold default or download.

## Fixed finite algorithm

Use exactly the two original actual calibration tasks acquired by the sealed
`p1_direct_cuda_calibration_main_20261027_v2` driver. Preserve each task_id,
native192 goal from goal RGB, identity Q, action mean/std, full3-state/between2
normalized-action context, causal zero-padding and per-packet cap
`float(.75*sqrt(5))`. No transplantation of old goal-conditioned Q or old roots.

1. At root zero logits evaluate a genuine complete H2 prediction with independent
   whole and segmented replay (6 predict/action_encoder calls). Calculate a
   fresh FULL terminal192×20 Jacobian with BOTH packets in the graph (2 AD
   forward calls and 192 reverse directions). Use its FIRST TEN columns to
   compile prefix response under the actual terminal H2 objective, not an H1
   greedy objective. This initial J is NEVER reused as a local suffix J.
2. One thin NumPy SVD of QJ_prefix/sqrt(192), one bounded L2 ridge solve with
   damping .002, objective MSE-linearization+.002||step||². Canonicalize each
   right vector by its maximum-absolute entry; ridge coefficients preserve the
   corresponding sign. Prefix slots in fixed order: zero (1), bounded DLS
   scales .25/.5/1 (3), two radii cap/3 and cap × top3 directions × signs -/+ (12).
   Total16. Float32 feasible inward rounding is numerical, not a Q threshold.
3. For EVERY prefix slot, including duplicates: actually execute its packet
   from root (1 call). At that exact full predicted mid-context, execute zero
   suffix nominal (1 call), then build a NEW LOCAL192×10 suffix Jacobian with
   one new AD forward (1 call) and 192 reverse directions. Save forward/context
   before any later derivative/solve failure. Compile another one-SVD/one-solve
   bounded ridge under the SAME goal/Q and independent original suffix ball.
   Frozen prefix differentiation is deliberately absent in this local stage;
   this is own-state suffix recompilation, not an approximation to joint H2 GN.
4. Execute all four suffixes zero/.25/.5/1×own-context DLS, each nonlinearly
   from its own actual shared prefix context (4 calls, no nominal-as-candidate
   cache). Rank all64 composed paths by original nonlinear terminal Q MSE,
   retaining stable first ties. No greedy-prefix filter, early stop, retries,
   dedup, rescoring with pose or posthoc slot selection.
5. Winner pays a NEW nominalH2 + independent wholeH2 + independent segmentedH2
   (6 calls). Its commands, complete two contexts and original Q cost must equal
   the saved search path exactly. The exported `witness` is directly compatible
   with the old `check_prediction` schema; it is not a fresh physical execution.

Every packet is 5×2 time-major raw commands from NumPy float32 tanh, raw[-1,1].
Local derivative is wrt raw controls at zero, with normalization inside graph;
tanh'(0)=1. Root full-H2 logit derivative uses the original sealed core's exact
raw-command/tanh chain. Native normalization is compared exactly BEFORE each
neural packet. All parameters/model buffers remain frozen eval CUDA; MATH SDPA
is selected for AD/replay. No CPU neural fallback. Real CUDA determinism is
not established by NumPy mocks.

## Resources (explicit, NOT full compute matched)

| Resource per root | Producer | Independent witness | New total |
| --- | ---: | ---: | ---: |
| predict / action_encoder, each |120|6|126|
| Jacobian invocations |17|0|17|
| mathematical VJP directions |3264|0|3264|
| NumPy SVD / bounded ridge solve, each |17|0|17|
| prefix-bank generation |1|0|1|
| suffix proposals / nonlinear score+selection, each |64|0|64|

Producer predict=6 root replay+2 root AD+16×(1 actual prefix+1 suffix
nominal+1 suffix AD+4 actual suffix candidates)=120. Thus the two roots add
252 predict +252 action_encoder, 34J (2 full192×20 +32 local192×10), 6528
mathematical reverse directions, 34SVD/solve and128 score/selections. Every
native model.predict must enter predictor/action_encoder exactly once; actual
module prehooks enforce phase-specific caps BEFORE dispatch. Failed work is
not refunded, and failures stop the run without replacement roots/candidates.

Whole3-token context action_encoder and native pred_proj are included in each
native predict path. VJP count is not engine kernel multiplicity/FLOPs. Bounded
ridge CPU coefficient evaluations, bracket doublings and 70 bisections when
active are explicitly recorded; they are not extra unreported SVDs. QJ/CPU
arithmetic, normalization receipt checks, transfers, serialization, hashing,
fsync, CUDA synchronization and model loading remain overhead, not FLOP-matched
by the logical call counts. Each selection pays exactly one nonlinear Q score.

The original source reset/render and reset-internal physics accounting, native
8 encoder+8 projector acquisition, and saved direct work are inherited by
source-bound accounting receipt, not repeated or free. No new image encoding
or physics is used here. Root H2 nominal work belongs to PRODUCER, not a shared
free resource. Shared prefix transition is charged once per prefix and reused
only among its four nonlinear suffixes. This is not equal candidate/predict or
total compute matching against GN/GD. Future paired comparisons must separately
attribute inherited acquisition and this additional search budget.

## Source binding and eligibility boundaries

`collect` calls original direct prepared verify and source-only preflight.
It reads ONLY original task files, source scan, image acquisition journal and
aggregate direct report provenance/resource fields; it NEVER opens direct
per-method trajectories/costs, old P3 outcomes or future physical labels. Task
latent histories and goal must equal all8 original native image-completion
receipts, whose RGB hashes must equal original raw source frames. Normalizer,
identity Q, model/config hashes, task-contract IDs and raw-padding arrays are
checked against original prepared protocol. Complete actual root/task mapping
is independently rechecked by original `to_admission_scan`, with no fake IDs.
The original aggregate report must certify unchanged model/verified sources
and complete original image acquisition; direct efficacy is NOT an input.

The source remains `calibration_exploration`, with repeated-current RGB causal
padding (one observed frame, zero public warmup), NOT observed-transition history
or fresh confirmation. **Main's independent audit reports all four GN/GD runs
still improve from B1 to B2. They cannot be called direct stall.** This producer
does not alter B budgets, choose a plateau/escape/target threshold, or mark any
root eligible. Actual finite-budget stall evidence remains insufficient. A good
witness, if found, does not fill that gap. Independent predeclared threshold
authority/calibration/confirmation and genuine physical mechanism/consequence
remain absent. All output P1_admission/model_task_eligible flags stay false.

Producer/witness journals retain old exact core fields and evaluation_id
`producer` / `witness`. Extended journals separately count VJP, solve, generation,
proposal and score/selection. Future admission-row assembly must explicitly
freeze a new policy binding THIS producer source and these producer budgets,
while retaining the original direct trajectories/budgets. Do not alter old
policy seals or treat the new output as an already-completed admission row.

## Durable failures and storage

NEW prepared/run paths only; no resume/overwrite. Source-bound prepare seals
this code/tests/doc plus runtime dependencies and all original task bindings.
Run is explicit main-authorized. Every attempted dispatch is fsynced before
work; returned packet/forward/J rows and candidate/incumbent receipts are fsynced
as produced. Root report preserves all candidate paths, each local derivative,
primary error and completed/failed attempted counts. Later synchronization,
final model/source checks and report errors remain separate; no false success.
A storage failure cannot promise final-report durability, but stops further
work and retains already-written journals. Never auto-reseal/retry an old path.

Global run output≤64MiB, each journal≤16MiB, each serialized item≤8MiB.
EVERY write (reports included) checks global quota and leaves200MiB free.
Prepare checks complete snapshot/metadata headroom. No model files/image banks
are copied. This implementation does not itself create prepared/results assets;
main invokes the commands below after source review.

## Commands

From `/home/gbk/code/LeWorldModel/le-wm-main`; replace `<DIRECT_PREPARE_SHA>`
with independently retained SHA256 of the existing direct prepared receipt:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_p1_two_stage_witness_20261027_v1.py

.venv/bin/python -m experiments.spectral_reachability.p1_two_stage_witness_20261027_v1 prepare --direct-prepared results/p1_direct_cuda_calibration_main_20261027_v2_prepared01 --direct-prepare-sha <DIRECT_PREPARE_SHA> --task-directory results/p1_direct_cuda_calibration_main_20261027_v2_cuda_main01 --output results/p1_two_stage_witness_20261027_v1_prepared01

.venv/bin/python -m experiments.spectral_reachability.p1_two_stage_witness_20261027_v1 verify --prepared results/p1_two_stage_witness_20261027_v1_prepared01 --prepare-sha <NEW_PREPARE_SHA>

# MAIN ONLY after review; NOT executed by the implementation author:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p1_two_stage_witness_20261027_v1 run --prepared results/p1_two_stage_witness_20261027_v1_prepared01 --prepare-sha <NEW_PREPARE_SHA> --output results/p1_two_stage_witness_20261027_v1_cuda_main01 --authorize-host-cuda
```

Mock tests establish engineering contracts, state-dependent local recompilation,
call accounting, exact old witness schema and fault/partial behavior only.
No actual witness quality, direct stall, model eligibility or P1 evidence has
been obtained by those tests.
