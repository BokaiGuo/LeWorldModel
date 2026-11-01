# P1 observed suffix: exploratory kernel handoff

Status: blocked; no GPU planning or physical run is authorized by this artifact.
The kernel is not a runnable production entry point and does not implement an
executor, an isolated planner process, a source verifier, or a claim mechanism.

## Evidence and scope

The existing `p1_constructive_observed_suffix_cuda_20261027_v1` compiler
selects among three H1 candidates using the same nonlinear objective and
replays the winner. Its per-H1 declared budget is five predictor and action
encoder calls, one Jacobian and one SVD, including the AD forward. The new
`codex_p1_observed_h2_kernel_20261030_v1` calls that kernel twice, with the
second H1 context taken from the first predicted winner, not from cp10 RGB.
It uses one actual cp5 observation and proposes ten actions. This is an
exploratory composition, not an established full-H2 algorithm or evidence of
equal total budget. Its injected factories are responsible for enforcement;
the kernel does not itself verify actual operation receipts.

The backtrack task obtained through `replay.sources()` has top-level keys
`action_mean`, `action_std`, `binding`, `goal`, `projection`, `root_context`.
The observed compiler requires a different exact task schema, including
`root_raw_action_history`, model configuration and source hashes, fixed horizon,
packet size and feedback declaration. No adapter has been established: values
must not be guessed or merely relabeled. Check exact normalizer equality and
the action-history reconstruction as well as all authority hashes.

The existing saved physical zero-prefix run has 16 selected/zero rows across
eight logical arms. It does not supply the complete 24-condition
selected/zero/greedy roster. Existing inventory's incumbent candidate zero is
not a greedy proof. Establish full candidate-library immediate-objective scores,
deterministic tie handling, frozen greedy action identity, and authentic cp5
RGB for all eight greedy cases before treating the roster as complete.
Preserve duplicate conditions and native stops in the fixed denominator.

## Required work before a runnable handoff

1. Independently verify source hashes, original model/task and normalizer,
   all 24 prefix identities, actual cp5 image bytes and pixel identities, and
   a source-backed adapter to the observed compiler's task contract. Fail
   closed on any missing or inconsistent source. Derive the actual prefix
   latent from the frozen encoder on coordinated host CUDA; no CPU fallback.
2. Separate source/physical authority from a planner process with an explicit
   allowlisted manifest and file access boundaries. Give it only the legitimate
   task target, model context, own executed actions, and cp5 RGB/latent.
   Never expose physical state, snapshots, future RGB, rewards or outcomes.
   Test process isolation and injected forbidden fields/paths, not just schema.
3. Enforce and audit fixed independent budget guards for both packets, including
   actual encoder, predictor, action encoder, Jacobian, SVD, replay and any
   preprocessing. Pre-register whether equality covers only the suffix or also
   prefix production and total work; identical suffix caps alone do not prove
   matched total budgets. Prove the composed method and stopping rules do not
   conflict with the original contract before using it.
4. Implement a separate physical executor that atomically claims a new attempt,
   replays the fixed warmup and prefix from the authorized root, verifies
   checkpoint identity and relevant hidden-state/snapshot invariants on the
   physical side, then executes exactly the frozen ten new commands. No
   state/snapshot is sent to the planner. Log provenance, commands, terminal
   and partial failures; stop on mismatch or native termination, do not pad or
   retry, and use a separate strict saved-result consumer.
5. Bind source and runtime manifests across processes, verify the fixed 24-row
   order and resource caps, test tampering and partial/nonzero exit behavior,
   and obtain external host/GPU exclusivity coordination and main-thread review.
   Existing claims and sealed products must remain untouched.

Until these items are complete there is intentionally no host execution
command. No training, scientific gate changes, new independent roots, or P1
admission are implied by this kernel or its mock tests.
