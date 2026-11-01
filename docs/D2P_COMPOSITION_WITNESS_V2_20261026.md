# P2 composition witness v2 — implementation only

P1 NOT PASSED; P2 remains blocked. The latest sealed PushT 4×16 physical
execution passed its five integrity gates but selected programs were worse
than zero-prefix, with raw/projected metric misalignment (parent-provided
current-state boundary). This change neither revises that verdict nor opens
dependent scientific stages. No GPU, neural inference, training or adapters
were launched. CPU execution below is pure contract validation, never a neural
fallback or physical efficacy experiment.

## Inspected concrete gaps

- `reacher_d2p_program_dispatch.py:dispatch_decision` binds bank order, selected
  index and model score; it is not a general chain verifier and does not bind
  full context, root, checkpoint, metric or action domain across program edges.
- `run_reacher_d2p_p1_reconfiguration.py:_two_stage` already restores physical
  roots, checks successor identity, observes/recompiles a suffix and concatenates
  raw/normalized witnesses. Those checks are valuable, not absent. However its
  fixed two-stage result is not a bounded multi-edge composition receipt with
  each generating action and complete predictor context bound per transition.
- `pusht_d2p_metric_program_compiler.py:compile_programs` already records own-state
  context and explicit prefix/suffix parameters. It explicitly leaves bank
  provenance to its caller. Its atanh parameters, tanh live commands and raw
  environment actions must not be silently treated as the same action domain.
- `D2P_RESEARCH_REQUIREMENT_EVIDENCE_MATRIX_20261026.md` correctly leaves P2
  blocked and requires action witnesses, feedback/recompilation and physical
  increment over no-composition controls. The original proposal §4.3 requires
  generating actions for every intermediate state, not arbitrary latent targets.

## New bounded contract

`le-wm-main/experiments/spectral_reachability/d2p_composition_witness_v2.py`
introduces `compose`: explicit root/goal RGB hashes; checkpoint, encoder,
source, protocol, metric and bank SHA256 declarations; raw environment command
units/bounds, normalization identity and steps-per-token; unique edge IDs and
progress/reconfiguration kinds; exact ordered action concatenation and bounded
edge/action counts. Full latent AND incoming-action contexts are compared,
not only terminal latents. A caller-supplied trusted replay must regenerate each
postcontext from its predecessor and explicit action chunk. Each receipt binds
edge/content hashes and step offsets. No clipping, waypoint optimization,
implicit normalization or tolerance relaxation occurs. Inputs are snapshotted
and replay receives isolated copies.

The generic replay/context mechanism avoids importing sealed neural modules;
it deliberately does not reuse private task-specific compilers with different
action domains. Existing compilers can later produce witnesses through new
adapters without modifying sealed sources.

This is content binding, NOT source authentication. A dishonest callback can
invent transitions; declared hashes are not independently checked files.
`provenance_independently_verified`, `physical_efficacy_verified` and
`P2_admission` are always false. Exact deterministic replay is the supported
contract; a changed numerical runtime needs a separately frozen contract.
The portable JSON context schema is not yet an adapter for any live model.
It now supports only explicitly declared `states_with_between_state_actions_v1`:
L>=2 latent rows of width `latent_dim` and L-1 action rows of width `action_dim`,
all rectangular finite numeric lists (no strings, booleans or objects).
`context_length` must equal L; both feature widths and layout stay fixed across
edges. The absent incoming action before the first state is intentional; no
fake zero row is inserted. Other live layouts need an explicit new adapter,
not silent padding. Key names alone do not establish full model context.

### Saved forecast-vs-actual diagnostic

Independently inspected
`le-wm-main/results/pusht_d2p_sealed_prediction_residuals4_v1_20261026.json`:
4 roots × 16 candidates, explicitly `selection_changed=false`,
`P1_admission=false`, known-calibration fixed-program diagnostics only.
Median `projected_vector_error_mse` over each root's 16 candidates,
prefix → terminal: root0 0.0466 → 0.0517; root1 0.0771 → 0.2107;
root2 0.0531 → 0.2233; root3 0.0333 → 0.5084.
This supports requiring intermediate forecast bindings AND observed-feedback
recompilation rather than unchecked imagined composition. It is not causal
proof, a selection rule, a new tolerance, or evidence of P1/P2 efficacy.
Parent's new actual64-prefix/fixed-suffix replay diagnostic is pending approval
as reported; no new physical inference is claimed here.

## Remaining gates (not implemented or satisfied here)

1. Parent completes official-vs-object metric contract audit; obtain formal P1
   admission under unchanged scientific gates before any P2 scientific run.
2. New integration adapter verifies actual checkpoint/encoder/source/protocol/
   metric/bank files against declarations, bank independence, root/goal pixels,
   complete predictor history shape/window/action alignment and frozen parameters.
   Bind task/environment, reset snapshot, physical seed and split into protocol
   and full context. Never infer these from a program index or array fixture.
3. Adapter freezes raw↔normalized↔atanh conversion including action repetition,
   normalization, actuator bounds and actual execution equality. Current module
   accepts only explicit per-environment-step raw actions; it does not certify
   these conversions or temporal model semantics itself.
4. Coordinate parent before HOST CUDA replay; adapter must enforce CUDA-only,
   eval/frozen weights, unchanged hashes and real call/backward/time accounting.
   Replay each concatenated program through the actual frozen recurrence and
   independently check split/full rollout equivalence under its history policy.
5. Add physical executor receipts linking exact executed commands and root
   snapshots to observed intermediate RGB/context. Prediction is not observation:
   after real feedback, re-encode and recompile a new chain from the observed
   root, invalidating the old imagined suffix rather than relabeling it verified.
6. Freeze bounded multi-program search/refinement and all failures/abstentions;
   implement independent physical success/cost, matched no-composition SRA and
   flat-search controls with equal candidate support and resource accounting.
   Use independent held-out roots, paired uncertainty and multiplicity/kill
   gates. Unit tests, receipt hashes and model replay cannot admit efficacy.

## Validation

From `le-wm-main`:

```sh
.venv/bin/python -m pytest -q tests/test_d2p_composition_witness_v2.py
```

Initial RED: collection failed because the new module did not exist. After
implementation and context-hardening negative cases: 48 new-module tests pass;
with the existing dispatch tests, 58 tests pass. Direct context-validator tests
first reproduced 11 failures before the hardening fix. These exercise
three-edge concatenation, deterministic receipts, non-admission, root/provenance/
domain/context/transition tampering, malformed action dimensions, booleans,
nonfinite/out-of-bounds actions, empty/duplicate edges, horizon/block limits,
missing provenance, callback mutation isolation and receipt hashing.
All tests are synthetic contract tests only. No existing/shared/sealed file
was edited by this task.
