# P2 conditional B0 physical executor: development protocol

Status: implementation and CPU-contract testing only. No physical execution
or GPU operation has been performed under this plan.

The executor first reconstructs the separately SHA-bound conditional recipe.
Inside a bubblewrap view it mounts the Git-history P0 source read-only over
its original path and calls the *unmodified* historical physical preparation
verifier, including its old witness validation. Only after that succeeds does
it create a separate derived plan with the new model-winner references. It
does not relabel the new winner as the old one or modify the old verifier.

Two fresh PushT instances use the same B0 root, task, goal, original warm10
and prefix15. The reused native `episode` checks native state and observed
RGB/action history; before step 16 it compares the complete 106-event
warmup/prefix reference and durably records a guard. A mismatch cannot
dispatch the suffix. The suffix is the literal SHA-bound ten-action winner
for the corresponding fixed arm. Actual terminal/truncated flags stop that
branch without padding. Engineering failure or cleanup failure stops the
remaining branches; all attempted native calls and errors are retained.

The run holds the existing nonblocking advisory cooperative lock through old
verification, native execution and postchecks. The lock coordinates only
participants using the same lock file; no device-wide exclusivity is implied.
The executor uses the existing native resource meter and caps, including
constructor/close 2, public steps 70, physics steps 708, and renders 222.
The 900-second bound is checked at dispatch boundaries and after postcheck;
it is not preemptive and does not include initial plan/old preparation
verification. FLOPs, energy, I/O outside the bounded store, inherited setup
and complete transitive runtime are not measured.

Plan:
`le-wm-main/results/p2_conditional_physical_executor_20261102_exclusive_PLAN_01.json`,
SHA-256 `e77ea26feaedd6b92be2b3d290e1c976b6c33114e9de9152a522a375bb9abb7a`.
The independently created historical source snapshot has SHA
`5d3463aa69038d3dbe5e6cf6c6f5ad31ceba56211e872e72a2573de9a9f18dc0`;
the working-tree shared source was not replaced. The executor source SHA is
`76e2fac6d65d7c87641523448bc60b9b2f1569ab5007ad5706521d1b12b41239`.
The plan binds an explicit source set, not all Python/native dependencies.

## Commands for later coordinated host execution

From `/home/gbk/code/LeWorldModel/le-wm-main`, check without physics:

```sh
.venv/bin/python -B -m experiments.spectral_reachability.p2_conditional_physical_executor_20261102_exclusive check --plan-sha256 e77ea26feaedd6b92be2b3d290e1c976b6c33114e9de9152a522a375bb9abb7a
```

Only after main-thread review and resource coordination, the proposed host
command is:

```sh
.venv/bin/python -B -m experiments.spectral_reachability.p2_conditional_physical_executor_20261102_exclusive run-host --plan-sha256 e77ea26feaedd6b92be2b3d290e1c976b6c33114e9de9152a522a375bb9abb7a --authorization P2_CONDITIONAL_B0_DEVELOPMENT_PHYSICS_ONLY
```

The command exclusively creates
`le-wm-main/results/p2_conditional_physical_executor_20261102_exclusive_SESSION_01/`;
the nested output is `codex_p2_multistart_joint_physical_20261030_v1_CONDITIONAL_01/`.
The root filesystem is mounted read-only in the child, except the exclusive
session directory and the pre-existing cooperative lock file. `/dev` is a
minimal bubblewrap device view. Host bubblewrap permissions, native renderer
availability, full runtime dependencies and actual lock availability must be
verified by the operator. A failed launch may leave an empty or partial
session; do not replace, erase, or retry it under the same plan.

The report status `recorded_development_unconsumed` is deliberately not an
independent consumer approval. The executor does postcheck old/source seals,
the prefix guard, native labels, and literal returned actions, but a separate
new-witness-aware saved-output consumer covering all PNG bytes/pixels,
journal-chain chronology, completion binding, and source artifacts has not
yet been implemented. A crash or storage failure can leave a partial journal
without a final report. Preserve it and classify the run as incomplete.
Do not interpret latent-model score as physical benefit.

The two model initializers had unequal model-side work. The control is a
joint-spectral candidate bank, not historical 40-candidate SRA. This is one
exposed B0 development root, not equal-total-budget evidence, independent
confirmation, P1 admission, or P2 passage.

## Actual no-physics checks

The new executor and handoff CPU tests passed (10 total); they exercise
fixed source/witness construction, different env instances, early-stop and
engineering-failure orchestration, and blocked unauthorized/current-source
entry. The older native runner's CPU fixture for early stop and prefix
mismatch passed separately (1 passed, 11 deselected); it tests the reused
native `episode`, not the new runner in a real environment. A fresh isolated
read-only *saved physical consumer* returned success without new physics;
this does not prove the new executor's bubblewrap/render configuration.

```sh
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p2_conditional_physical_executor_20261102_exclusive.py tests/test_p2_conditional_physical_handoff_20261102_exclusive.py
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_p2_multistart_joint_physical_20261030_v1.py -k stop_and_prefix_mismatch
```
