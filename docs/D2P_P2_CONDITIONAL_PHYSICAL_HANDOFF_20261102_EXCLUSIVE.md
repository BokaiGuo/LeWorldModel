# Conditional model winners: known-root physical handoff

Status: **handoff and native recipe prepared; real physical runner and
winner-aware saved consumer not implemented**. Do not dispatch this recipe
through the older B0 runner or treat this document as execution authority.

The model plan SHA is
`a0d05017806eb70d2b8514cc5179a3a3026d5eb45cdb829e6ff8783a3a88064f`;
the sealed model report SHA is
`4f7fe258941ac4157671143dd025db51ffa41a70dc75c8bea7b22efc6b48e02a`.
The handoff rechecks the pinned model report, journal, source plan, isolated
historical physical proof and original B0 preparation. It checks the journal
hash chain and binds all 32 candidate/arm selected iterates to 288 optimizer
return events. Per arm it deterministically takes the smallest saved model
score among the 16 candidate winners, earliest candidate on ties; no physical
outcome enters selection. Both winners are candidate 12, iteration 8.

The two suffix SHA-256 values are:

- composition: `cfa80fa3e8c961d38babd742ac7fff140a657949e903b54c4f1f015dbd62c257`
- unmodified joint-spectral control: `abc456ddf36ecbb2cb150bc3e358f5c400d4df7701a1a9eb2e9138f89c9550d4`

Each full 25-action recipe concatenates the same previously returned
15-action float32 prefix and the appropriate 10-action winner. The B0
physical preparation's complete 106-event warmup/feedback prefix reference,
root and task are bound. CPU tests additionally compare the three capture
image pixel hashes at steps 5/10/15 to that reference; the current plan
builder does not itself enforce that particular three-frame comparison, so
the future runner must enforce it as part of its own source checks.

The sealed handoff is
`le-wm-main/results/p2_conditional_physical_handoff_20261102_exclusive_PLAN_01.json`,
SHA `df6c41689dc75340b16e297bc12a3267bb554e1dae82c0854468d2c1b69fa035`.
The separately generated recipe is
`le-wm-main/results/p2_conditional_native_recipe_20261102_exclusive_01.json`,
SHA `6009cbaf5a44169e364f330492ece7ac8111fe4cf594fa93ade6477cbf2d9555`.
Its new witness is explicitly a reference to this model report, not the
historical model witness. The original physical runner's validation rejects
these substituted plans, as tested.

## Proposed physical execution, not yet authorized by code

Use two fresh native PushT environment instances on the same B0 task/root and
goal. Each follows the original warmup10, original prefix15 and its own
model-winner suffix10. Before dispatching step 16, compare the entire actual
106-event warmup/prefix reference, including native state and RGB, to the
sealed source; on mismatch or early native stop, do not execute a suffix and
record actual stop and partial counts without padding. Continue to record
native task outcomes rather than using model MSE as physical benefit.

Reuse only the lower-level physical placement, image, action and native-label
instrumentation after new-witness-aware validation has been written. The
existing executor's fixed old schema, action authority and consumer cannot
authenticate these new winners. A complete new runner still needs: exclusive
output/no retry, durable attempted/returned/failed ledger, the original
cooperative lock policy, bounded resource guards, pre/post source checking,
exact action and prefix checks, failure/termination handling, image and
native-result consumer, and binding its own implementation and runtime.
The old physical contract's upper bounds are constructor/close 2 each,
public steps 70, physics steps 708 and explicit renders 222; new code must
enforce and verify them rather than assuming those costs occurred.

The current worktree's historical P0 source fails the original absolute-path
SHA binding. Its Git-history bytes were previously verified in an isolated
read-only saved consumer, but that receipt is not a new physical execution
and does not remove the need for a validated execution environment. Main
thread must decide and authenticate a suitable isolated runtime, then review
the new runner and consumer; do not alter shared source or loosen old checks.

The composition arm used additional compilation/model calls; this is not
equal-total-budget or the historical 40-candidate SRA comparison. B0 is an
exposed development root, not independent confirmation, a P1/P2 gate result,
or physical efficacy evidence.

## CPU-only commands

From `/home/gbk/code/LeWorldModel/le-wm-main`:

```sh
.venv/bin/python -B -m experiments.spectral_reachability.p2_conditional_physical_handoff_20261102_exclusive check --plan /home/gbk/code/LeWorldModel/le-wm-main/results/p2_conditional_physical_handoff_20261102_exclusive_PLAN_01.json --plan-sha256 df6c41689dc75340b16e297bc12a3267bb554e1dae82c0854468d2c1b69fa035
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p2_conditional_physical_handoff_20261102_exclusive.py
```

Actual validation: check returned `handoff_verified_physical_execution_blocked`;
six CPU tests passed, including winner/event mutations, three-frame source
matching, new-vs-old witness separation, old-validator rejection, and the
fail-closed `run` entrypoint. There is **no runnable real physical execution
command** yet. No GPU or new physical execution was performed.
