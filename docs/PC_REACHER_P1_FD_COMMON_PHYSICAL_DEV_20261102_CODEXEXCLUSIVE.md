# FD suffix: fixed common-tail physical development contract

This is an execution proposal, not a physical result. It binds the frozen FD
model report `b76a55c6b6f8e13e7c4cbba0887e6ec6babc062c1a3245ddabf3991d39b00401`,
model plan `12ff26c6c052e02fe487bde8affd21104b09b8950f6a5340f5538fea766a2307`,
and saved-output audit `127300623499118458970a433521e5c8a3f2c1e00c55c9d2ba69986c11f7ce9c`.
The development selector chose index 4 on each of the two already-exposed
roots; the unchanged strict selector returned null on each. No physical
outcome may affect selection, action construction, branch order or omission.

## Fixed execution

In root order `[20263160, 20263161]`, run prefix 4 (`selected`) followed by
prefix 0 (`parameter_zero`), with the same saved suffixes 0 through 4 in
order per prefix. This is exactly 20 planned branches, each containing five
prefix actions and ten suffix actions. Parameter-zero is not asserted to
be raw-zero torque. There is no per-root/tail cherry-picking, adaptive
fallback, re-planning, training, model inference or GPU.

First prepare an exclusive plan: check the saved consumer and recompute its
saved-data selector/action validations, verify model plan and source hashes,
copy exact float32 branch actions, reconstruct and freeze the existing
source-supported 3×5-action warmup packets and indices, bind dependencies and
current runtime versions. The plan binds the reported root/goal RGB hashes;
it does not claim the current environment is historically identical.

At execution, rebuild the plan and compare it and its externally supplied
SHA before creating a fresh output directory. Reconstruct each root with its
frozen seed and warmup; require exact root/goal image and warmup hashes.
Capture complete MuJoCo integration and scene state, then restore it afresh
for each branch using the existing tangent physical mechanism. Require exact
restored RGB and goal plus exact captured integration/scene fields. A mismatch
is an error, never a tolerance relaxation.

Each returned step records raw action, qpos, qvel, target, distance, reward,
stop flags, and losslessly compressed RGB with SHA-256. Record the initial
state/RGB, complete planned actions and hash, attempted/returned steps,
early stop and errors. An environment stop at any step, including step 15,
is `stopped`, not an unstopped completion. No unobserved continuation is
imputed. Branch errors are retained and remaining branches continue from a
fresh restored snapshot. A root reconstruction failure prevents that root's
branches; later roots still run. Global failures and all denominators are
preserved. The report is rewritten only in the newly created run directory.

## Review and future command

From `le-wm-main`, after independent review and explicit physical-run
authorization, choose new absolute paths `PHYSICAL_PLAN.json` and `NEW_DIR`.
Preparation is CPU-only but loads the signed warmup support:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.spectral_reachability.run_reacher_pc_p1_fd_common_physical_20261102_codexexclusive \
  --model results/reacher_pc_fd_suffix_cuda_20261102_v1/report.json \
  --saved-audit results/reacher_pc_fd_suffix_saved_audit_20261102_v1.json \
  --model-plan results/reacher_pc_fd_suffix_cuda_20261102_PLAN_v1.json \
  --preflight results/reacher_pc_p1_common_tail_preflight_20261102_codexexclusive_02.json \
  --plan PHYSICAL_PLAN.json --prepare
```

Record and independently review the printed hash and the complete plan. The
execution command uses the same four input paths and `--plan PHYSICAL_PLAN.json`,
replaces `--prepare` with `--plan-sha256 REVIEWED_SHA256 --output NEW_DIR`.
The executor makes no P1 admission and does not authorize its own launch.

Planned contrasts are selected versus parameter-zero under each identical
tail; retain the 5-pair denominator per root and all stopped/failed branches.
Any native task success, physical reconfiguration, equal-compute comparison,
or independent generalization needs separate validated evidence. This
experiment uses exposed development roots and a different, optimistic model
selector; it cannot turn the strict null into a strict mechanism witness.
