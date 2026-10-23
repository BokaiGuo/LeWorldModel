# PointMaze HTSC affine local-repair V4

## Mechanism diagnosed before the intervention

The legacy active arm, `effect_adapter_rgb_online_verified_local`, learns an
episode-local residual with a 12-column action/velocity polynomial basis.  It
can become *verification-trusted* after four observed pulses, although four
pulses cannot algebraically identify that basis.

The retrospective raw-trace audit at
`results/pointmaze_htsc_active_probe_conditional_20261021_cuda/mechanism_audits/identifiability_audit_20261022.json`
found 2/12 roots with executed probes.  Both had full-rank affine action
designs but rank-deficient 12-column residual designs.  The remaining 10/12
roots did not execute an information action.  This is a mechanism diagnosis,
not evidence that the historical calibration-rejected controller helped or
harmed planning.

## V4 intervention

V4 adds `effect_adapter_rgb_online_affine_verified_local`:

1. Keep the frozen effect-policy release rule unchanged.
2. If it releases, use the global policy and do not update the local chart.
3. If it declines, fit only `effect = bias + action @ Jacobian` from RGB
   successors of task-safe probes.
4. Permit a local override only when the chart is full rank, below a finite
   leave-one-out threshold frozen on calibration roots, the selected candidate is inside the observed probe-action hull,
   and it is no worse than the visual-dynamics proposal under the frozen
   visual consistency criterion.
5. Verify a released local action against its RGB successor.  A failure
   quarantines the chart; fallbacks never train it.

Thus V4 changes the local basis, not the root split, candidate bank, probe
budget, action-support geometry, or global fail-closed rule.

Before the first planning action, V4 writes a raw-trace repairability record:
probe design rank and condition number, cross-fitted local error, candidate
hull coverage, and the frozen LOO threshold.  This permits a later audit to
separate “no useful information was acquired” from “a repairable chart was
acquired but the final release gate rejected it.”

## Frozen run contract

The CUDA-only launcher is:

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
bash experiments/run_pointmaze_htsc_affine_probe_v4_cuda.sh
```

It requires a host context with a visible CUDA device and uses
`run_with_torch_cudnn.sh` to prioritize the PyTorch-bundled cuDNN.  It creates
the new immutable output directory
`results/pointmaze_htsc_affine_probe_v4_20261022_cuda/`, audits raw traces,
and writes an identifiability audit.  There is no CPU fallback.

## Acceptance rule

V4 is not a positive result until a fresh GPU run passes the raw-trace audit
and shows an actually exercised affine override on held-out roots.  Task
benefit must then be paired against HTSC fallback under the fixed physical
pulse budget; a larger release rate alone is not a success criterion.
