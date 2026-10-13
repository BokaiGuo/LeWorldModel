# LeWorldModel: transfer-sufficient diagnostics for frozen visual world models

This repository contains the reproducible research code and curated evidence
for a study of finite-horizon transfer in learned visual world models. It
extends the original [LeWorldModel](https://github.com/galilai-group/le-wm)
codebase with spectral transfer diagnostics, horizon-conditioned task
readouts, fail-closed risk gates, and independent planning-consequence audits.

The repository deliberately separates three things:

1. the frozen visual world-model implementation under `le-wm-main/`;
2. theory and diagnostic tooling under `le-wm-main/experiments/` and `docs/`;
3. small, curated result artifacts under `release_results/`.

## What is established

- The finite-horizon transfer object is the task-space operator
  \(G_h^\\Phi = D_\\Phi A_\\Phi^{h-1}B_\\Phi\), not the predictor spectrum
  alone.
- The code includes a local nonlinear applicability checker, explicit
  counterexamples, and fail-closed control gates.
- After correcting train-only latent centering and horizon-specific task
  readouts, one pre-specified PointMaze adapter audit reports a positive paired
  held-out consequence: mean gain `0.17135`, 95% CI `[0.02309, 0.33601]`,
  `n=32`, one-sided `p=0.0202`.
- The subsequent calibration-only permission gate gives the more conservative
  architecture-conditioned result: JEPA task-effect adapters retain gain
  `0.09803` (five seed clusters, 95% CI `[0.06492, 0.13277]`, sign-flip
  `p=0.0306`), while all 25 DINO checkpoints are rejected and safely fall
  back to fixed horizon 1.
- The result is conditional: it is one PointMaze environment and does not
  establish a universal world model, causal latent dynamics, or cross-
  environment transfer.
- DINO adapter runs remain a negative/boundary result under the same protocol;
  they are retained to identify representation-conditioned action-to-task
  insufficiency rather than hidden by selection.

## Reproduce the lightweight checks

```bash
cd le-wm-main
python -m pytest -q
```

The full data-dependent experiments require the datasets and checkpoints
listed in `REPRODUCIBILITY.md`; they are not committed to this repository.

## Repository map

- `le-wm-main/` — LeWM implementation, experiment runners, tests, and configs.
- `docs/` — finite-horizon transfer theorem and formal applicability notes.
- `release_results/` — compact JSON/Markdown evidence selected for public
  inspection.
- `HORIZON_CONDITIONED_TRANSFER_SUFFICIENT_CONTROL_20261013.md` — current
  method specification and claim boundaries.
- `IMPLEMENTATION_COMPLETENESS_AUDIT_20261013.md` — implementation status.
- `LITERATURE_EVIDENCE_MAP_20261013.md` — related-work evidence map.

## Relationship to upstream LeWM

The base LeWorldModel implementation and its original license are retained in
`le-wm-main/LICENSE`. The research additions are intended as an auditable
extension, not a claim that the upstream LeWM authors endorse every diagnostic
or result in this repository. Please cite the upstream LeWM paper separately.

## License

See `LICENSE` and `le-wm-main/LICENSE`. Data, pretrained weights, and external
repositories remain subject to their own licenses and are intentionally not
bundled here.
