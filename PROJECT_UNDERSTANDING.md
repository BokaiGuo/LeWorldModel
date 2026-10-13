# LeWorldModel Research Project Understanding

Last inspected: 2026-08-01.

This repository contains two layers:

1. The upstream LeWorldModel implementation in `le-wm-main/`.
2. A repo-local research extension that turns two long spectral-analysis proposals into code, tests, and partial PushT artifacts.

## Core Research Object

The upstream model is LeWM: an end-to-end JEPA-style world model from pixels. In code, `jepa.py` defines:

- an image encoder that produces latent embeddings `emb`;
- an action encoder;
- an autoregressive predictor that predicts future embeddings from context embeddings and actions;
- a planner-facing terminal latent cost between predicted embeddings and goal embeddings.

Training in `train.py` keeps the original LeWM objective:

- next-embedding prediction loss;
- SIGReg Gaussian latent regularization from `module.py`.

The research extension asks a more specific mechanism question:

> Are LeWM latent eigenspaces merely high-variance embedding directions, or do they carry predictable, controllable, planner-relevant structure?

The current implementation tests this through spectral transfer, tangent localization, controls, offline interventions, and an intended planner-in-the-loop intervention.

## Proposal Split

The repo deliberately separates the two proposal tracks:

- `experiments/theory_spectral_transfer/`
  - no checkpoint or dataset dependency;
  - simulates spectral-transfer operators, slope laws, predictor transfer, heat filtering, oblique-frame stress, SIGReg local anisotropy, and identifiability-vs-controllability toy cases;
  - entrypoint: `experiments/theory_spectral_transfer/run_analysis.py`;
  - coverage manifest: `experiments/theory_spectral_transfer/alignment.py`.

- `experiments/empirical_spectral_audit/`
  - consumes an existing `latents.npz`;
  - computes state/action/latent/goal/episode/phase-conditioned eigenspace audits when fields exist;
  - reports DTL, rank-wise fingerprints, PSE, PPTE, NCR, controls, multi-horizon diagnostics, controllability, observability, eigenspace transport, planner interventions, spectral planner weights, exploration score, and OOD shift when data exists;
  - entrypoint: `experiments/empirical_spectral_audit/run_audit.py`;
  - coverage manifest: `experiments/empirical_spectral_audit/alignment.py`.

`tests/test_proposal_alignment.py` guards the manifest coverage. This is important because the user's acceptance criterion has historically been proposal completeness, not just passing syntax or having one script.

## Experiment Pipeline

The intended pipeline is:

1. `theory_spectral_transfer/run_analysis.py`
   - writes `theory_spectral_transfer.json`.

2. `latent_eigenspace_audit.py`
   - loads a LeWM checkpoint and dataset;
   - encodes rollout windows;
   - writes `latents.npz` containing `latents`, `dynamics`, `planning`, `nuisance`, `predicted_dynamics`, `prediction_residual`, `actions`, and `states`;
   - optionally writes richer fields with `--rich-fields`: goals, success, episode/phase labels, physical variables, counterfactual tangents.

3. `empirical_spectral_audit/run_audit.py`
   - reads `latents.npz`;
   - runs the separated offline empirical audit;
   - supports deterministic `--max-samples` so large compressed NPZ files do not require a separate sampled artifact.

4. `planning_alignment_audit.py`
   - offline test: do latent/spectral planner costs rank closer physical endpoints correctly?
   - writes `planning_alignment.json` when the dataset is available.

5. `intervention_adapter_audit.py`
   - offline feature-space adapter intervention;
   - tests whether suppressing one target tangent, such as nuisance, selectively changes other tangent families.

6. `eval.py` with `eval.spectral_planner.enabled=true`
   - planner-in-the-loop causal test;
   - wraps the LeWM cost model with `experiments/spectral_planner.py`.

7. `train_lewm_directional.py`
   - optional directional training variants;
   - keeps all directional weights defaulted to zero in `config/train/lewm.yaml`, preserving original LeWM unless explicitly enabled.

8. `summarize_directional_results.py`
   - writes the Markdown summary table.

`experiments/run_full_spectral_transfer.sh` is the one-command orchestration surface. It supports `--resume`, `--skip-completed`, `--no-keep-chunks`, `--rich-fields`, `--empirical-max-samples`, `--run-eval`, `--eval-spectral`, and directional-training flags.

## Current Local Evidence

Tests:

- `python -m pytest -q tests` passes: `46 passed in 1.98s`.

Existing PushT artifacts under `le-wm-main/results/pusht/lewm/`:

- `theory_spectral_transfer.json`;
- `latents.npz`, about 17G;
- `empirical_spectral_audit.json`;
- `intervention_adapter.json`;
- `summary.md`;
- explicit blocker files: `planning_alignment_blocked.json`, `eval_blocked.json`.

The current `latents.npz` contains:

- `actions`: `(5945163, 10)`;
- `dynamics`: `(5945163, 192)`;
- `latents`: `(5945163, 192)`;
- `nuisance`: `(5945163, 192)`;
- plus `planning`, `predicted_dynamics`, `prediction_residual`, and `states`.

It does not contain the richer optional fields needed for goal/episode/phase/physical/counterfactual audits.

The current empirical audit sampled 200000 rows and found strong state-conditioned top-band concentration:

- `dynamics` lift at k=32: `4.711`;
- `predicted_dynamics` lift at k=32: `4.687`;
- `planning` lift at k=32: `4.624`;
- `prediction_residual` lift at k=32: `3.687`;
- `nuisance` lift at k=32: `4.431`;
- predicted-vs-true tangent mean cosine: `0.9564`.

Controls such as norm-matched random and isotropic Gaussian dynamics are near 1.0, so the top-band effect is not a trivial energy artifact.

However, nuisance lift is also high. The current result supports "latent top eigenspaces are structured and contain dynamics/planning signal", but it does not yet support a clean claim that top eigenspaces are specifically planner-relevant and nuisance-free.

The adapter audit shows increasing nuisance suppression with larger lambda. For example, at lambda `1.0`, nuisance retention is about `0.445`, while dynamics/planning/predicted/residual retentions are about `0.655`, `0.667`, `0.657`, and `0.708`. This is useful offline intervention evidence, but it is not the same as an online control improvement.

## Current Blocker

The main blocker is external data availability:

- missing dataset: `/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5`;
- `planning_alignment_blocked.json` records `reason: missing_dataset`;
- `eval_blocked.json` records the same missing HDF5 blocker for online spectral eval.

Local pretrained weights do exist:

- `/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm/weights.pt`;
- plus TwoRoom, Reacher, and Cube LeWM weights.

So the current problem is not "no checkpoint"; it is "the HDF5 dataset required by planning/eval is absent".

## Claim Boundaries

Claim-safe now:

- The repo has separate, test-covered theory and empirical spectral-analysis code paths.
- The code implements a resumable full-run surface with artifact skipping and duplicate-writer guards.
- PushT offline latent evidence exists for top-band localization of dynamics, predicted dynamics, planning, residual, and nuisance tangents.
- Predictor displacements closely match true dynamics displacements in the current sampled offline audit.
- Offline feature-space nuisance suppression has been computed.

Not claim-safe yet:

- Do not claim online spectral planner success/score improvement.
- Do not claim latent/spectral cost alignment with task progress, because `planning_alignment.json` is missing.
- Do not claim directional regularizers improve training or planning, because no validated directional-training matrix exists.
- Do not claim goal/episode/phase/physical/counterfactual mechanisms from the current PushT artifact, because the required rich fields are missing.
- Do not claim cross-environment robustness or OOD failure prediction from the current local artifacts.

## Best Next Step

Restore or generate `/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5`, then run only the blocked stages:

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main

bash experiments/run_full_spectral_transfer.sh \
  --resume \
  --output results/pusht/lewm \
  --skip-completed \
  --empirical-horizons "1 3 5" \
  --run-eval \
  --eval-spectral \
  --eval-num 50
```

After that, regenerate `results/pusht/lewm/summary.md` and re-evaluate the claim boundary. If planning/eval completes, the next serious research step is a small `--rich-fields` pilot in a fresh output directory, not another speculative code expansion.

