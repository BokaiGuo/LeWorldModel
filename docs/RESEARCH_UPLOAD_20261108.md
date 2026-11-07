# Research source and report publication — 2026-11-08

This publication merges the latest local research source, regression-test source,
protocols, reports, and documentation figures into the existing public `main`.
It preserves the previous remote development snapshot and its evidence limits.
It is a publication snapshot, not a scientific-gate promotion or a claim that
the full experiment suite was rerun.

## Included

- Experiment and analysis implementations under `le-wm-main/experiments/`.
- Regression-test source under `le-wm-main/tests/`.
- Research protocols, positive and negative results, and interpretation limits
  under `docs/`, including the November 7–8 work.
- Documentation figures in PDF, PNG, and SVG form, with figure provenance
  manifests when available.
- Relevant source/configuration updates. YAML configuration files are source
  files even if their directory is named `config/train/data`.

The publication was prepared in an isolated checkout; the active research
worktree, its index, its branch, and local experiment files were not changed.
README navigation from the existing remote was preserved. Two report conflicts
were blank-line-only differences and were resolved without changing conclusions.

## Explicitly not uploaded

Per the user's revised instruction, raw datasets are not uploaded. This includes
training and counterfactual data pools stored inside `results/`, not merely paths
under a directory named `data`.

The bulk experiment archive is paused. This publication does **not** upload raw
experiment arrays, detailed trajectory payloads, model weights, virtual
environments, download caches, or the prepared full-history Git bundle. No
complete-backup claim is made, and no local research file should be deleted on
the assumption that these excluded artifacts are stored remotely.

The exact source/report snapshot and SHA-256 values are listed in
[`release_results/source_reports_20261108/manifest.json`](../release_results/source_reports_20261108/manifest.json).
That manifest describes prepared source/report files, not a full data archive.

## Checks and limitations

- Published Python source was parsed for syntax without executing experiments.
- Included source/report contents were scanned for credential patterns.
- Dataset, weight, and raw-results paths were excluded from the new publication.
- Existing research findings, input-permission boundaries, and gate status were
  not revised.
- Research tests and GPU experiments were not rerun for this publication task.
- Later concurrent research edits require another publication snapshot.
