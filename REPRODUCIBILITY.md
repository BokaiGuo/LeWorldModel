# Reproducibility and release boundaries

## Lightweight environment

The original LeWM installation uses Python 3.10 and the upstream
`stable-worldmodel` and `stable-pretraining` packages. A typical setup is:

```bash
cd le-wm-main
uv venv --python=3.10
source .venv/bin/activate
uv pip install 'stable-worldmodel[train,env]'
python -m pytest -q
```

The exact data-dependent experiments may require additional packages described
by the individual runner and the local environment. The repository's tests
are the first reproducibility check and do not require downloading the private
local caches that were used during exploration.

## Data and checkpoints

Raw datasets, pretrained weights, external repositories, and GPU environment
files are excluded from Git history because they are large and have separate
licensing/provenance requirements. The upstream LeWM README points to the
public Hugging Face collection and the original project repositories. Before
running an experiment, verify the upstream license and checksum of every
downloaded artifact.

## Public result bundle

`release_results/` contains only derived, small artifacts. Each artifact keeps
its protocol and claim boundary. In particular:

- `pointmaze/independent_planning_gain_gate.json` is a paired held-out
  consequence audit;
- `pointmaze/failure_diagnostic.json` decomposes horizon ranking and readout
  behavior without selecting on outcome rows;
- `pointmaze/corrected_readout_gain_panel.json` reports the architecture and
  seed panel after the evaluation repair;
- `pointmaze/representation_condition_diagnostic.json` records the
  representation-conditioned action-to-task diagnosis;
- `theory/` contains the finite-horizon and covariant-control audit outputs.

These artifacts are evidence records, not a claim of universal planning.
