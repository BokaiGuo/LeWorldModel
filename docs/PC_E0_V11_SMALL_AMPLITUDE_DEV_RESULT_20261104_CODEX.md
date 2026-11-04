# E0 v11 fixed-amplitude ablation: development result

## Result

The preregistered eight-root panel completed all 24 plans. All arms failed the
`<0.05` terminal-distance success criterion:

| Arm | Success | Mean terminal distance | Mean tokens |
|---|---:|---:|---:|
| v11, amplitudes 0.10/0.20 | 0/8 | 0.501351 | 1904.5 |
| v11, amplitudes 0.25/0.50 | 0/8 | 0.500869 | 1906.5 |
| Three-start direct CEM | 0/8 | 0.500794 | 1920 |

On this small exposed development panel, fixed shrinking did not provide a
successful complete chain. This does not show that an adaptive trust region,
different response search, or the theory cannot work. Joint full-chain CEM was
unrestricted in both v11 arms, and the arms can consume different numbers of
tokens because the candidate rank and remaining whole batches vary. Same cap
does not mean matched wall time or equal total computation.

The panel stops here under the protocol. No roots were replaced or extended.
Task identities were checked against the four named earlier panels; broader
historical independence is unknown. This is neither an independent confirmation
nor PC2, PC3, or PC4 evidence.

## Audit and reproducibility

All 16,944 scored trajectories were replayed using both the exact endpoint
backend and a separately coded line-segment collision implementation. The
audit checked float32 actions, bounds, packet-based token and sample counts,
stored positions and scores, complete-chain incumbent retention, root/arm
identities and dependency hashes. The two implementations share the same
mathematical environment; this is not independent experimental replication.
All checks passed. No specific implementation bug was established by this
ablation.

From `le-wm-main`, the unique-output commands are:

```sh
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.pc_e0_v11_small_amplitude_dev_20261104_codex run
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.pc_e0_v11_small_amplitude_dev_20261104_codex audit
```

The commands refuse to overwrite their existing outputs. Reproduction must use
a clean copy with a distinct output location; the recorded dependency hashes
and frozen task/seed definitions must be checked before interpreting it.

Artifacts: `docs/PC_E0_V11_SMALL_AMPLITUDE_DEV_PROTOCOL_20261104_CODEX.md`,
`le-wm-main/results/pc_e0_v11_small_amplitude_dev_20261104_codex/manifest.json`,
`result.json`, `audit.json` and the per-root action traces in that directory.
