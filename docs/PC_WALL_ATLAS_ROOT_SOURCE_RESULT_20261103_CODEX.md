# Wall Atlas root-source ablation: development result

The prewritten protocol compares three root candidate sources on the same
previously exposed exact wall task, with a common four-packet Atlas search.
The existing 16-step feasible witness replayed successfully before the
comparison. No external action chain was injected into the search.

| Search seed | Spectral terminal error | Random terminal error | Ordinary terminal error |
| ---: | ---: | ---: | ---: |
| 20261103 | 0.513514 | 0.513514 | 0.513514 |
| 20261104 | 0.513514 | 0.513514 | 0.513514 |
| 20261105 | 0.513514 | 0.513514 | 0.513514 |

All nine runs selected the shared zero root and ended at x=-0.013514, on the
starting side of the wall. The zero and progress branches tied at model
cost 0.0878987; stable ordering chose zero. The negative spectral branch
also tied that cost in every seed, while its positive counterpart was
worse. None of these ties demonstrates useful non-greedy capability or
incremental spectral benefit. The random alternatives differed by seed.
These are repeated searches on one task, not three independent roots.

Every run used 172 backend rollouts, 21 Jacobians, 336 finite-difference
rollouts inside those Jacobians and 2,080 total state transitions. These
counts match because all arms compute local progress geometry, including
where the candidate source does not need it. They do not establish equal
natural compute; this small single-process run's wall times are descriptive.
The common continuation has only zero, progress and in-place turns, so
failure could also reflect its coverage or pruning, not just the root source.
This negative result is not a learned-model, physical or admission result.

## Protocol wording discrepancy

The protocol describes the spectral radius as “normalized action radius.”
The implementation instead normalizes the raw direction by its largest
absolute component, then multiplies by 0.3 in raw action units. Translation
and rotation have different bounds, so this is not a 0.3 bound-normalized
radius. The original protocol and receipt are preserved; the result is for
the implemented raw-coordinate rule and must not be described as a
validated normalized-coordinate test. Any bound-normalized variant needs
its own predeclared contract and new result.

The added tests rerun all three arms for the first seed, replay all 12 saved
programs produced in those runs, check first-packet bindings, bounds, terminal
cost and stable selection, and verify the signed spectral pair. Together
with the existing wall tests, **6 passed**. Tests are not an independent
experiment.

- Protocol: `docs/PC_WALL_ATLAS_ROOT_SOURCE_PROTOCOL_20261103_CODEX.md`
- Runner: `le-wm-main/experiments/spectral_reachability/pc_wall_atlas_root_source_20261103_codex.py`
- Receipt: `le-wm-main/results/pc_wall_atlas_root_source_20261103_codex_DEV01.json`
- Tests: `le-wm-main/tests/test_pc_wall_atlas_root_source_20261103_codex.py`
