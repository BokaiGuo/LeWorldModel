# Wall Atlas bound-normalized correction: development result

The correction was specified after the raw-coordinate outcome was already
known. It is a new development test, not a replacement of the earlier receipt.
The same exposed wall task and three repeated search seeds were used.

| Root source | Earlier raw-coordinate error | Bound-normalized run error |
| --- | ---: | ---: |
| Spectral | 0.513514 | 0.505186 |
| Random | 0.513514 | 0.513514 |
| Ordinary | 0.513514 | 0.513514 |

These values are identical across the three seeds. The corrected spectral
search selected the negative spectral root, four straight backward steps of
`-0.09`, with prefix model cost 0.616533 versus initial 0.333333. Its final
position was `[-0.005186, 0]`, still on the starting side of the wall. The
other arms selected zero and ended at `[-0.013514, 0]`. Thus a small terminal
distance advantage and a nonprogress prefix are observed on one exposed
analytical instance; neither crossing, success, changed control capability,
independent generalization nor an admission gate is established.

The corrected radius uses `max(abs(action)/HIGH)=0.3`; earlier code used
`max(abs(action))=0.3`. Each arm used 2,080 state transitions, 172 rollouts
and 21 Jacobians, including 336 finite-difference rollouts. All arms are
deliberately charged for local geometry, so these counts do not establish
equal natural total compute. Random and ordinary action banks, programs,
counts and selection were independently checked against the old runner for
the first seed. All corrected spectral branches were replayed and checked
against saved contexts and cost. Combined with the original root-source and
shadow-renderer saved-result tests, **8 passed**.

The task's already known successful witness requires a detour; the selected
spectral program did not discover one. A common same-suffix intervention and
separately optimized suffixes would still be needed for a capability claim,
and even those would remain development evidence on this exposed task.

- Protocol: `docs/PC_WALL_ATLAS_BOUND_NORMALIZED_PROTOCOL_20261103_CODEX.md`
- Runner: `le-wm-main/experiments/spectral_reachability/pc_wall_atlas_bound_normalized_20261103_codex.py`
- Receipt: `le-wm-main/results/pc_wall_atlas_bound_normalized_20261103_codex_DEV01.json`
- Tests: `le-wm-main/tests/test_pc_wall_atlas_bound_normalized_20261103_codex.py`
