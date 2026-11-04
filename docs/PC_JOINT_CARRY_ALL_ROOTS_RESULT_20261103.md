# Joint-plan carry on all exposed development branches

The prewritten protocol ran all 17 saved branches across four exposed roots,
including repeated action chains and the three roots without any qualified
nonprogress candidate. All 17 prefix RGB preflights passed on the host
renderer. All 17 zero-initialized replans exactly reproduced their original
saved actions and full RGB histories. The initial sandbox attempt stopped
after seven prefix commands because its software-rendered prefix did not
match; it produced no output and is not included in the successful host
counts below.

| Root | Carry zero | Carry greedy | Carry raw winner | Carry spectral winner | Carry qualified |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8343 | 0.15775 | 0.07178 | 0.07178 | 0.07178 | absent |
| 8345 | 0.60215 | 0.52519 | 0.36362 | 0.49350 | absent |
| 8349 | 0.22165 | 0.14061 | 0.29287 | 0.20304 | 0.18806 |
| 8351 | 0.69230 | 0.55323 | 0.24928 | 0.55323 | absent |

Entries are final Euclidean position distances from the separately replayed
position backend, lower is better. These are all 17 source roles, not 17
independent initial conditions; some roles are identical. Across source
roles, mean distance was 0.40313 for unmodified old tails, 0.78968 for
zero-initialized CEM, and 0.32070 for own-tail-initialized CEM. Carry beat
zero-init on 17/17 roles, but beat the old tail on only 13/17; it was worse
on four roles, including a duplicated greedy/spectral pair at root 8351.
The model preferred the refined tail in all 17 cases, so improvement against
its own old-tail physical outcome was not guaranteed by model preference.

At the sole qualified root 8349, the selected prefix increased physical
distance from 1.65537 to 1.66743, and its carry-replanned result 0.18806
beat zero's 0.22165 but not greedy's 0.14061. Unconditional raw winners
beat both zero and greedy at 8345 and 8351, but their prefixes also improved
the physical position error relative to cp2. These findings do not prove
nonprogress mechanism or P1, and the old joint-search priors carry upstream
information and cost not matched by zero initialization.

Post-hoc, compare each role's selected carry score with its separately
evaluated terminal distance within the same root. Exact-tie-excluded
ranking reversals are 3/3, 2/6, 6/10 and 2/5 for roots 8343, 8345, 8349
and 8351, respectively. The lowest model-score role differs from the lowest
physical-distance role on every root; ties and duplicated roles are retained.
This is a descriptive comparison of outputs conditioned on *different,
already executed* cp7 RGB observations, not a prospective selector: acquiring
all these real successor states before choosing a prefix would itself
require counterfactual executions. It is not an independent performance
result. Reproducible analysis:
`le-wm-main/results/pc_joint_carry_posthoc_rank_20261103_DEV01.json`.

The host run used 119 RGB prefix-preflight commands, 867 native RGB
execution commands, 1,326 position-replay commands, 1,700 two-packet model
samples, one goal encode and 17 cp7 encodes. It inherited all source joint
search and source-generation costs. All 34 feedback decisions were formed
before opening new suffix outcomes in this process, but they were held in
memory rather than separately sealed to disk before execution. The
unchanged historical packet CEM scores clipped proposals but fits its
distribution to unprojected proposals; these results are conditional on
that kernel, not proof of projected-update parity. Training provenance,
observation timing and general equivalence of position and native RGB
backends remain open.

Offline audit verifies the 17 source plan bindings, 34 final CEM decisions
and decoded actions, 51 saved trajectory/prefix bindings, zero-init replay,
and position-distance arithmetic, plus artifact/code/protocol hashes.
The audit does not rerun the neural model or simulator. Related tests:
**5 passed**.

- Protocol: `docs/PC_JOINT_CARRY_ALL_ROOTS_PROTOCOL_20261103.md`
- Runner: `le-wm-main/experiments/pc_joint_carry_all_roots_20261103.py`
- Saved audit: `le-wm-main/experiments/pc_joint_carry_all_roots_saved_audit_20261103.py`
- Result: `le-wm-main/results/pc_joint_carry_all_roots_20261103_DEV01.json`,
  SHA-256 `f5977bd71e636a316935ecfce1045968adbcf6c70b100de5c825bfc20f2fe849`
