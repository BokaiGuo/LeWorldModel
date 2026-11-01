# zero5 vs selected-prefix5: fixed signed matched-time comparison

## Scope / authority

NEW saved-only implementation:
`experiments/spectral_reachability/agent_mixedmode_zero_signed_comparison_20261028_v1.py`,
`tests/test_agent_mixedmode_zero_signed_comparison_20261028_v1.py`, and this doc.
No sealed file is changed. No model/environment/neural/physics/network call,
training, fitted Q, changed threshold, selected root or selected probe is allowed.
Actual saved assemble is **MAIN-only**; worker validation is RAM tests only.
No new MAIN output or duplicate physical consume is created by this delivery.

Inputs are the two actual complete physical reports and the already accepted
MAIN strict receipts, each pinned to an external file SHA:

| Input | SHA256 |
|---|---|
| `results/agent_mixedmode_zero_time_control_20261028_v1_MAIN01/report.json` | `dfba2b92f1e4c7a4f56a8d8168fca427c0595236a21588aea1a97cae0f897e8e` |
| `results/consume_zero_and_rolling_MAIN_20261028_v1_zero_MAIN01.json` | `7c22af94f7aebaabbe4857cb1797c1b7fe857020145829273fb2349b45ea8e19` |
| `results/agent_mixedmode_capability_physical_20261028_v1_MAIN01/report.json` | `57ac436ff6cd7fcffcda62e0a165782b719c0383404647f2f87c1fa97ab009c8` |
| `results/consume_mixedmode_capability_MAIN_20261028_v1_MAIN01.json` | `d1efceb6aa0eb4e72a586bb0c1ba297ecb433f12e048e7a0c966450b71156537` |

The zero receipt is the own `.consume` directly. The original receipt is
`MAIN.mixedmode_capability_acceptance.v1` with the real probe-v2 receipt nested
at `.receipt`. These are explicitly validated, not coerced into one fictitious
schema. All receipt rows/counts/errors must equal their accepted actual report;
partial or substituted inputs fail closed. The external strict-receipt SHA is
the acceptance authority: this module does NOT repeat the existing complete
physical/probe consumers or claim a new independent artifact audit.

Both prepares are checked by external actual file SHA and must bind the same
original probe prepare
`e8e4554ea2448a8c4bcf4e916934800fa8c7afe3eda335174056a551620da3d5`.
The own20 job definitions are regenerated; full context digest, task/root/goal7,
condition-specific goal pixels, renderer, parent selection and parent after-job
identity must match. Zero5 is an experimental intervention, not a model witness.
The parent selected prefix is compared as ancestry/actions, never reselected.

## Fixed arithmetic and denominator

All four original tasks `(0,off),(0,on),(1,off),(1,on)` and all five original
probes `zero,+x,-x,+y,-y` are retained. The original40 report is accepted whole;
the fixed original **after** role supplies20 matched branches, while the20
before branches are excluded by their predeclared unmatched elapsed-time role,
not by outcome. New zero20 and original selected-prefix20 each actually return
warmup10 + intervention5 + probe5. Both journals' SHA/hashchain and all matched
literal float32 raw returned commands, dtypes, termination flags and state joins
must confirm this; a nominal `elapsed=15` field alone is insufficient.

Within each task/arm, all five independently reconstructed probe starts must
have identical full state7 and exported start snapshot. Across arms these starts
need NOT be equal: state/snapshot equality is reported, never silently assumed.
Each matched branch also checks the actual physical vector (20public,
204internal physics,66explicit render,2reset and all remaining hook/close axes).
The report separately retains the full original40 resource vector and zero20
vector. Matching the20-branch physical subsets is not matching parent search,
all original work, FLOPs, or independent samples.

Math reuses the frozen original signed-analysis pure helpers, not its `analyze`
or `assemble` runners. Fixed spaces: `joint_xy` (agent+object), `agent_xy`,
`object_xy`, `object_angle_wrapped`. For every space, one unit direction comes
from the **original root-to-goal** residual and is shared across both arms.
An exactly zero original residual yields null direction/projections, not a new
after-state target. Angle differences are wrapped with atan2(sin,cos) separately
from pixels; branch-cut flags are retained and no fitted mixed units are added.

For each arm, `response(probe)=end(probe)-end(zero)`; central columns are
`[end(+axis)-end(-axis)]/(2*a)` using the original float32 amplitude held5 steps.
This is per held raw command, NOT divided by5 and NOT a full10-coordinate
Jacobian. Dot products use the original shared signed direction. Output changes
are always **selected_prefix5 minus zero5**, without absolute values or winner
selection. Zero drift and ± asymmetry are separately retained.

Output: eight task×arm groups,160 signed projections,64 signed secant scalars,
80 signed response changes and32 signed secant changes. No probe/feature/root
is dropped for a zero or negative result. No unsigned-spectrum gain substitutes
for signed task response; object-only non-improvement does not imply failure
of the full native agent+object task. Endpoint distance/native success is not
used as a proxy for capability. No trajectory-contact claim comes from an
endpoint snapshot.

Time matching removes the original10-vs15 public-step difference only. This is
still a state/velocity/contact/solver-cache mediated intervention comparison on
two known roots, not state-matched causal capability identification, new roots,
P1 admission, independent confirmation or theoretical rejection. P1=false.

## Commands / handoff

From `/home/gbk/code/LeWorldModel/le-wm-main`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
 .venv/bin/python -m pytest -q -p no:cacheprovider \
 tests/test_agent_mixedmode_zero_signed_comparison_20261028_v1.py

# MAIN executes actual saved assemble; no new physical/model job is invoked.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
 .venv/bin/python -m experiments.spectral_reachability.agent_mixedmode_zero_signed_comparison_20261028_v1 \
 --destination results/agent_mixedmode_zero_signed_comparison_20261028_v1_MAIN01.json
```

Default input paths/SHAs are the actual fixed files above. Explicit CLI options
exist for each report and receipt path/SHA; changed SHA pairs are rejected by
this fixed-calibration version. Omit `--destination` for stdout only. Output
uses a new exclusive own-prefix `.json`, ≤1MiB,200MiB disk floor, fsync and no
overwrite. MAIN-named runtime paths are reserved for MAIN, not worker use.

Actually completed: **45 RAM tests passed** (1.84s), including closed-form signs,
negative secants, zero drift, zero direction, ±2π/±4π angle invariance, full
denominator, identity/action/elapsed/resource mismatches, both real strict
receipt layouts and assemble end-to-end through RAM seams. Fixture data is not
actual physics. A duplicate-keyword error in the first RAM fixture was fixed
without changing production acceptance or scientific definitions.

Not performed by worker: actual saved comparison, real consumer replay, result
write, new prepare, neural/physics/network. Remaining dependency: MAIN review,
RAM rerun if desired, and actual saved assemble against the fixed accepted
inputs. No actual signed comparison result or result SHA is invented here.
