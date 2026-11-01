# P2 CP15 matched-pair saved integrity

Scope: additional CPU-only verification of the already sealed
`p2_matched_program_host_20261101_parallel` comparison and the existing reused
physical trajectories. No new model or physical calls and no training.

The existing runner compares `prefix_chain` composition with
`joint_multistart` non-composition, not with the separately named
`no_composition_sra` implementation. The existing saved verifier replays
candidate construction and checks eight measured axes: packet,
action encoder, predict, predictor, prediction projection, Jacobian, SVD,
and score. This is not equal total FLOPs, energy, I/O or elapsed wall time.

The new saved-only verifier first re-executes both original saved consumers
and compares their outputs to their externally pinned receipts. It additionally
checks, for each pair:

- The same root, task, goal image, observed CP15 context and first 15 action
  bytes, with the recorded feedback guard binding actual observation to
  compile input.
- Each saved model branch's input context and goal against the physical
  plan's saved model context and goal.
- The full 25-action physical plan's float32 command bytes and hash, and
  that the last 10 actions equal the matched planner's selected winner.
- Fixed physical attempts/returns and no early termination.

Result on the pinned artifacts: all four pairs satisfy those checks; four
slots represent three distinct root/context/goal combinations and only two
known root hashes. Bases 2 and 3 repeat the same root, context, prefix and
both selected action sequences; they are not two independent observations.
Receipt-based action lineage is inherited from the saved physical records;
this audit does not independently observe environment execution or prove
the hardware and upstream history.

The physical plans have a metadata inconsistency: their `raw_steps` field
is 30, but the saved `commands` and `raw_float32_hex` encode 25 steps, and
the physical receipt records 25 returned control steps. The H25 assembler
copies a parent plan and replaces commands without replacing `raw_steps`.
This audit reports the discrepancy; it does not alter any sealed file or
interpret 30 as the executed count.

Reproduce the verification:

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p2_cp15_pair_integrity_20261102_codexexclusive
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_p2_cp15_pair_integrity_20261102_codexexclusive.py \
  tests/test_p2_matched_program_20261101_parallel.py \
  tests/test_p2_matched_program_physical_reuse_20261101_parallel.py
```

These are development-only saved-result integrity checks. P1 and P2 admission
remain false; no independent confirmation or causal mechanism claim follows.
