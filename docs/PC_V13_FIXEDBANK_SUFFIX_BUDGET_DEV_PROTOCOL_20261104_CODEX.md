# v13 fixed-bank suffix-budget development protocol

Hypothesis: the earlier 119-sample, four-knot search may be too weak to
find useful continuations after generated prefixes. Test it without changing
the v13 generator or evaluating a new final-method claim.

Use four exact-Wall tasks, Cartesian product of start `[-0.5,y,theta]` for
`y ∈ {-0.23,0.21}` and `theta ∈ {-0.9,0.95}`, goal `[0.5,0.13]`. Check
against the explicitly enumerated prior panels in the v13 prospective
runner and its own tasks. Broader historical independence is unknown.
This is a new diagnostic protocol, not a retry or replacement for any
stopped panel.

For each root run the unchanged v13 compiler with a 1920-token cap,
`seed=2029141000+root`, and seal all ten first-layer candidates and actual
retention. Each candidate gets identical zero suffix initialization and
the same four-knot, batch-12 CEM with prefix held fixed:

* small: 120 proposals, stream `2029142000+root*10`;
* large: three separate 480-proposal starts, streams
  `2029142000+root*10+restart`, restart 0,1,2.

Score the zero tail once per candidate and include it as incumbent in both.
The small stream is exactly the first 120 proposals of large restart 0.
Store every actual float32 action and score, generation statistics, root,
candidate, restart, proposal counts, and full-horizon token cost; no injected
oracle or privileged plan. Report per-candidate and per-root best scores,
all-bank and retained-bank successes under the existing 0.05 threshold, and
differences between the two fixed budgets. Count tasks, not optimizer draws,
as outcome units. This is a candidate-quality diagnostic: its large total
budget is **not** an equal-cost full-planner comparison, nor evidence that
the original 119-sample four-knot configuration was sufficient or that the
larger one is globally sufficient.

After exactly four tasks and 40 candidate pairs, stop; record any failures
and do not replace roots or tune. Source files and protocol are hashed before
run. Save actions and scores exclusively; independently audit every saved
trajectory with the alternate same-environment geometry and check prefix,
budget, shared stream, incumbent, and source hashes. This is not independent
replication, PC2, or PC3 confirmation.
