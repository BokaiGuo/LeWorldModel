# P1 fixed backtrack physical exploration, 2026-10-30

This is known-root exploration after model outcomes were observed. It is not
fresh-root confirmation, a capability experiment, equal-budget evidence, or
P1 admission. The 16 logical rows were all executed without deduplication.

## Execution and code failure

The first attempt, `p1_backtrack_physical_runner_20261030_MAIN01`, failed before
reset or any physical action: the meter's candidate `role` collided with the
original placement callback's reset `role`. It is an engineering failure,
not a negative model or physical result. The output was retained. The candidate
field was renamed `candidate_role` and a nested reset-role regression was added.

The separately sealed second attempt used prepare SHA256
`f387cd9cf094c09efe8219b6fe7fc189fdef6fc72491adfe5f30b040f14a8e81`.
Its [report](../le-wm-main/results/p1_backtrack_physical_runner_20261030_MAIN02/report.json)
has SHA256 `26802255a8192e95ae3d67a089ee673ba6e8a7331786481fefde9f4fd1e84de9`.
All 16 independent environment instances reached the 15-step horizon;
400 public steps include 160 warmup and 240 planned steps, with 4,064
internal physics steps and 1,296 explicit RGB renders. No neural model was run.

The separate saved consumer verified the hash-chain journal, literal requested
and returned actions, all PNG file/pixel hashes, checkpoint time indices,
operation census and recomputed native metrics. The consumer itself initially
searched for a nested receipt field at the wrong event level; after correcting
that analysis bug it passed. This does not prove hidden state identity beyond
the fields recorded or validate later model-RGB inference.

## Descriptive observations

At plan step 15, the recorded native success predicate was true for logical
rows 5 and 7, both with the same root, on condition, and identical 15-step
action SHA `e33b2bfce09336385e8626782db42a4a50142f3d4329f3f7cc2c118b76fad5d0`.
They are two executions of the same task/action, not two independent tasks.
The corresponding incumbent rows 4 and 6 did not satisfy that predicate at
step 15. The experiment is not a strong-baseline or equal-compute comparison.

For root slot 1, selected rows 9/11 have native7 distance 269.55 at the root,
294.47 at step 5 and 121.74 at step 15; rows 13/15 have 269.55, 285.04 and
91.99 respectively. Native7 includes velocity and is not the success
predicate. These values are descriptive and do not by themselves establish
physical task regression, a change in future controllability, or P1 passage.

## Remaining dependencies

The preset 16 × 21 physical common-prefix capability probes have not been
executed. Open-loop model predictions, actual RGB latents, and predictions
after real-RGB context refresh have not yet been compared on these new
trajectories. The proposed scoring or feedback changes remain exploratory;
after fixing and freezing them, independent roots and fair strong-baseline
comparisons are required before any scientific admission.

The saved consumer plus runner/contract/episode/guard/source tests passed
17 tests. They exercise these recorded paths and selected tampering cases;
they are not proof that every possible implementation error is excluded.
