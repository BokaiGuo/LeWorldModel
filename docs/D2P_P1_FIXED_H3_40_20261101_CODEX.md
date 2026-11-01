# Six-prefix fixed-H3 40-step exploratory controller

The fixed roster is structured, native, random, immediate_progress, zero and
same_bank_greedy. Each branch replays the same sealed first ten actions used by
the 20-step feedback experiment and verifies its original 0/5/10 checkpoints.
It then plans at actual boundaries 10, 15, 20, 25, 30 and 35, executes the
selected first packet after each plan and observes the actual RGB. Native stop
or canvas censoring terminates that branch without filling subsequent nodes.

Each planner invocation uses coherent structured H3, six gradient steps and
logical optimizer boundary 5. Physical lineage uses actual boundaries, never
the logical boundary. The old complete three-packet plan is sealed at its real
selection boundary; actual returned actions must equal its first packet. For
H3 rescoring only, the supplied four-packet previous value is
`[actual_issued, old_remaining_0, old_remaining_1, zero_packet]`. This is a
**hold-extended carry**, not the original unmodified prior plan. The append is
fixed prospectively and identical across branches and boundaries.

Each actual feedback RGB is encoded and charged separately, with rolling three
actual latents and last ten returned actions. Node costs and RGB encoding costs
are separately metered through durable ledgers. No endpoint neural diagnosis.
The fixed roster remains six even on failure or censoring. At most 36 H3 nodes
and 36 additional encodes are planned; actual usage and stop events are reported.
This is a new exploratory controller/horizon and not an old 20-step P1 pass,
an independent confirmation, or a fair matched-budget comparison with prior
20-step outputs.

Each row explicitly reports the observed cp20 and cp40 checkpoints (or null
when not reached), in addition to the full checkpoints. The cp20 observation
distinguishes the horizon/controller change at the old endpoint from any effect
of the additional physical steps. The H3 planner is more costly than the old
shrinking H2/H1 policy; comparisons are not matched-cost. No roster or
parameter adjustment based on outcomes is permitted by this protocol.
Study status is complete when all six branches have valid terminal statuses,
including native stops and canvas censoring. Full-horizon completion and
per-status counts are separate; failed or unattempted branches make the study
partial. Expected and measured node counts remain distinct.
