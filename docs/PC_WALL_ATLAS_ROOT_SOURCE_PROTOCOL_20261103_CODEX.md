# Wall task: root candidate source under common Atlas continuation

Development-only test on the existing exposed exact wall-unicycle task and
the three existing seeds 20261103–20261105. The existing executable witness
establishes feasibility, not independent evaluation or physical evidence.
Freeze this protocol before running the new comparison.

Use the existing exact backend, goal, action bounds, four-step packets,
depth 4, width 2 and terminal cost. Each root bank contains zero, local
progress, and two source-specific proposals:

* spectral: signed first nonzero four-step local action-Jacobian direction,
  scaled to the existing 0.3 normalized action radius;
* random: two independently sampled bounded four-step raw action packets;
* ordinary: two packets with an opposite in-place turn in the first step and
  zero thereafter.

Every later bank is the same zero, local progress and the two in-place turn
packets, using the same compiler and deterministic ordering. Thus source is
changed only at the root; full search, scoring and continuation are shared.
Spectral and progress use the same existing geometry computation. All arms
call `progress` at every visited node; its Jacobian work is consequently
charged to every arm. This is deliberate padding and does not certify equal
natural total compute. Separately report actual compiler/backend calls,
transitions and wall time. There is no physical feedback and no externally
optimized action is injected.

For each run save root candidate actions, all selected programs, prefix and
terminal costs, replayed physical state and counts. Require action bounds,
finite scores, matching direct/split replay and identical compiler counts
within each seed. Compute all-arm outcome differences and selected root
names without post-hoc selecting seeds. Any conclusion is restricted to this
single previously exposed exact task; seeds are not independent tasks.
No change to PC or legacy P1 admission is permitted.
