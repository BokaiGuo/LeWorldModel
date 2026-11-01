# P1 fixed-prefix suffix refinement: exploratory next candidate

Saved strict receipt `a5fd056c99860b5b43425cd68447bab52aaff77eeb3bafa3313e5c13fbb0ebc2`,
report `73e0a684f3641c343d02d2537344479e47854ae4dbdc55cf181cad14f7042a3a`.
The 72 same-iterate records include 17 strict stage-uphill, 28 strict
terminal-better, and zero intersection. Descent has 0 uphill across all four;
ascent first crosses initial stage at iteration 5 / never / 2 / 3 for slots
0..3. Slot0 ascent terminal improvement occurs at iterations 1..3, not 5..8.
All statements are model-only, two known roots, descriptive and exploratory;
the 72 include duplicate iteration0 across the two arms, not 72 independent
experiments. No physical success, global infeasibility, or P1 claim follows.

In the prior code both packets update simultaneously: the suffix gradient uses
the *previous* prefix's context, and the next score uses the updated prefix.
That wiring provides a concrete reason to test suffix refinement after fixing
the prefix; the saved numbers do not prove it caused the lack of witnesses.
This new proposed method retains **all** source iterations 1..8 for **both**
arms across all four slots, no outcome-dependent filtering. For each of 64
groups it replays the exact prefix, starts from that source iterate's suffix,
then performs eight original eta=1 chart/projected suffix gradient steps on
the fixed prefix context. It scores all nine suffix iterates and replays the
best Q, including groups whose fixed prefix is not uphill. The strict original
conjunction remains stage Q > initial root Q and terminal Q < same-slot
original winner Q; none means null. Neither group nor root is chosen from a
physical outcome; no extrapolation beyond original packet balls or new Q.

Each group pays prefix replay1 + suffix scores9 + suffix AD forwards8 +
independent suffix replay1 = 19 predict and 8 reverse. New 64-group budget is
1216/512; inherited source costs 288/128, combined 1504/640, without counting
earlier ancestry. Equal group/API counts across arms are not equal FLOPs or
wall time. The fixed 576 candidate slots are not comparable to a 72-slot run
as equal total budget. Output failures retain the full group and candidate
denominators with no padding or fallback.

This is an exploratory callback kernel, not a production runner or a valid
retrospective preregistration. A future reviewed host must fully reconstruct the
original receipt, authenticate same-root/time/goal/source and native IO, seal
model/class/precision, enforce the full resource vector and independently audit
scalar losses and partials. The pure join assumes its caller performed original
full consume. No GPU, physics, training, freshness or P1 admission is supplied.
