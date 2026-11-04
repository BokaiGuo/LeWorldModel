# RGB bridge host smoke: frozen before execution

Select indices 0, 1, 2, 3 from the existing v4 audit's fixed 64-trajectory
ordering, through the same `build_inputs` binding. On each fresh bridge:
verify initial image hash against v4 boundary 0; execute the same 17 recorded
actions in packet lengths 1,5,5,5,1 and compare at boundaries 1,6,11,16,17;
then reset to the original setup and repeat the exact actions using packet
lengths 5,5,5,2 and compare boundaries 5,10,15,17. Check all returned clocks,
reset count, RGB shapes and hashes. The reset image must match boundary 0.
This is 4 trajectories × 2 passes × 17 commands = 136 high-level actions;
there are no model or planner calls and no physical state is returned by the
bridge. Fail closed on any mismatch, retain attempt log, do not overwrite.

This is a debugging smoke on exposed actions and the same NVIDIA host, not
independent validation, performance evidence, or coverage of unseen models.
