# RGB feedback tail retention: integration result

The new runner replayed the sealed root plans through the existing RGB
PointMaze bridge, scored each old five-action tail at the cp10 observation,
and compared it with a new tail searched in the same refreshed context.
One sample was spent on the old tail and 95 on the new search for every arm.
The complete decision record was written before executing any selected tail.

The smoke run executed one chain; the full development panel executed all
32 chains (four reused roots, two budgets and four methods). Warmup images,
cp10 images, frozen model, normalization and sealed dependencies passed the
runner checks. All 32 decisions selected the new tail on model score, so
this panel did not exercise an actual environment execution of the old-tail
branch; unit tests separately exercise its retention and budget accounting.
The 32 settings are not independent roots. No physical terminal-distance
comparison was performed, so these results do not establish physical
improvement or gate passage. Full end-to-end cost certification remains false.

Results are stored separately from the original experiment:

* `le-wm-main/results/pc_rgb_feedback_tail_retention_20261103_codex_full_cp10.json`
* `le-wm-main/results/pc_rgb_feedback_tail_retention_20261103_codex_full_execution.json`
* `le-wm-main/results/pc_rgb_feedback_tail_retention_20261103_codex_full_images.npz`

The smoke files use the same prefix with `smoke` instead of `full`. The
original sealed root and feedback artifacts were not overwritten.
