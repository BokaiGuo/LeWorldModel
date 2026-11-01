# Complete fixed-catalog frozen CUDA orientation diagnostic

One run, all 104 inputs in sealed catalog order, batch1, duplicates charged.
104 native encoder and 104 existing projector calls, measured module entries;
zero predictor/action encoder/AD/physics/training. No outcome files consumed or
new Q fit. This is NOT a renderer-faithful orientation intervention, independent
confirmation, action optimization, P1 pass or repaired task metric.

Plan fixed before encoding: report native192 mean-square differences of every
rotation to the agent-removed clean image; preserve original-to-clean nuisance
change, all paired resampler differences and exact-input repeat discrepancies.
No angle or root selection, no fitted thresholds, no classification accuracy.
Interpret angular response together with raster interpolation artifacts already
reported in the source catalog. One unique source object patch, not four
independent physical orientation scenes.

Prepare reconstructs catalog and pins source/test/docs/native transformation and
checkpoint hashes. Run is MAIN host CUDA only; explicit prepare SHA required.
Durable per-frame/module-entry journal preserves returned latents even on error.
No retries; errors are engineering evidence, never theory negatives. Model tensor
hashes checked before/after; source closure revalidated after execution. Output
budget4MiB plus200MiB floor, walltime600s checked between images, max allocated
CUDA3GiB checked after each synchronized encode. These are abort guards, not
kernel-preemptive guarantees; no runtime/FLOPs equivalence to planners is claimed.

From inner repository, use module CLI `prepare --catalog DIR --catalog-sha SHA
--output NEW_PREPARED`; then `run --prepared NEW_PREPARED --prepare-sha SHA
--output NEW_ACTUAL`. Source catalog SHA
`122ac48c1aa144b86814d2cccb9fb223f917f12f11a592a768676e3bec5dcf5f`.
Mock tests validate numerical analysis only. Actual outcomes go in a separate
run note, never edit this source-bound document after prepare.
