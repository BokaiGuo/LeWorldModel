# Causal-history RGB tick timing: independent side diagnostic

Development-only, non-planning diagnostic on the four previously exposed roots
8343, 8345, 8349, 8351. Replay their saved frame-40 initial states and the
actual two zero actions in the causal-history result. Use the legacy 2.1.2
bridge's XML, camera, target-site relocation, qvel clipping, forward, control,
five `sim.step()` ticks and post-step render, with no extra forward before
render. Record qpos, qvel and particle geom world position after every tick,
and the exact three images. Independently compare them byte-for-byte with
branch 0 saved history from the causal-history result.

Use the existing green-pixel centroid and analytic RGB position estimator.
For each of reset→step1 and step1→step2 compare image displacement/0.05 s
to qpos displacement/0.05 s and endpoint instantaneous qvel. Also compare
RGB-derived position against both boundary qpos and the positions represented
by MuJoCo's post-step geom position. These are different estimands, not
interchangeable ground truths. Identify cached-geometry timing by recording
both tick positions and geom positions, without perturbing the physical loop.
Save raw states, images, hashes, per-interval calculations, and inputs.

No model, optimizer, gate or planning inputs; no fresh root or effectiveness
claim. Rendering can use a graphics context, but CUDA is hidden and no GPU
model is loaded. The executor is an independently instrumented copy of the
bridge loop, not the original already-completed execution; byte-identical
saved images are an explicit check rather than an assumption.
