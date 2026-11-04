# RGB goal geometry and physical outcome alignment

On the 624 fixed, outcome-selected terminal images from
`a0_pointmaze_failure_real_rgb_metric_20261103_DEV01`, evaluate the existing
RGB marker detector and static-camera projection on each real terminal image
and on its saved frame-10 RGB goal. Do not rerun physics, fit parameters,
alter the latent score, change the camera, or choose a subset from outcomes.
Compute RGB-position distance to RGB-goal position, and evaluate only
afterward against the already sealed simulator position, physical goal
position and latent scores. Record all segmentation failures explicitly;
do not treat a missing detection as an arbitrary distance or skip a bank.

For each bank with every detection valid, report projected image/physical
position errors, RGB-goal projection error, RGB-distance versus physical
distance reversals and ties, latent-score versus RGB-distance reversals,
and the physical distance at RGB-distance argmin. Preserve the previously
documented rendering-time, pixel-quantization and scene-calibration caveats;
image observation and integrated position need not be exactly simultaneous.
Do not use the resulting future-image argmin as a planning controller.

This is a geometry/metric alignment diagnostic on exposed, adaptive banks,
not a new planning experiment, independent confirmation or P1 result.
The detector and camera were developed on earlier exposed data and may fail
under occlusion. Report both RGB and simulator goal positions separately.
