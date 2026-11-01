# RGB-visible goal development protocol

Previous reverse V3 final target visibly clips the T block at the right edge.
The original center-coordinate physical guard does not certify full visual
visibility. Preserve all prior results and their original contracts.

New V4 is a developmental task-generation revision on the same known root.
Before running it, fix phases to (130,-100,4),(-100,-110,6),(-70,-70,3),
(20,55,4),(130,80,3). This is privileged task generation, not a model policy.
Keep all prior physical eligibility thresholds including20-pixel displacement.
Additionally require goal object's and pusher's renderer-specific RGB masks
to remain at least8 pixels inside the224px image. Do not lower this threshold
after seeing results. This screen detects boundary proximity, not hidden
occlusion or full shape completeness.

Goal-state fields remain evaluator-only. Subsequent task metrics may use only
goal RGB and frozen encoder evaluations. Any resulting method remains a
developmental hypothesis and needs separately frozen control and independent
confirmation. No additional model/critic/adapter training, no gate changes.
