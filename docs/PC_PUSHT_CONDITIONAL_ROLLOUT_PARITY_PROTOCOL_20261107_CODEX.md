# PushT conditional rollout implementation parity

On the existing four exposed development tasks, compare the custom conditional
rollout to the checkpoint's native `rollout` implementation. Feed identical
saved latent observations and normalized actions; substitute only the image
encoder output in the native call. No simulator truth or success labels are
needed. Check all originally captured first-plan candidates at all five packet
horizons against saved native predictions, and all 301 selected shared-prefix
candidates per task at all three suffix horizons against a native rollout from
the same saved ten-step observation. Compare the suffix scores to the sealed
previous results and report maximum numerical differences and winner changes.

This tests rollout/action-window numerical parity, not the correctness of the
encoder, physical state reset, environment clock, model dynamics, task score,
or an end-to-end feedback policy. Keep the checkpoint frozen and verify its
fingerprint. This is a development implementation audit, not independent gate
evidence.
