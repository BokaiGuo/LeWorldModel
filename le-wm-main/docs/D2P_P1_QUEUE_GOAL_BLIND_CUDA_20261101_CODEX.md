# Development queue blind CUDA encoding

Only the sealed reused development seed 301000001 is accepted. Source verification
rechecks the actual reset, same-instance goal, journal, frame and action history,
and matches the sealed three-field blind archive byte-for-byte at array level.
The original encoder's fixed whitelist is unchanged and is not bypassed by
pretending this input is approved there.

This independent runner reuses the original native checkpoint, image transform,
encoder/projector dispatch guard, four-image ledger, task construction and
normalization. It requires the project cooperative GPU lock and CUDA; no CPU
neural fallback, predictor, action encoder, backward pass or physical step.
Identical observed pixels at steps 0, 5 and 10 are still encoded separately.
The source receipt and software/model hashes are rechecked before and after.
Failures retain a partial report; no silent retries. Saved verification checks
the source chain, ordered module journal, ledger, latent hashes and recomputes
the actual exported task from those latents and the real raw action history.

This is development encoding, not a control experiment or P1 admission.
Historical exclusion, training membership and independent confirmation remain
unknown or false; the public normalization's training provenance is unverified.
