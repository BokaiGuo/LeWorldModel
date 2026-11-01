# E1 four-slot staged execution slice

The new pure orchestration runs fixed root0/root1 × zero/witness four slots:
four independent original 72-candidate controllers, exact per-root bank checks,
then four extra candidates and sealed choices, then four whole/segmented
replays. It uses original native Dispatch, explicit guard callbacks and hook
meters; per-phase counts survive exceptions in the caller-owned rows. Saved IO
reconstruction uses the prior independent pair consumer for every slot and
reconstructs the expected barrier values. `verify_timeline` also checks the
single ordered event stream and all three global barriers before reconstruction.
Fixture tests drive fake backends and
temporary durable journals, not real neural inference.

This is not a production runner or complete strict consumer. The caller must
write and authenticate one ordered, hash-chained global journal and pass its
complete contents to `verify_timeline`; split events alone do not prove
cross-phase chronology. Also required are source/task
authentication, model load closure and weight/precision guards, disk/VRAM/wall
census, exclusive output and durable partial reports, global journal parser and
independent reconstruction of all resource/phase counts, and MAIN-authorized
prepare/run/consume CLI. No production readiness or new evidence is claimed.
