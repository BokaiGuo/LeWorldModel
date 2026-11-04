# P2 native cp5 session interface

Development code only: this is not a launcher, authorization, frozen-winner
consumer, or evidence of a real execution. It does not construct an environment.
The caller must authenticate source and environment identity and pass the
historic module, its physical Store/Meter and a fresh native environment.
An arbitrary provider or `guard` string is not root authentication.

The split loop calls the historic placement, observation, public dispatch,
return and metric checks. Warm10 RGB/actions and native state/goal root hash
must match; after plan step 15 the full historic prefix event list must match
before candidate actions can be dispatched. Each step emits intent and returned
or failure/stop events through the supplied store, in addition to the existing
native meter's operation receipts. The session closes at cp5: no sixth candidate
step and no winner suffix can be dispatched. Stops are not padded.

Fake tests establish the code path, not that the real environment or source is
authenticated. A future launcher must verify the original preparation via the
historic verification function, enforce source rechecks, isolated fresh
environments, lock, disk and per-run caps, failure finalization and saved-output
consumer. It must validate a frozen model winner independently of physical
outcomes and replay all cp5 RGB/history/state bindings before authorizing any
suffix. No such winner authority exists here; the interface therefore fails
closed at cp5.

The historic per-session meter can record its own public/native calls only if
correctly wired by a future launcher. Cross-session caps and aggregate wall
time, model/encoding work, FLOPs and full end-to-end costs are not provided by
this interface. No equal-cost, independent-confirmation, historical SRA or P2
admission claim is supported.
