# v13 PushT new-root feedback: infrastructure-stopped development run

## Witness

The predeclared new-root evaluator-only witness completed and replayed
exactly at all 26 boundaries. It displaced the object 262.1193 pixels and
rotated it 1.9995 radians. This establishes feasibility of the constructed
15-action goal on this backend; the witness uses privileged physical
feedback and is not a planner or baseline.

## Attempted feedback

Both arms reached the common post-warmup state. At their first decision,
the isolated worker failed before issuing a model decision:

```
bwrap: loopback: Failed to create NETLINK_ROUTE socket: Operation not permitted
```

No planning output was produced and no selected action was physically
dispatched. The collector recorded both arms as failed. These are
infrastructure failures, not planning or scientific negative outcomes.
The fixed protocol prohibits retries and root replacement; both attempts
are retained and this panel is stopped.

A separate read-only `bwrap --unshare-net --ro-bind / / -- /bin/true`
preflight succeeded outside the restricted execution sandbox. This isolates
the observed issue to the interaction with that sandbox, without rerunning
the experiment or weakening planner isolation. Any separately authorized
future experiment needs an environment that permits the existing isolated
worker, a new fixed protocol, and preserved provenance.

The v13 worker and same-bank CEM control are implemented, but have not been
validated here against an actual neural decision or feedback cycle. The
existing v12/v13 compiler regression suite passed 24 tests before launch;
that is not evidence that v13 RGB feedback works. No PC gate is passed.

Witness: `le-wm-main/results/pc_jepa_v13_new_contact_witness_dev_20261104_codex/`.
Failed-run records:
`le-wm-main/results/pc_jepa_v13_contact_feedback_dev_20261104_codex/`.
Protocols: `PC_JEPA_V13_NEW_CONTACT_WITNESS_PROTOCOL_20261104_CODEX.md`
and `PC_JEPA_V13_NEW_CONTACT_FEEDBACK_PROTOCOL_20261104_CODEX.md`.
