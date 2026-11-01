# E3 source setup draft

Implementation replays original seeded placement and literal warm10 with single
native captures, checks existing source history/actions/states/snapshots, verifies
the original state+original-goal root hash, then binds separately authenticated
new goal and goal RGB without changing native snapshot/pose. It records actual
root guard before prefix, executes source prefix via the single-capture adapter,
and checks actual cp5 state, dynamics and pixel hash before returning.

This is a draft, not production-ready: tests currently cover only root-hash
semantics, not the full live setup callback ABI. The caller must independently
authenticate all arguments and handle setup exceptions, native partials, resource
meter, output limits and close. It has no source-closure preparation, encoder,
later planner dispatcher, final durable ledger or saved strict consumer.
