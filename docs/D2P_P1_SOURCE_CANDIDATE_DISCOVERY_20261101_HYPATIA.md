# P1 candidate-source discovery: bounded blocked result

This is a read-only, bounded project-source review, not an exhaustive proof
that no suitable source exists. No root was sampled or selected, no reserved
root was opened, and no model, GPU, environment, physical execution or training
was run. Existing gates and sealed files are unchanged.

## Evidence

1. The four-slot moving proposal is defined in
   `docs/D2P_MOVING_FRESH_ROOT_PROTOCOL_20261027.md`, SHA-256
   `42f1bcbf517dfd46b8ba05525663d82efac32dbb5fb24f87a8ca572bef507a2f`.
   It is a design and fixed seed proposal, not an actual source-authorized
   candidate pool. It explicitly withholds fresh-root source execution until
   historical coverage and the source contract are met.
2. The saved fixed root manifest
   `le-wm-main/results/chat_freshroots_20261027_root_manifest.json`, SHA-256
   `3627dd2561a496ea2a0831c2aafa11337363ac1e9ca01285907c9f20e3a7b838`,
   records four proposed slots, `source_run_authorized=false`,
   `coverage_certified=false`, `freshness_certified=false`, zero certified
   fresh roots, and missing actual root identities. Its seeds do not supply
   observed state7 or make the proposal an authorized candidate source.
   The existing consumer handoff
   `docs/chat_freshroots_20261027_consumer_v2_handoff.md`, SHA-256
   `3d7e3b2c9cb56af285fbfa70d6cfff6cf8d894b17222ec8bd9c71108d8094ca5`,
   also reports the four missing actual identities and blocked source run.
3. The local PushT archive preflight
   `docs/D2P_P1_NONRESERVED_ARCHIVE_PREFLIGHT_20261030.md`, SHA-256
   `b6b13bccaa720beaeb964d60de5f98e5f66598cf2b2b4ea9384ff93e90429d58`,
   describes a bounded earlier read of four archive starting states; it
   explicitly records `source_run_authorized=false`, incomplete task/history
   binding, and unknown training relationship. Those four are not an
   authorized confirmation pool. Previously read source records are not
   reclassified as fresh; no other archive slices were opened in this review.

Search was limited to project P1/moving freshness documentation and
`le-wm-main/innerdocs` source/authority/freshness references. That bounded
search did not identify a later source authorization and actual candidate
identity manifest meeting the request. Search results and filenames do not
prove that none exists elsewhere.

## Consequence and next input

The explicit collision checker was **not run on candidate roots**, because
its input contract requires actual source-authorized identity values and
neither reviewed source supplies such a pool. Feeding the four proposal seeds
or the archive's earlier exploratory rows as if source-authorized would
change their recorded authority; fabricating state7 would be worse. Existing
checker tests remain engineering evidence only.

To continue, locate or provide the precise authorized source manifest and SHA,
its allowed scope, fixed candidate order and actual per-candidate root/reset
seed and state7 (or mark genuinely unavailable fields null), with source-to-
state provenance. Bind those bytes in a new candidate identity file, retain
all prescribed candidates without outcome-based selection, and run
`p1_explicit_collision_check_20261101_hypatia check` and `verify` into new
exclusive files following
`docs/D2P_P1_EXPLICIT_COLLISION_CHECK_20261101_HYPATIA.md`. History coverage
and training provenance remain unknown even if the checker reports no
collision; no freshness certification or P1 admission follows.
