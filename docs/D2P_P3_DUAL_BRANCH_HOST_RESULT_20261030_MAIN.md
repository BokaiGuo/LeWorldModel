# P3 dual-branch host CUDA exploratory run

The independent `p3-dual-host-20261030-codex-main01` service finished with
exit status 1. The run is retained in
`le-wm-main/results/p3_dual_branch_host_driver_20261030_codex_v1_RUN_MAIN01/`.
The service was observed on the host GPU; 12 targeted driver/decision tests
had passed before launch.

Unlike the preceding R-only driver, this run accepted and executed progress
decisions. The two signal-policy rows each recorded four packet and four
encoding callbacks, with status `callback_complete_not_certified`. The four
fixed-H1 and reverse-signal rows each stopped with
`R group has no witness; both panels required`, after two packets for root 0
and three packets for root 1. The terminal report records
`candidate_complete=false`, `physical_certified=false`, `scientific_pass=false`.
These observations have not yet passed an independent full saved consumer.

For the first failed slot, the two fixed R panels had respectively one and
zero eligible candidates. In the latter the candidate with the lower reach
value also had lower immediate model cost than zero, so it did not satisfy
the sealed strict comparison requiring *higher* immediate cost and lower
reach. The recorded differences were approximately -0.01404 and -0.03919,
respectively; this is not an equality-rounding case. It does not prove the
model, action generation, physical mechanism, or all other failure causes
are correct. The original proposal allows no immediate progress, including
equality, but this run's already fixed strict protocol cannot be changed
after observing its results.

Both horizon panels are mandatory under this exploratory contract; requiring
both can stop even if one has a witness. This must be evaluated in a new
predeclared design rather than retrospectively reinterpreted as a completed
control. Actual costs differ across branches, and the failed roster cannot
support an equal-budget scheduling advantage or P1 admission.

SHA-256: `end.json`
`e9ffcd90d91f7fc37af39a951f21048b70b30d6a9017ae78470c29056d961db5`;
`decision/journal.jsonl`
`ff67c50f2f76b2d0bc3efcc774e60e42c8a55c564f6beaa4e9d706aa74e61da2`.
