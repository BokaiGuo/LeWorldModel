# Same-action model scoring after real cp5 RGB

For both exposed roots and all 13 fixed prefixes, take the saved L2 and
dimension-normalized-direction suffixes that were physically executed.
The two runs have already been verified to have the same cp5 RGB, goal and
action history. Feed each stored bounded normalized suffix directly to the
frozen model on host CUDA, bypassing the parameter-to-action transform;
compare L2 model terminal costs and normalized-direction costs with the
corresponding physical final joint distances. Use the same observed cp5
context for each pair and record all 26 pairs.

Before scoring, bind the normalized witness to the raw executed action via
the checkpoint statistics, and fail closed if reconstruction differs by
more than 1e-6. This tolerance does not establish bitwise identity. Verify
source hashes, model parameter hash, RGB/goal/history binding, step records
and arm denominators. Retrospective scoring cannot certify original optimizer
costs or exclude batch-dependent numerical differences. The two roots and
action outcomes were already exposed; no new selection, physical steps,
training, independent confirmation or P1 admission are implied.
