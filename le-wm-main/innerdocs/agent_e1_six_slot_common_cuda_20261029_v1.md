# E1 six-slot common controller and separately paid replay

Two fixed known roots × three fixed arms; the learned arm is the saved FIRST
target, not a newly trained subgoal. All six targets are sealed before model
loading. Each slot runs the original 72-candidate H2 controller (144
predict/API/AE/proj calls), seals and reloads literal raw10, then pays four
additional calls for whole H2 and sequential 1+1 replay. All six replay
outputs are sealed before replay comparison. Planning scores candidates
inside each CEM iteration; the barrier applies to replay comparison, not those
planning scores. P2 engineering normalization is used once for every arm; it
is not the historical Hi scaler or an official Hi reproduction.

Each phase has its own actual-hook `PhaseMeter`; failed attempts are not
refunded. The consumer reconstructs optimizer arithmetic and every saved
controller/replay operation, verifies hashes, target/plan/action identities,
four-axis resource counts and the replay barrier. Checkpoint bytes are hashed
from the same immutable buffers passed to the config parser and weight
deserializer. Frozen state tensor hashes and the existing source closure are
checked before and after. MATH is requested, not independently observed;
TF32/autocast and CUDA allocation are separately guarded. No encoder/AD/physics
or training call is permitted. Saved IO and loader attestations do not
independently rerun a neural forward.

## Storage and timing bounds

The 9.77 GiB initial estimate multiplied 10,000 events by a 1 MiB uncompressed
event maximum and repeatedly expanded the same arrays. This writer preserves
complete values as exact f64 array bytes with SHA-addressed deduplication;
events and 27 payloads retain full references, and the reader verifies every
byte, offset, canonical metadata, hash-chain and absence of unreferenced data.
The integer bound covers 10,000 event records × 16 KiB metadata and 32 KiB new
array bytes, plus 27 payloads × 4 MiB metadata and 1 MiB new array bytes:
277,086,208 metadata bytes and 355,991,552 array bytes, respectively, below
the two enforced 384 MiB ceilings. The success schedule is 9,382 events:
six × (1,519 controller + 42 replay + two seals), plus four global events.
The zlib bound adds per-flush overhead to the metadata ceiling; including the
64 MiB report/close reserve, the conservative total is 873,178,201 bytes,
below the 1 GiB output ceiling. Disk free floor is 200 MiB, wall guard 900 s,
and CUDA allocator peak allocated/reserved each 4 GiB. These are enforced
limits, not equal-compute or guaranteed completion.

The pinned native configuration declares action embedding, predictor and
projection width 192 and history 3. The earlier four-arm controller's actual
full IO receipt reported 6,085 journal events, 9,384,096 archived array bytes
and 5,984,180 expanded metadata bytes. These observations corroborate the
format but do not prove unseen learned-arm results or future runtime sizes.
If a record exceeds its ceiling, the run fails closed, preserves the attempted
denominator, and does not retry or trim evidence. Filesystem failure can also
prevent final report publication; a partial output is not a successful
receipt. Wall timing in completion includes report fsync but explicitly
excludes completion fsync; it is not a full inclusive wall proof. Allocator
peak is not total GPU memory used by other processes or all driver memory. No
native production fit has been observed by these fixtures.

## MAIN-only commands

Run from `/home/gbk/code/LeWorldModel/le-wm-main`, after independent source,
tests and resource review. Output paths must be absolute and not exist.
Replace `<PREP_SHA>`, `<REPORT_SHA>` and `<COMPLETION_SHA>` with actual emitted
SHA values. `prepare` binds this source, tests and document: do not edit them
after preparing.

```sh
.venv/bin/python -m experiments.spectral_reachability.agent_e1_six_slot_common_cuda_20261029_v1 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e1_six_slot_common_cuda_20261029_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_e1_six_slot_common_cuda_20261029_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e1_six_slot_common_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha <PREP_SHA>
.venv/bin/python -m experiments.spectral_reachability.agent_e1_six_slot_common_cuda_20261029_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e1_six_slot_common_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha <PREP_SHA> --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e1_six_slot_common_cuda_20261029_v1_cuda_MAIN01
.venv/bin/python -m experiments.spectral_reachability.agent_e1_six_slot_common_cuda_20261029_v1 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e1_six_slot_common_cuda_20261029_v1_cuda_MAIN01 --report-sha <REPORT_SHA> --completion-sha <COMPLETION_SHA> --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e1_six_slot_common_cuda_20261029_v1_consume_MAIN01.json
```

These commands are an implementation handoff, not evidence that MAIN has run
them or that the scientific gate has passed. Old E1 receipts are unchanged.
