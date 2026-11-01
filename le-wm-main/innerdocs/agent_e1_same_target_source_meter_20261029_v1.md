# E1 shared-target source and actual-hook metering seam

`source_inputs()` reads the pinned original common-controller prepare, report,
and persisted complete receipt, the two task gzip exports and the compiled
result gzip files. It checks per-file SHA, root/task/blind identity, original
compiled raw10 and selected path, whole/segmented context equality, prefix
context and target bytes against the original prepare and receipt. This is a
saved-source exact join inheriting the old complete receipt, not a rerun of
fullstrict, not a new physical feasibility certificate. It binds the MAIN-fixed
intervention source `31ad853d…` and its test `bd18f3ad…`; these replace the
earlier SHA. No output is a production authority.

`compare_banks` requires all 72 ordered rows and byte-identical canonical JSON,
including every field, costs, signed zero and complete contexts. It does not
independently reconstruct either bank: the future saved consumer must run the
original `verify_controller` on each complete module IO before applying it.
It must also bind both tasks/targets/seeds and independently seal banks before
evaluating the extra candidates. Differences cause failure, not tolerance.

`HookMeter` subclasses the same original actual-hook meter used by the E1
controller; separate instances cap controller/extra/replay at 144/2/4 for each
of predict_API, predictor, action_encoder and pred_proj. Predictor sample
counts are verified as well. Original nested ticket and failure handling remain
in force. RAM tests drive hooks directly, not a model; meter alone does not
prove a future backend called those hooks. Real callers must use the frozen
native backend guard with each phase and save all tickets/inputs/returns; replay
and selected raw must be sealed and reloaded. Phase errors and partial attempts
must remain durable and cannot be refunded. Four complete slots total 600 on
each axis; proposal, encoder, native, memory, wall and storage are separate.

This is an integration slice, not a prepare/run/consume CLI. Still required:
MAIN approval of the prospective intervention; a production runner binding the
original task object and backend guard, source closure, immutable model hashes,
exact bank barriers, actual-token normalization/replay, bounded durable outputs,
saved full consumer, and separately agreed native/RGB ceilings. No production
or scientific authorization follows. No historical Hi scaler claim.

Test: `.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_e1_same_target_source_meter_20261029_v1.py`. Reads only explicitly pinned saved artifacts; no NN/physics/fullstrict.
