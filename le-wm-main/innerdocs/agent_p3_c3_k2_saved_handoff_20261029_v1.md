# K2 saved summary and physical handoff

K2 is exploratory after K1. This is a saved-only consumer and non-dispatching
contract, not a physical adapter, no new experiment and no gate change. It
requires the actual externally SHA-pinned complete K2 HOST consume receipt,
prepare, report, manifest, completion and all CAS files. It inherits MAIN's
strict consume as authority and rechecks the saved transport and selected
scalar/resource invariants; it does not rerun fullstrict or certify the neural
implementation independently. Missing, incomplete or RAM-only receipts reject.

All six original root/mode slots and 864 identities remain. Per slot report
all 27 caps/attempted/completed/unspent, 144 job statuses and queue request
counts; refused requests overlap unassigned slots and are not extra jobs.
Same caps are not equal actual usage or FLOPs. Full saved model plan and both
replays must agree before its first five raw actions are exposed as a candidate;
no-plan has null, with no fallback. No candidate is a physical success.

Every slot remains physical_dispatch_ready=false: the model root is not an
actual-cp5 native state. Before any separate preapproved physical adapter,
MAIN must authenticate same actual native reset/root/goal, exact model-history
to native observation guard, action-domain/normalizer roundtrip, and fixed
six-slot execution/early-stop/resource/storage contract. Without these there
is no authorized physical call, no inferred reset and no substituted action.
Physical outcomes remain null; never compare nested wall times as additive,
assert a matched FLOP budget, or infer a benefit from the model score.

After a real complete receipt exists MAIN can run from the project root:

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p3_c3_k2_saved_handoff_20261029_v1 analyze --receipt /absolute/K2_consume.json --receipt-sha ACTUAL_SHA256 --output /absolute/K2_cuda_directory --analysis /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_c3_k2_saved_handoff_20261029_v1_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p3_c3_k2_saved_handoff_20261029_v1 verify --receipt /absolute/K2_consume.json --receipt-sha ACTUAL_SHA256 --output /absolute/K2_cuda_directory --analysis /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_c3_k2_saved_handoff_20261029_v1_MAIN01.json --analysis-sha ACTUAL_ANALYSIS_SHA256
```
