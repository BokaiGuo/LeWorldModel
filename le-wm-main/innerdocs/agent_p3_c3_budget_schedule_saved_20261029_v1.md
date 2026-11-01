# C3 saved budget and scheduling analysis

This is a saved-only extension of the existing preregistered C3 analysis, not
a replacement or another fullstrict consumer. It uses the externally SHA-pinned
MAIN strict HOST receipt as authority, reuses the frozen prereg receipt checks,
and opens the frozen compact HOST Reader to authenticate the manifest,
completion, report and saved result objects. It does not load/run a model,
physics, training, or the producer's full saved-mathematics reconstruction.
An externally fabricated receipt is not made authentic by this tool.

All 12 fixed known-root × K1/K17 × mode slots remain present. Each slot retains
all 144 original job status identities (1,728 total), each original queue's
decision and admission status, and all 27 resource axes: cap, attempted,
completed, cap minus attempted, and attempted minus completed. Actual queued
request identities are matched as a multiset against the original jobs with
assigned requests. Refused queue requests are generated before admission but
remain unassigned in the original jobs. Therefore refused requests overlap
`not_requested` job capacity; they must not be added to the 144 denominator.
The derived remainder is only the unused unassigned capacity after accounting
for refused queue requests, not an estimate of unobserved work or cost.

Final headroom is not the exact budget at a previous decision and does not
identify the blocking axis. Inclusive nested operation times are not additive.
Unknown FLOPs, CUDA timing and backward-engine work stay null. Shared cap is
not equal actual resource use, equal FLOPs, fair performance, demonstrated C3,
or a physical/scientific benefit. No ranking or outcome-based filtering.

The fixed real receipt verified for this implementation is
`results/agent_p3_c3_compact_host_cuda_20261029_v1_consume_MAIN01.json`, SHA
`cdd61fc20ff4b819e64092f226cbe84ce3777edfc2a2c8472322a01ec3a0601f`;
its report SHA is `88e7ac468dbcda7dddd07ba72e3c42793ff0a3d10f61e2e0c338ff1e85521dfe`.
The analyzer takes the receipt hash explicitly and never infers it from output.

From `/home/gbk/code/LeWorldModel/le-wm-main`, MAIN can run:

```sh
.venv/bin/python -m experiments.spectral_reachability.agent_p3_c3_budget_schedule_saved_20261029_v1 analyze --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_c3_compact_host_cuda_20261029_v1_consume_MAIN01.json --receipt-sha cdd61fc20ff4b819e64092f226cbe84ce3777edfc2a2c8472322a01ec3a0601f --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_c3_compact_host_cuda_20261029_v1_cuda_MAIN01 --analysis /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_c3_budget_schedule_saved_20261029_v1_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p3_c3_budget_schedule_saved_20261029_v1 verify --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_c3_compact_host_cuda_20261029_v1_consume_MAIN01.json --receipt-sha cdd61fc20ff4b819e64092f226cbe84ce3777edfc2a2c8472322a01ec3a0601f --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_c3_compact_host_cuda_20261029_v1_cuda_MAIN01 --analysis /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_c3_budget_schedule_saved_20261029_v1_MAIN01.json --analysis-sha ACTUAL_ANALYSIS_SHA
.venv/bin/python -m pytest -q tests/test_agent_p3_c3_budget_schedule_saved_20261029_v1.py
```

Output creation is exclusive and fsynced. Verify rebuilds the entire summary
from the saved transport; it is not an independent producer fullstrict audit.
