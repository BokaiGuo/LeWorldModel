# Fixed-prefix host integration (under review)

The fixed four slots, two parent arms and iterations 1–8 produce 64 groups,
each with nine candidates (576 denominator). Every group pays 19 predicts
and eight scalar reverses, added to the inherited parent 288/128. This is not
an equal-FLOPs claim, a new-root result, a physical result, or P1 admission.
The source `prepare` reconstructs the parent full strict and requires bytewise
equality with the actual persisted parent receipt. Each sealed group is audited
from its native transcript and independently checked scalar objectives. Partial
receipts keep all 64 statuses and paid resources; they are consumable as partial,
not automatically resumable, and cannot release a plan.

Engineering limits are 2 GiB output including 256 MiB final reserve, 200 MiB
disk floor, 7200 seconds through the pre-report wall sample, and 3/4 GiB GPU
allocated/reserved. The final report fsync is outside the wall sample. Limits are
hard protections, not a pre-run guarantee of fit. Source, installed class,
checkpoint state and precision are guarded. The runner stops on first failure;
no resource refund, fallback, training or physics. A persistence failure can
prevent a complete report despite the reserve and must not be called a receipt.

RAM/temporary-disk test: `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_agent_p1_fixed_prefix_host_cuda_20261029_v1.py tests/test_agent_p1_fixed_prefix_saved_audit_20261029_v1.py`.

Only MAIN may authorize production after review and full 64-group fixture
validation. From the project root, using actual absolute paths and SHA-256s:

```
.venv/bin/python -m experiments.spectral_reachability.agent_p1_fixed_prefix_host_cuda_20261029_v1 prepare --parent-output /ABS/PARENT_OUTPUT --parent-report-sha REPORT_SHA --parent-prepared /ABS/PARENT_PREPARE --parent-prepare-sha PREPARE_SHA --parent-receipt /ABS/PARENT_RECEIPT --parent-receipt-sha RECEIPT_SHA --output /ABS/results/agent_p1_fixed_prefix_host_cuda_20261029_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p1_fixed_prefix_host_cuda_20261029_v1 verify --prepared /ABS/results/agent_p1_fixed_prefix_host_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p1_fixed_prefix_host_cuda_20261029_v1 run --prepared /ABS/results/agent_p1_fixed_prefix_host_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output /ABS/results/agent_p1_fixed_prefix_host_cuda_20261029_v1_cuda_MAIN01 --authorization MAIN_P1_FIXED_PREFIX_HOST_CUDA_ONLY
.venv/bin/python -m experiments.spectral_reachability.agent_p1_fixed_prefix_host_cuda_20261029_v1 consume --output /ABS/results/agent_p1_fixed_prefix_host_cuda_20261029_v1_cuda_MAIN01 --report-sha ACTUAL_REPORT_SHA --receipt /ABS/results/agent_p1_fixed_prefix_host_cuda_20261029_v1_consume_MAIN01.json
```

Verify and consume both rebuild full parent strict. No full production strict
or neural execution was performed by the worker. If verification or output
creation fails before the durable binding is written there may be no consumable
report; the existence of a partial directory is not a receipt.
