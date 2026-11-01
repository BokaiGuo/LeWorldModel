# P2 intersection v2: same-root recorded resource differences

This is an incremental saved-only paired-axis view, not a second plan/MSE panel. It pins and fully rebuilds the existing six-plan panel from its fixed prepare/report/MAIN strict receipt, then computes left-minus-right for all 27 original axes and side counters for the three fixed method pairs per root. Unknown propagates; FLOPs and equal FLOPs remain null. It retains six slots and six within-root pairs and never selects a winner. The input MAIN receipt remains the authority; this is not an independent neural, gradient, physical or full-strict verification. Elapsed differences are the recorded cooperative search scopes and not isolated hardware costs.

Existing panel: `results/agent_p2_intersection_v2_saved_analysis_20261029_v1_MAIN01.json`, SHA `5cc0c8e0bf370ba24a366555b2cbc7c29ef275f46206b4906f904cc64549a1b9`; its own source rebuilds the fixed report, prepare and receipt joins. Imported panel and budget-axis source files are SHA-pinned. No producer, neural model or physical execution is called.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_intersection_resource_deltas_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_intersection_resource_deltas_20261029_v1 build --output results/agent_p2_intersection_resource_deltas_20261029_v1_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_intersection_resource_deltas_20261029_v1 verify --output results/agent_p2_intersection_resource_deltas_20261029_v1_MAIN01.json --sha256 ACTUAL_OUTPUT_SHA
```

Shared per-coordinate caps are not equal actual usage, equal costs, or equal FLOPs. J/VJP, AD primal and scalar backward have different semantics; CPU capture includes its recorded scope. Replay is paid, not padding. A stronger comparison needs an a priori common resource measurement and stopping contract with known treatment of hidden AD and memory/capture work, all methods and complete attempts, source-verified original baseline and separately disclosed setup/measurement overhead, common task/action/criterion/time support, and independent fresh roots. Do not tune budgets based on these observed results or drop GRASP. P1/P2 scientific gates remain unchanged.
