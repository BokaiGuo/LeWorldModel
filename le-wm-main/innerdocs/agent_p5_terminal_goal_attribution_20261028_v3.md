# P5 terminal RGB goal attribution v3：authority 顺序修复

v3 是独占顺序修正版，不修改 v2、v1 或任何旧资产。唯一公开 authority 顺序固定为：

1. goal prepare、goal report、goal receipt；
2. physical manifest、physical report、physical receipt；
3. numeric manifest、numeric report、numeric receipt。

源码 `prepare(*authority)`、CLI `--authority`、本文示例和 RAM 测试均使用此顺序。v2 内部历史函数需要 numeric→physical 的参数顺序，v3 只在一个明确的 translation point 转换，避免用户按本文合法顺序触发 SHA 误报。v2 source SHA 被绑定，v2 不被修改。

其余 v2 严格合同全部保留：saved prepare 对 goal/numeric/physical 三套 consumer 返回值做精确比较；goal patch 和四个 prediction/terminal token 绑定 SHA/bytes；verify 完整重算 authority join；CUDA 强制 model_before==model_after 和 imported source closure；四词全分母、单已知 OOD root、feature MSE 与 native coverage 分离。新增成本仍仅四个 terminal RGB encoder forward，goal 不重编码、predictor 不重跑。

RAM 测试：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p5_terminal_goal_attribution_20261028_v3.py
```

MAIN prepare（顺序不可交换）：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_terminal_goal_attribution_20261028_v3 prepare \
 --manifest results/agent_p5_terminal_goal_attribution_20261028_v3_prepare_MAIN01.json \
 --authority \
 results/agent_p5_frozen_goal_scoring_20261028_v1_prepare_MAIN01.json \
 results/agent_p5_frozen_goal_scoring_20261028_v1_cuda_MAIN01/report.json \
 results/agent_p5_frozen_goal_scoring_20261028_v1_consume_MAIN01.json \
 results/agent_p5_contact_action_words_physical_20261028_v1_prepare_MAIN01.json \
 results/agent_p5_contact_action_words_physical_20261028_v1_physics_MAIN01/report.json \
 results/agent_p5_contact_action_words_physical_20261028_v1_consume_MAIN01.json \
 results/agent_p5_contact_action_response_20261028_v1_prepare_MAIN01.json \
 results/agent_p5_contact_action_response_20261028_v1_cuda_MAIN01/report.json \
 results/agent_p5_contact_action_response_20261028_v1_consume_MAIN01.json
sha256sum results/agent_p5_terminal_goal_attribution_20261028_v3_prepare_MAIN01.json
```

只有 MAIN 审计并填入实际 prepare SHA 后可授权 HOST CUDA；无 CPU fallback：

```bash
env -u PYTHONPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_terminal_goal_attribution_20261028_v3 cuda --manifest results/agent_p5_terminal_goal_attribution_20261028_v3_prepare_MAIN01.json --manifest-sha256 ACTUAL_PREPARE_SHA --confirm-host-cuda --output results/agent_p5_terminal_goal_attribution_20261028_v3_cuda_MAIN01
sha256sum results/agent_p5_terminal_goal_attribution_20261028_v3_cuda_MAIN01/report.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_terminal_goal_attribution_20261028_v3 consume --manifest results/agent_p5_terminal_goal_attribution_20261028_v3_prepare_MAIN01.json --manifest-sha256 ACTUAL_PREPARE_SHA --report results/agent_p5_terminal_goal_attribution_20261028_v3_cuda_MAIN01/report.json --report-sha256 ACTUAL_REPORT_SHA --output results/agent_p5_terminal_goal_attribution_20261028_v3_consume_MAIN01.json
```

worker 未运行 prepare/CUDA/神经/物理；测试不构成科学证据。
