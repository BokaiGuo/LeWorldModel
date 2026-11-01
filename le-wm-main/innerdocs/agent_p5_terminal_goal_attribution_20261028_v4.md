# P5 terminal RGB goal attribution v4

v3 的 public 顺序修复仍未把参数交换传给 v2 delegate。v4 是独占修正版；v2/v3/旧资产和 v3 失败输出均保留不覆盖。

唯一 public/CLI 顺序固定为：

`goal prepare, goal report, goal receipt, physical manifest, physical report, physical receipt, numeric manifest, numeric report, numeric receipt`

v2 内部 ABI 固定为 `goal, numeric, physical`。v4 `prepare` 的唯一 translation point 显式把第4–6与第7–9交换后调用 v2；源码、文档、测试一致。测试第一项直接使用工作区九个真实文件调用 `prepare`，逐文件实际 SHA 由 v2 authority join 检查，不使用假路径或 monkeypatch。

## MAIN saved prepare

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p5_terminal_goal_attribution_20261028_v4.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_terminal_goal_attribution_20261028_v4 prepare \
 --manifest results/agent_p5_terminal_goal_attribution_20261028_v4_prepare_MAIN01.json \
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
sha256sum results/agent_p5_terminal_goal_attribution_20261028_v4_prepare_MAIN01.json
```

输出必须 exclusive create；v3 失败文件不复用。只有 MAIN 审读实际 v4 prepare SHA 后才可沿用 v3/v2 的 `cuda --confirm-host-cuda` 与 `consume` 流程，且不允许 CPU fallback。v4 不重跑 goal/predictor/physics，仍只新增四终态 RGB encoder forward；不声明跨模型闭环或科学准入。
