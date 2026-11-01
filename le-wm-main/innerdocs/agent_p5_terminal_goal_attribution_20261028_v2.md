# P5 terminal RGB goal attribution v2

v2 是 v1 的独占修正版；不修改或覆盖 v1。它仍是已知单 OOD 根的 saved/CUDA 诊断，不是跨模型闭环、C2 任务收益或物理新实验。只使用四词 frame20 RGB、sealed native double/public state、sealed goal patch 和 sealed numeric predictor token；不重新编码 goal、不重跑 predictor、不读 private 后验来择词。

## v1 四项缺口的闭合

1. `goal_patch.npy` 的 path、SHA、bytes、shape、dtype 写入 v2 manifest bindings，`verify` 与 `consume` 均重验。
2. prepare 动态加载已冻结 goal scorer 与 numeric scorer，并调用各自 saved `consume`；同时调用 MAIN physical thin wrapper。每个返回值必须与实际 receipt JSON 精确相等，不能只看 status。
3. 三套 manifest/report/receipt、三套 producer source、physical wrapper、goal scorer v1、numeric scorer v1、v1 lineage、全部 prediction token 和四个 terminal RGB/state/transition 文件均加入 authority join；`verify` 重新执行完整 saved prepare 并要求 manifest 深相等。
4. CUDA finally 强制 `model_after == model_before`；consume 拒绝不相等或缺失。四个 terminal feature artifact 的 path/SHA/bytes/shape/dtype 写入 report，consume 逐项重验并重算四词指标。

## 固定指标与预算

四词固定顺序 `xy,yx,x_minus_x,zero_zero`，四分母全部保留。新增成本严格为 official encoder 4 API / 4 forward samples / 4 frames，batch=1；goal encoder 1 和 numeric predictor 8 只作为已保存输入，不重算、不重复退款。proprio/predictor/AD/backward/JVP/VJP/SVD/physics=0，FLOPs=null，walltime 独立，不相加。

逐词输出 sealed prediction terminal→goal 的 float64 MSE、真实 frame20 RGB encoder terminal→sealed goal 的 float64 MSE、二者差值，以及 native state/proprio/coverage/reward/done 分轴。feature-goal MSE 不等于 native coverage；不按 coverage 调分数，不输出 winner，不把任何差异解释为理论失败。

## MAIN 命令

先 RAM：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p5_terminal_goal_attribution_20261028_v2.py
```

prepare 的 `--authority` 九个路径依次为：goal prepare、goal report、goal receipt、physical manifest、physical report、physical receipt、numeric manifest、numeric report、numeric receipt。示例：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_terminal_goal_attribution_20261028_v2 prepare \
 --manifest results/agent_p5_terminal_goal_attribution_20261028_v2_prepare_MAIN01.json \
 --authority results/agent_p5_frozen_goal_scoring_20261028_v1_prepare_MAIN01.json results/agent_p5_frozen_goal_scoring_20261028_v1_cuda_MAIN01/report.json results/agent_p5_frozen_goal_scoring_20261028_v1_consume_MAIN01.json \
 results/agent_p5_contact_action_words_physical_20261028_v1_prepare_MAIN01.json results/agent_p5_contact_action_words_physical_20261028_v1_physics_MAIN01/report.json results/agent_p5_contact_action_words_physical_20261028_v1_consume_MAIN01.json \
 results/agent_p5_contact_action_response_20261028_v1_prepare_MAIN01.json results/agent_p5_contact_action_response_20261028_v1_cuda_MAIN01/report.json results/agent_p5_contact_action_response_20261028_v1_consume_MAIN01.json
sha256sum results/agent_p5_terminal_goal_attribution_20261028_v2_prepare_MAIN01.json
```

只有 MAIN 审计 source/manifest 后允许 HOST CUDA；无 CPU fallback：

```bash
env -u PYTHONPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_terminal_goal_attribution_20261028_v2 cuda --manifest results/agent_p5_terminal_goal_attribution_20261028_v2_prepare_MAIN01.json --manifest-sha256 ACTUAL_PREPARE_SHA --confirm-host-cuda --output results/agent_p5_terminal_goal_attribution_20261028_v2_cuda_MAIN01
sha256sum results/agent_p5_terminal_goal_attribution_20261028_v2_cuda_MAIN01/report.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_terminal_goal_attribution_20261028_v2 consume --manifest results/agent_p5_terminal_goal_attribution_20261028_v2_prepare_MAIN01.json --manifest-sha256 ACTUAL_PREPARE_SHA --report results/agent_p5_terminal_goal_attribution_20261028_v2_cuda_MAIN01/report.json --report-sha256 ACTUAL_REPORT_SHA --output results/agent_p5_terminal_goal_attribution_20261028_v2_consume_MAIN01.json
```

RAM 测试和 saved prepare 不是神经科学证据；worker 不运行 prepare、CUDA、物理或模型。任何 authority 缺失、receipt/report 不精确相等、goal/token bytes 漂移、模型前后不等或额外输出文件均 fail-closed。
