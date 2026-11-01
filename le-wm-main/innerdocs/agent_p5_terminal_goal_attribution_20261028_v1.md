# P5 四词真实终态 RGB–goal attribution（MAIN HOST CUDA）

这是诊断模块，不是新的物理实验、预测实验或任务收益声明。只新增本 stem 三件套；不改 frozen physical、numeric predictor 或 goal scorer。它消费四词 physical full strict 的真实 frame20 RGB、完整 native double/public state、transition，以及已通过 strict 的 numeric prediction artifact 和已封存 goal patch。private pose/contact/suffix 不进入 planner，也不用于择词；它们仅随 physical strict envelope 做来源闭合。

## 固定来源与前置

物理固定四词顺序 `xy,yx,x_minus_x,zero_zero`，单一已知 OOD root，四词全部保留。固定 physical source/manifest/report/receipt SHA 分别为：

`0a00d0f7864f03fc8e9ddf1fa7cef749a1d57f6c977357ce78c309262aa4556f`、`46f8cf63d045123edfa61b257ad11bd94b3741b47dfe2244463c5c5096df0ab7`、`a07c2583c5d5dd553324d0e5c0b740ffbde3fa7a806798b3a8dcb0fe77ab6633`、`f6cd1f53d7537a6292c49647a3e20cfd61caf8a4b5e5e9d395223fc5845a39ee`。

goal scorer 已知 source/prepare/report/consume SHA 为 `d987d0146627ad6067c11665d0a2e86f27d9fbda86f400db0ccf3d2d9f970e9c`、`aecd86b684ab664254a695c4ca10f131a90cb8e8f77e651eae8864b4ed2b07ac`、`a732ba40426ff56d3d82db59c30759f7c2db7a54f970d33bc6dc6412c81407e5`、`22b8319b443d6dcf5d475e0149df9df3909bcb7106db57fbfb01e77778bd76d4`。numeric predictor source/manifest/report/consume SHA 是 `bd5b85b308fcb621adda1f4452911f58e233b9ba0812e8d569d0dbd540940930`、`572e2fded1c8927302c4ffe66921ce05e58f5954b43a9bc44bc022c19c6bbfb6`、`4fd032a4e5fb365f5a11eecbaba416c1ace7352e7afd092aa0adf6b972bded7f`、`247a8a6328053f34b8367f39dca63f81fb009ca2a41734db0f618d1ea6eab2c3`。

这里沿用已封存 goal feature，不再编码 goal；新 CUDA 仅编码四个真实 frame20 RGB。四个新调用均为 official encoder batch1、forward sample1、frame1；proprio/predictor/AD/backward/JVP/VJP/SVD/physics 新成本均0。旧 goal encoder1 与 numeric predictor8 是 saved/reused 轴，不与新增4相加，不称 equal total compute/FLOPs。

## 输出与归因口径

每词保存完整终态 `terminal_features.npy` `(256,384)` 及 SHA。逐词输出：

- `predicted_goal_mse`：封存 numeric predictor 的 frame10 suffix terminal visual token 与 sealed goal token 的 float64 MSE；不重跑 predictor。
- `real_terminal_goal_mse`：真实 physical frame20 RGB 经同一 encoder 与 sealed goal token 的 float64 MSE。
- `model_prediction_minus_real`：两者的诊断差，不声称误差分解或因果归因。
- native `state`（含 agent/block double）、`proprio`、coverage/reward/done、transition 原样并列；这些 native coverage 数值绝不用于调整 feature 分数、选词或理论失败解释。

真实 feature-goal MSE 不等于 native coverage：RGB包含 agent、block、goal raster 与渲染因素，且 feature space 与 coverage 单位不同。即便 zero 被 goal scorer 选中，也不能推出理论失败；本模块不输出 selected word，不改变任何任务门槛。

## 严格消费与失败边界

prepare 必须同时接收实际 goal scorer、numeric predictor、physical 三套 authority；逐文件 SHA、报告状态、四词顺序、终态 frame20 inventory、预测 token `(3,256,384)`、真实 RGB/state/transition closure 均 fail-closed。goal scorer receipt 只作为已保存 goal token 的权威，不重新运行 goal scorer。consume 重算四行指标、8个 encoder event 的链、完整 output inventory 与 meter；任何 partial/error/finalization error、model drift、篡改或额外文件拒绝。无补帧、无重试、无缺失填零。

所有 `scientific_admission/C2_established/task_success/physical_release` 恒 false/null。单 OOD root 的四词不是四个独立样本；这是“预测 token 与真实 RGB token 对 goal 的差异诊断”，不是 C2 任务收益、物体非交换能力或 native task success。

## MAIN 命令

RAM 测试（worker 已执行纯 RAM，不能替代 CUDA）：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p5_terminal_goal_attribution_20261028_v1.py
```

准备（所有路径实际存在且输出不存在）：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_terminal_goal_attribution_20261028_v1 prepare \
 --manifest results/agent_p5_terminal_goal_attribution_20261028_v1_prepare_MAIN01.json \
 --goal-prepare results/agent_p5_frozen_goal_scoring_20261028_v1_prepare_MAIN01.json \
 --goal-report results/agent_p5_frozen_goal_scoring_20261028_v1_cuda_MAIN01/report.json --goal-receipt results/agent_p5_frozen_goal_scoring_20261028_v1_consume_MAIN01.json \
 --physical-manifest results/agent_p5_contact_action_words_physical_20261028_v1_prepare_MAIN01.json --physical-report results/agent_p5_contact_action_words_physical_20261028_v1_physics_MAIN01/report.json --physical-receipt results/agent_p5_contact_action_words_physical_20261028_v1_consume_MAIN01.json \
 --prediction-manifest results/agent_p5_contact_action_response_20261028_v1_prepare_MAIN01.json --prediction-report results/agent_p5_contact_action_response_20261028_v1_cuda_MAIN01/report.json --prediction-receipt results/agent_p5_contact_action_response_20261028_v1_consume_MAIN01.json
sha256sum results/agent_p5_terminal_goal_attribution_20261028_v1_prepare_MAIN01.json
```

实际 SHA 由 MAIN 填入 verify。HOST CUDA 必须用 fresh process、`HF_HUB_OFFLINE=1`、`env -u PYTHONPATH`、`--confirm-host-cuda`，输出 `results/agent_p5_terminal_goal_attribution_20261028_v1_cuda_MAIN01`。之后实际 runner/consume：

```bash
env -u PYTHONPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_terminal_goal_attribution_20261028_v1 cuda --manifest results/agent_p5_terminal_goal_attribution_20261028_v1_prepare_MAIN01.json --manifest-sha256 ACTUAL_PREPARE_SHA --confirm-host-cuda --output results/agent_p5_terminal_goal_attribution_20261028_v1_cuda_MAIN01
sha256sum results/agent_p5_terminal_goal_attribution_20261028_v1_cuda_MAIN01/report.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_terminal_goal_attribution_20261028_v1 consume --manifest results/agent_p5_terminal_goal_attribution_20261028_v1_prepare_MAIN01.json --manifest-sha256 ACTUAL_PREPARE_SHA --report results/agent_p5_terminal_goal_attribution_20261028_v1_cuda_MAIN01/report.json --report-sha256 ACTUAL_REPORT_SHA --output results/agent_p5_terminal_goal_attribution_20261028_v1_consume_MAIN01.json
```
