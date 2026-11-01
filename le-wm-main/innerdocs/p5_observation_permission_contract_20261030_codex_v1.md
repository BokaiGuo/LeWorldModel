# P5 同物理面板的观察权限合同

本合同为后续比较事先规定标注和禁止的解释；它建立时既有 fixed164 的结果已经可见，因此**不是那些结果的预注册**，不能成为独立确认或事前时间戳。使用 experiment-design 的控制变量和对照原则，不使用其不适用的训练、调参与多数据集模板。

## 权限与时序

| 既有臂 | 观察权限 | 其他信息和范围 |
|---|---|---|
| LeWM existing | RGB；当前绑定不独立证明实际模型每个历史时槽的输入 | 既有 LeWM task 是 goal-conditioned；其 task binding 从物理回执继承。不要因 RGB-only 隐去 goal 权限。 |
| JEPA native H1 | 历史 RGB 加 agent 位置 `(x,y)` 和速度 `(vx,vy)` | 历史 global 5/10/15；predictor 不接 goal。该臂明确不是 RGB-only。 |

动作是另列的控制输入，不算成观察权限；JEPA 3 个 token 对应各 5 条 raw 动作。successor global20 是评价标签，不得进入 predictor 或 selector。物体状态、未来 proprio 也不能作为 predictor 的额外信息。历史输入、目标权限、preprocessing 与动作合同必须独立比较：相同物理 PNG 来源不证明同一模型输入、同权限或同计算量。

固定全部 164 枝、80 对、4 个 baseline，包含两个已知根和四个 root×condition 层。每枝合同保存 bridge 三帧文件/像素 SHA、raw15/normalized action hash、raw/normalized 位置速度 hash、successor 身份及 LeWM task binding。它不保存本体数值或标签像素。复核仅对已绑定 JSON 的身份和完整顺序；原始 PNG 文件、normalizer、物理时序及模型加载边界继承上游绑定，不在本工具内重验。

## 允许的解释与未来对照条件

当前只允许“同一已保存物理 panel 的描述性并列”，必须披露权限和历史差异；不允许称 JEPA RGB-only、同权限公平性能、位置速度的因果效应、跨架构 raw latent MSE 排名、控制收益或 P5 科学通过。

未来同权限跨模型比较须在任何新结果前单独冻结：两臂可实际运行的权限（包括 goal）、原生 preprocessing、历史与 action 时间 ABI、共同物理身份及 task 输出指标、各自 encoder/predictor 的调用与 batch/frame 预算、固定分母和失败规则；逐次提供模型实际输入与加载证据。当前 JEPA native 必需的本体输入不能仅靠重命名消失，也不能以填零称 RGB-only。若选择同模型本体输入消融，应固定 checkpoint 和干预、如实标注零本体可能 OOD；该臂仍接收本体通道，不自动提供权限匹配或物理反事实证据。当前这些未来条件未完成，合同不授权生成实验。

## 实现与实际核验

只读取固定两个已存在 JSON：物理身份绑定 SHA `80ed10d3b8f623d361cc25e2764d390bc9cc758178c89e7467e99da3d785ef98`，JEPA bridge prepare SHA `1e042999e6970261882324c8adfc45ef6303decd047e2ac7f01f88942488586f`；不跟随其路径、不读 PNG、不访问保留根。核 schema、SHA、164/80 顺序、task、帧、标签及 arm 权限；未知、缺失、重标观察通道和 unsupported claim fail closed。JSON 中字符串和 hash 是合同声明与本地文件绑定，不是实际运行时权限拦截器。

实际输出 `results/p5_observation_permission_contract_20261030_codex_v1_MAIN01.json`，SHA `73c88ad0ff522c6bf21d729895c0d9985a5ec167c2ea6bc82061ee82e54702a8`。新测试 22 项；与上一绑定器联合 **37 passed**。verify 用同一实现重建一致，不是独立实现或 fullstrict。实际 `check` 对 descriptive 返回须披露权限、非同权限；对 `matched_observation_cross_model` 返回非零并拒绝。没有 GPU、NN、物理、训练或新根采集。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p5_observation_permission_contract_20261030_codex_v1.py tests/test_p5_crossmodel_panel_binding_20261030_codex_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p5_observation_permission_contract_20261030_codex_v1 verify --artifact results/p5_observation_permission_contract_20261030_codex_v1_MAIN01.json --sha256 73c88ad0ff522c6bf21d729895c0d9985a5ec167c2ea6bc82061ee82e54702a8
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p5_observation_permission_contract_20261030_codex_v1 check --artifact results/p5_observation_permission_contract_20261030_codex_v1_MAIN01.json --sha256 73c88ad0ff522c6bf21d729895c0d9985a5ec167c2ea6bc82061ee82e54702a8 --claim same_physical_panel_descriptive
```
