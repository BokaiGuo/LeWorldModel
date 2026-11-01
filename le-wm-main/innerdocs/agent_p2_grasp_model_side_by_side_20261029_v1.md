# GRASP variant 与共同 cap 模型计划：saved-only 并栏

仅新增本 stem 源码、测试和本文；旧 producer、consumer、结果均不变。未发现已有六计划实际模型 Q 并栏；现有 common-resource GRASP spec 是前瞻协议，不能替代此表。两个 known/off 根全保留、非 fresh。不打开 blind NPZ、PNG、模型权重或物理报告，不读取或等待物理 outcome。

## 实际模型数值（非作者表值，非物理收益）

|known root|模型计划|原始模型 Q / 独立 float64 MSE|predict API / batch1 samples|scalar loss backward|Jacobian / 数学 VJP|搜索 wall 秒|
|---|---|---:|---:|---:|---:|---:|
|0|history-aware joint30-bounded GRASP variant|0.13271686313507314|2706 / 2706|900|0 / 0|unknown|
|0|共同 cap joint|0.3634363508809801|1902 / 1902|0|85 / 16320|67.83588980697095|
|0|共同 cap composition|0.32377574079538207|1062 / 1062|0|85 / 16320|46.08488543401472|
|1|history-aware joint30-bounded GRASP variant|0.04535450828709742|2706 / 2706|900|0 / 0|unknown|
|1|共同 cap composition|0.08947335474243788|1062 / 1062|0|85 / 16320|56.55687868897803|
|1|共同 cap joint|0.09379228960727053|1902 / 1902|0|85 / 16320|114.95711375400424|

Q 为原 identity192 下模型预测终态与原 goal 的均方距离（无物理单位，越小越低）；不是 native7、成功率或作者官方 GRASP 指标。该 GRASP 是 history-aware H3 算法适配，不是官方 laststate 实现或官方表值。无跨 cohort 提升汇总、不按结果筛计划。

## 身份与域

固定原 task_00/task_02 的 SHA、task_id、完整真实 root latent3/action2、goal/Q、模型 tensor hash/策略、原 normalizer、warm raw10；指针绑定 source blind SHA 但不打开。原导出 task.horizon=2 不变，实际规划未来3包，每包5动作、总15。全30 float64 logits 在同一 L2 半径 `.75*sqrt(15)` 球，NumPy tanh 后一次 float32，原 float32 normalization；不再除100、不修复越界计划。两组输入与动作域相同，不要求算法字段相同，不把算法不同当身份失败。

这是 identity/domain 一致的**异资源模型计划并栏，不是 matched 对照**。GRASP 保留750 joint+150 sync、噪声及原 loss；每根2706 predict/900 backward。joint/composition 同 cap1902，但实际1902/1062，85J 的尺寸分别489600/247680，AD步255/129；各16320数学输出方向、2 root SVD+1 ridge。J内部引擎 backward 次数和 FLOPs unknown/null，绝不与 scalar backward 混同。每次 AE/predictor/pred_proj 与 batch1 samples、3 context tokens/proj rows 单独列于 JSON；0新 encoder/projector/physics。

GRASP 实际90.6606秒为含验证/载入/两根/最终hash的整段 wall，上限3600；不捏造每根搜索时间。共同 cap 的每 arm 搜索上限450秒，整段上限2400；不同计时范围不直接算速度比。双方完整付费，不 refund、不 padding。P1/P2 admission、physical_release、matched、equal_budget、equal_FLOPs 均 false。

## 权威与可执行接口

源码 PINS 固定两份完整 strict 收据（GRASP v2 `55c192a9…`；common v1 `7b3eba57…`）、两 prepare 和两 report 的完整 SHA。保留 receipt 自身 schema，不伪造 producer authority。使用各 producer 自己的 digest 验 selection；继承已完整 strict 的神经数组/优化/计量认证，不重复运行旧 consume。另独立复算每份保存的192终态 Q、raw15 chart/bits、完整输入身份及全部已付费用字段；缺失拒绝、不填0。只读取显式绑定 task gzip、model-only JSON 和 source 文本；不扫目录、不验证或读取物理 outcome。

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```sh
.venv/bin/python -B -m pytest -q tests/test_agent_p2_grasp_model_side_by_side_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_grasp_model_side_by_side_20261029_v1
# MAIN如需独占持久结果（存在即拒绝，不覆盖）：
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_grasp_model_side_by_side_20261029_v1 --output results/agent_p2_grasp_model_side_by_side_20261029_v1_MAIN01.json
```

stdout 为完整6行分析；可选输出上限16MiB、floor200MiB、排他新路径。测试包括真实封存 JSON schema 回归、RAM coherent wrong-Q/raw 和身份/费用缺失篡改、冷导入阻断神经/物理依赖。不执行模型或重放物理。科学限制：仅2known根；没有物理收益结论，也不能由模型 Q 较低宣称 P2机制、GRASP官方复现或科学准入。下一步是否设计共同 cap 必须另行事前固定，本工具不授权运行。

## 实际验证记录

本测试与冻结 GRASP v2 consumer 测试联合 `55 passed in 4.38s`；本测试39项。实际 saved build 全6行成功、172项显式文件绑定，原始Q与独立Q逐项精确相等，0新NN/物理，未写结果收据、未重跑旧consume。首轮本测试38pass/1fail：IO防泄漏测试误把文件名含 physical 的已绑定源码当成物理结果；现仅禁止物理**结果**读入，保留原断言目标与失败记录。此为测试边界修正，不是科学失败或阈值变动。MAIN如需持久表，使用上述排他 CLI；不需要等待物理终态。
