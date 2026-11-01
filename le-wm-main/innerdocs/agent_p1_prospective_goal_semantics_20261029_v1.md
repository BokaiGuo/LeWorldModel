# P1 前瞻目标语义合同（不执行）

目的：在事先固定的全部轨迹和检查点上，同时报告原 native success、原 native7 距离及其位置、原始角度、agent 目标速度误差分量；另检查事先列出的实际静态目标图对是否存在同图不同速度。原 goal7 原样保留、原 gate 原样保留，不按结果选根、轨迹、检查点或配对。没有环境、训练、物理或模型调用。

## 事前准备与边界

独立固定 spec（JSON）：`schema` 为 `agent_p1_prospective_goal_semantics_20261029_v1.spec`；`authority` 为包含外部实际来源文件的绝对 `path` 和 `sha256`；`slots` 是有序非空列表，每项仅 `slot_id`、`root_id`、原 `goal7`、`goal_authority_sha256`；`checkpoints` 是递增不重复非负整数；`rgb_pairs` 是固定 slot ID 对列表，可为空，不允许重复。源任务或结果出现后再修改 spec 就不是前瞻合同。相同 root 必须明确共用 root_id，不能当新独立根。所有参数和外部授权内容需要 MAIN 审核；程序对来源文件只做字节 SHA 校验，不能从任意自签文件推断真实实验权威。prepare 一律 `ready_for_execution=false`，没有下发任何动作或部署授权。

## saved receipt 接口

事后 receipt 的 schema 为本 stem 加 `.receipt`，字段仅 `schema,authority,rows`；authority 为外部已认证文件的绝对路径及 SHA。`rows` 必须与 spec 的每槽同序、全分母；每项为 `slot_id,root_id,goal7,goal_authority_sha256,goal_rgb,points`。goal7 的 float64 bits 必须与 spec 一致，包括负零。`goal_rgb` 可为 null，或 `{path,sha256,encoding:"RGB224_raw_u8"}` 指向已认证的 224×224×3 原像素字节，不以 PNG 文件字节相等替代像素相等。adapter 必须由 MAIN 另行确认图像来源、解码、renderer 情况和 receipt 对真实轨迹的绑定；本模块不重新认证上游 full strict。

`points` 对应全部预定 checkpoints，每项仅 `checkpoint,state7,native_success,native_distance,missing_reason`。真实缺失/早停时 state/success/distance 都为 null、reason 非空；绝不补点。非缺失时分别保存原返回的 success 和 distance；独立复算 success 并核对，距离另报告独立 double 算术与原返回差，不用容差改写原结果。分量为 goal−state：agent XY²、object XY²、raw angle²、agent velocity²、native7²。原 native success 使用原实现的角度 `min(abs(delta),2pi-abs(delta))`，不修正原公式；速度误差是对原非零 goal velocity，非速度幅值。

RGB 对在两图都可用时记录真实逐字节相等、goal 前五维 float64 bits 相同、后两维 velocity 值及 bits 是否不同；像素相同、前五维 bits 相同且速度数值不同才称**该对已观测像素碰撞**，正负零单独不同不算数值不同。null 图片得到 unknown。仅此特定配对说明这两个已绑定目标在提供的 RGB 表示上无法区分；不代表所有 renderer 或完整 observation 都无速度信息，尤其原 `goal_proprio` 可含速度。不得制造/变更目标来得到配对，不得以后验寻找匹配对改变分母；没有实际满足条件的既定对则不能宣称已验证碰撞。

## CLI

所有路径使用绝对路径，SHA 为实际文件 SHA256；输出独占 stem 且必须不存在。spec 必须在相关轨迹结果可见前完成并封存；receipt 仅来自后续 MAIN 已认证执行。无可用前瞻输入时不生成 ready 或真实结论。

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_agent_p1_prospective_goal_semantics_20261029_v1.py
.venv/bin/python experiments/spectral_reachability/agent_p1_prospective_goal_semantics_20261029_v1.py prepare --spec /ABS/SPEC.json --spec-sha SPEC_SHA --output /ABS/agent_p1_prospective_goal_semantics_20261029_v1_prepare.json
.venv/bin/python experiments/spectral_reachability/agent_p1_prospective_goal_semantics_20261029_v1.py analyze --spec /ABS/SPEC.json --spec-sha SPEC_SHA --receipt /ABS/RECEIPT.json --receipt-sha RECEIPT_SHA --output /ABS/agent_p1_prospective_goal_semantics_20261029_v1_analysis.json
.venv/bin/python experiments/spectral_reachability/agent_p1_prospective_goal_semantics_20261029_v1.py verify --spec /ABS/SPEC.json --spec-sha SPEC_SHA --receipt /ABS/RECEIPT.json --receipt-sha RECEIPT_SHA --output /ABS/agent_p1_prospective_goal_semantics_20261029_v1_analysis.json
```

两个 known root 不变为 fresh；本合同不证 P1 或因果机制。后续真实接入仍需要 MAIN 逐项授权/审查输入来源与预算。
