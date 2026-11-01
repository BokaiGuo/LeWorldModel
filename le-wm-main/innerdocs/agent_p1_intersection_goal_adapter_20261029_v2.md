# Intersection physical v2 → P1 目标 spec：接线准备

此版本只接受 `agent_p2_intersection_physical_20261029_v2.prepare` 和相应 v2 contract，并要求固定分母 6、固定六槽顺序及 cp0/5/10/15。原 goal7 与每个 native_root 位级一致、两根原身份不变、三个 arm/根、`rgb_pairs=[]`；不会构造新根或声称同图不同速度。physical v1 的 `fixed_denominator=4` 必须拒绝，旧 v1 不作为正式输入。

v2 source/prepare 尚未由本任务取得实际冻结 SHA；CLI 要求 MAIN 显式提供二者实际 SHA，并验证 prepare 绑定的 v2 source 与已知 intersection CUDA receipt SHA。receipt 的 full-consume 重建声明继承自 physical prepare，本工具不重新跑上游 strict。无真实 v2 ABI 时仅 fixture 测试；若实际 v2 与假定字段不符必须失败并经 MAIN 审查修正，不自行补字段或猜 SHA。

本文件只准备接口；当前禁止生成 spec。待 MAIN 提供 v2 三件套冻结/prepare SHA、审查通过且确认本次物理结果尚未对制定者可见，方可执行下列命令，实际事前时序须外部留证。不能自行将事后 spec 称为前瞻；`rgb_pairs=[]` 是未验证，不是速度可观测性结论。结果不改变 P1/P2 gate。未来 physical receipt 的真实归一化接线仍未实现。

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_agent_p1_intersection_goal_adapter_20261029_v2.py
.venv/bin/python experiments/spectral_reachability/agent_p1_intersection_goal_adapter_20261029_v2.py project --prepared /ABS/agent_p2_intersection_physical_20261029_v2_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --source-sha ACTUAL_SOURCE_SHA --output /ABS/agent_p1_intersection_goal_adapter_20261029_v2_spec_MAIN01.json
.venv/bin/python experiments/spectral_reachability/agent_p1_intersection_goal_adapter_20261029_v2.py verify --prepared /ABS/agent_p2_intersection_physical_20261029_v2_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --source-sha ACTUAL_SOURCE_SHA --output /ABS/agent_p1_intersection_goal_adapter_20261029_v2_spec_MAIN01.json
```
