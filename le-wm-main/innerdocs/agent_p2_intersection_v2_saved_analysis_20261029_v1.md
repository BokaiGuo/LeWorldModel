# Intersection HOST v2 六计划 saved-only 分析

绑定实际 MAIN01 v2 prepare/report/full strict receipt 三份完整 SHA；不是此前六目标 controller。六槽严格保持 root0 joint/composition/grasp、root1 grasp/composition/joint。每槽从 receipt `plans[i].contexts[2].latent_history[2]` 和原 prepare `slots[root].packet.goal` 读取192坐标，binary64差、平方、`math.fsum`/192 独立计算终态模型 latent MSE。保留完整原plan、每槽27轴、side、wall、stop及完整account，root目标和终态附 SHA/pointer；不筛槽、不改配置、不填缺值。六计划缺失或partial均拒绝。

验证固定 prepare→report→receipt SHA与固定顺序、任务和shape，完整重建报告逐字段。认证继承 MAIN 完整 strict receipt；本分析不重跑 fullstrict，不独立认证原神经输出/梯度，非物理收益、非等 FLOPs、非 fresh 科学确认。FLOPs维持null。

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_intersection_v2_saved_analysis_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_intersection_v2_saved_analysis_20261029_v1 build --output results/agent_p2_intersection_v2_saved_analysis_20261029_v1_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_intersection_v2_saved_analysis_20261029_v1 verify --output results/agent_p2_intersection_v2_saved_analysis_20261029_v1_MAIN01.json --sha256 ACTUAL_OUTPUT_SHA
```
