# 六槽 latent 预测与资源 saved panel

该新小切片复用已生成的 `agent_p2_six_target_prediction_comparison_20261029_v1_MAIN01.json`（SHA eb4e9acc5be07d4c9381d15a932acc14de2454d4db5aecb72f17a9b65fa447b7），并精确绑定实际六槽 report 和整份 strict receipt。上游比较已重建原 manifest、whole/segmented mean、源目标与原 goal，并保存24行及完整receipt；本件只核输入SHA、整份receipt与report资源交叉一致，再投影固定6槽×cp0/5/10/15、原目标/任务goal MSE、source时间、逐槽完整资源/输入/trace/数组描述符及全部27轴、side axes。它继承既有认证，不重复 fullstrict，不独立验证神经输出或梯度。计划位置通过实际 report trace、全部 arrays 和 input_files 原描述符绑定，不另造计划。

只描述模型预测，原归一化目标各不相同，tracking MSE不能跨目标直接排名；prior时间未知、generation预算不同、known两根不是独立确认。FLOPs为null，不宣称等FLOPs、物理收益或科学机制。禁止按效果筛槽；任何来源、顺序、缺项或资源账不符拒绝。

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_six_target_resource_panel_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_six_target_resource_panel_20261029_v1 build --output results/agent_p2_six_target_resource_panel_20261029_v1_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_six_target_resource_panel_20261029_v1 verify --output results/agent_p2_six_target_resource_panel_20261029_v1_MAIN01.json --sha256 ACTUAL_OUTPUT_SHA
```
