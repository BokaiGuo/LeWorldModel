# Moving live feedback saved analysis v2：严格native state7修复

仅本v2 source/tests/innerdoc三个新文件；v1三文件、driver与实际结果全部保持封存。
暂停teacher-only文档任务，无新模型、CUDA、CPU神经或物理；MAIN负责实际重跑。
experiment-code用于最小修复与RAM回归，不新增拟合/阈值/选择。

## v1工程错误及精确冻结来源

MAIN实际运行v1触发 `ValueError: finite exact array shape`：v1 native_metrics误要求
state/goal均5维，且RAM fixture同样只有5维，因此27 tests未覆盖真实7维接口。
该CLI失败没有生成分析结果，属于接口工程错误，不是反馈实验或理论负结果。
不能用截去末两维的方式兼容，否则会暗改native distance。

已逐字只读确认：

- `p1_conditional_procedural_source_20261027_v1.py:90` 的source.vector严格shape=(7,)，返回float64。
- `p1_constructive_plan_physical_20261027_v1.py:196` 的kernel.metric使用两个完整source.vector，
  `d=state-goal`，native_distance=`np.linalg.norm(d)`（全7维）。
- 冻结native `.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py:347`
  eval_state同样对`goal_state-cur_state`全向量求norm；其SHA为
  `c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2`。
- 同native源码_get_obs第360行构造agent_xy、object_xy、object_angle、agent_velocity_xy，共7维。

v2只将native_metrics state/goal双方校验改为严格(7,)；5/5、5/7、7/5全部拒绝。
distance保留velocity贡献；success仍仅`norm(d[:4])<20`与
`min(abs(d[4]),2*pi-abs(d[4]))<pi/9`。保留原angle无mod缺陷和独立wrapped诊断。
object_pose仍取state[2:5]；不改变原native阈值、object_position_error或latent Q度量。
terminal checkpoint与original final_state完整7维列表比较，不截5，不丢velocity。

## 不变的输入、配对和资源合同

严格绑定driver source SHA
`6742a1b35102e433d7772e4a208b4f97b37b76f815b991cfc0e472763754ace9`。
入口仍为 `build(report_path, report_sha, prepared, prepare_sha)`，无producer import/execute。
同一actual report：
`results/p1_moving_live_feedback_physical_20261027_v1_main01/report.json`，外部SHA
`990843e8bba15955db5163b9b5d77c1f8eecaa1870d3745b49d46cc22ca895e3`。
actual prepare外部SHA
`bc3fbc91d7a19ae256b24db22ad05b2526fcbe454e3b55d904136eba2f8fdff5`；
具体绝对路径由MAIN提供并与execution_binding.json闭合，不猜文件名。

固定12 episodes＝2root×GN/D2P×O/I/R，缺项null仍保全部分母；
same-prefix配对不成立则对应contrast为null。Q5/10从各自保存float32 native192 NPY与
原task goal按float64 mean平方差复算，不改原图/latent或科学阈值。
保原final_metrics/native flags/错误；native前缀指标标为saved-state复算而非伪造native receipt。
按journal episode/region汇总实耗suffix，I-O、R-I、R-O仅描述，不做显著性或独立样本推断。
API/module计数不相加，原成本不退款。完整输入封存、路径/NPY/goal/operations校验与v1相同，
不重新hash大模型闭包，不冒称完整runtime审计。输出新v2 schema、独占JSON≤1MiB、200MiB floor。
P1/P4_admission=false，independent_samples=false，significance=null，new_neural/new_physics=0。

## RAM回归与MAIN命令

v1的27项测试全迁移到独立v2测试；主fixture全部换为真实7维语义，新增：
5/5和5/7/7/5拒绝、末两维对完整norm的贡献、success不依赖velocity、
完整7维endpoint/final_state与原saved metric比较、错误截5 final_state拒绝。
例如state=[3,4,0,0,0,6,8]对零goal，native_distance为sqrt(125)，而非5；
state=[0,0,0,0,0,3,4]对零goal，distance=5，但native_success仍true。
这只覆盖数组合同，不作为实际分析已成功的证据。

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_moving_live_feedback_analysis_20261027_v2.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_live_feedback_analysis_20261027_v2 \
  --report /home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_live_feedback_physical_20261027_v1_main01/report.json \
  --report-sha 990843e8bba15955db5163b9b5d77c1f8eecaa1870d3745b49d46cc22ca895e3 \
  --prepared /ABS/ACTUAL_DRIVER_PREPARE \
  --prepare-sha bc3fbc91d7a19ae256b24db22ad05b2526fcbe454e3b55d904136eba2f8fdff5 \
  --output /ABS/NEW_analysis_v2_main01.json
```

本交付仅RAM测试，不执行actual saved analysis，MAIN实际重跑。
不改v1的failed CLI痕迹，不把Q改善或0/12 native success转换成P1通过。
goal API仍由已取消的旧blocked任务占用，不假标旧goal完成。
