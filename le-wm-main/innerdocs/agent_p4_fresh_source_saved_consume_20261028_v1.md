# 四槽 sourcepack saved strict consumer v1

独占新增本stem source/tests/innerdoc。旧source adapter三文件完全冻结；不得执行其
calibration_only到四个预留fresh槽，MAIN明确不dispatch，本worker也未prepare/采集。
本文及checker不产生根、证书、外部批准、MAIN结果或假future SHA。

## 实际接线与核验

输入为MAIN显式提供的output、sourcepack byte SHA、prepare路径及byte SHA。
只读stdout；无producer/环境/model导入，无run/prepare入口、网络或神经/物理调用。
绑定冻结producer三SHA、prepare完整依赖字节、execution_binding、sourcepack、完整journal
链及terminal rows/counts/pending；path/SHA绑定之外还实际进行：

- 固定4槽×construction/replay0/replay1共12 phase，失败/未开始均不删行。
- operation attempted→completed逐ID、实际嵌套entry/return及分phase计账：reset下
  set_state→goal_setter→goal_render→set_state；每setter实际1physics，每public实际10physics；
  goal_bind只允许1goal_setter。独立核全局及每完整phase资源，拒绝把计量移到无关scope。
- 所有fresh PNG字节、uint8 RGB224像素、实际canvas/color guard；state7及完整两body
  position/velocity/unwrapped-angle/angular-velocity/force/torque/sleeping/原float64 bits，
  重算native `_get_obs`对应位置/速度/wrapped-angle，非仅hash检查。
- 全warmup raw10值与float32 bits、warmup commit、原native stop flags；从实际root重算
  原radial/tangent witness logits及float32 tanh，无调幅/阈值或按outcome选根。
- 每个实际warmup10的state/pose/snapshot/RGB3身份，独立重算历史/批内重复判定，
  包含先前失败槽已经观测到的root；核实际gate事件先于任何witness public entry。
  duplicate槽不得dispatch witness或发布blind，保留4分母；replay必须同槽精确root/history。
- 全witness native7 reward/success/早停与实际state，包含agent+object位置；不把object-only
  投影当完整任务失败，不修复原native angle公式。两replay逐步state/snapshot/RGB对应construction。
- construction certificate的全部8数组逐值逐bit重算比对；blind严格只含原三个部署数组，
  RGB3必须来自replay0，goalRGB必须来自construction终态；root-goal和实际replay步数独立核对。

这里“4”是槽数，每槽原合同只有3 phase，不凭描述增加第4个phase。
输入非终态/不完整journal/错SHA抛错；有终态工程失败则保留所有已保存观测和ledger，
返回partial_preserved_NOT_complete，不从好子集推完整通过。

## 冻结 producer 的不可凭saved补足项

旧 `p1_constructive_moving_history_source_20261027_v1.episode` 的 `warmup_return`保存实际raw bits，
但`native_return`只有state/reward/terminated/truncated，**没有逐witness步latest_action值/bit**；
新adapter复用的native operation计量事件也没有动作payload。commit与certificate保存的是
预先命令，冻结源码有逐步actual==command断言，但这些不是独立保存的逐步actual动作见证。
因此此consumer可严格核完整commit/数组/状态/计量，不伪称独立测得witness actual raw。

同理source snapshot只导出state及两body，不含逐phase/绑定前后的goal_pose原值；
旧adapter实际断言goal_pose不变，但纯saved不能独立重测该renderer字段。
placement_recipe保存original_goal/requested state，但不含native polygon顶点/solver cache；
本工具校验其身份、reset次序、真实后续state及replay一致性，不声称重跑sample_state/physics。
warmup返回也未保存reward，此处独立核其native终止标志，不伪造warmup reward收据。

故即便其余结构全部通过，结果仍为：

```text
status=saved_structure_verified_STRICT_BLOCKED_witness_raw_telemetry
actual_witness_raw_per_step=null
independent_goal_pose_trace=null
strict_complete=false
freshness=false, confirmed=false, P1_admission=false, P4_admission=false, physical_release=false
CLI exit=2
```

exit2是已知缺证不放行，不是CPU fallback/重采集请求。历史cert缺失为unknown；若prepare携带
MAIN cert仅SHA绑定该输入，不替MAIN审批、扩大历史域或写approved=true。
当前无真实新sourcepack，因此本轮没有“真实saved消费通过”或真实outcome结论。
后续若需补逐步动作/goal_pose telemetry，只能另开producer版本，并先由MAIN批准是否采集；
绝不为了测试或补证开启这四个保留根。

## 测试与MAIN调用

测试仅把实际producer orchestration接到RAM fake API，PNG编码/NumPy算术和元数据读取；
没有真实PushT、根采样或模型执行。变异会重封journal/sourcepack摘要后再检，覆盖state/body bits/
PNG/raw bits/witness/identity/native reward/blind/full-denominator/ledger，另拒绝physics挪出public的
相同总数假账。RAM sourcepack不是实际source证据。

最终联合RAM回归：consumer 23项与冻结producer 41项，共64 passed；CLI `--help`
退出0。未执行真实prepare、sourcepack采集或真实saved消费；四个预留槽仍未打开。

```bash
env OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_p4_fresh_source_saved_consume_20261028_v1_tests.py
```

只有MAIN将来实际提供合法已保存文件时，才用以下只读接口（这里没有伪造SHA）：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.agent_p4_fresh_source_saved_consume_20261028_v1 \
  --output /ABS/MAIN_EXISTING_SOURCE_OUTPUT \
  --sourcepack-sha ACTUAL_SOURCEPACK_SHA \
  --prepared /ABS/MAIN_EXISTING_PREPARE.json \
  --prepare-sha ACTUAL_PREPARE_SHA
```

以上不是授权采集命令。继承当前producer时预期stdout结构核验结果并exit2，不能改出口为0来冒充strict绿灯。
旧producer三SHA仍为source9a94776b…e01335 / tests4487f5ac…5c638 / docd0bff239…ec47f。
