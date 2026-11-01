# Sourcepack saved consumer v2

新增版本，v1源码/测试/doc冻结。无producer/native/model导入，只读saved；不执行prepare或采集。
复用v1全部四槽/12phase、PNG/state7/body bits、证书/blind全数组、历史/批内pre-witness排重、
原native reward/success和完整嵌套资源账核验；显式绑定新producer三文件SHA。
新增每个public_step→native_action_return→native_return/warmup_return的一一对应：
实际float32值/bits、原命令、slot/phase/region、逐步顺序、operation内实际返回时间均独立核验。
所有reset/set_state/goal_setter/goal_render/goal_bind/public_step/explicit_render均须有
operation内goal_pose entry/return，float64[3]bits和初始原pose完全一致；PNG观测和source return
再次保存的pose也须一致。缺/重复/错接/改值，即使重封journal也拒绝。
原生首次setup前goal_pose尚不存在：只允许initial及首reset entry为null；
其后所有实际trace必须严格为冻结native默认outline [256,256,pi/4]的float64 bits。

完整saved工程校验成功为saved_structure_and_native_telemetry_verified_NOT_fresh，strict_complete=true，
CLI exit0仅工程收据，不是freshness/科学通过。历史证书未知保持unknown；freshness/confirmed/P1/P4/
physical_release恒false。部分失败保四槽12phase与实际账，不补造动作或return，不提升为完整。
不声称可从有限trace证明physics内部连续renderer pose，也不重跑采样/物理。

RAM联合测试：
```bash
env OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_p4_fresh_source_20261028_v2_tests.py tests/agent_p4_fresh_source_saved_consume_20261028_v2_tests.py
```

MAIN审读后仅对已合法保存的产物使用（不授权开启预留槽；真实SHA由MAIN提供）：
```bash
.venv/bin/python -B -m experiments.spectral_reachability.agent_p4_fresh_source_saved_consume_20261028_v2 --output /ABS/EXISTING_OUTPUT --sourcepack-sha ACTUAL_SHA --prepared /ABS/EXISTING_PREPARE.json --prepare-sha ACTUAL_PREPARE_SHA
```
Producer fresh prepare入口：module agent_p4_fresh_source_20261028_v2 prepare --mode fresh_candidate
--output /ABS/NEW_PREPARE.json --history-cert /ABS/REAL_CERT --history-sha REAL_CERT_SHA；
verify/run沿用其--prepared/--prepare-sha，run另需--authorize-main-physics和新独占--output。
当前缺真实完整历史证书，未实际prepare、run或saved消费；不得用calibration_only打开四个保留根。

最终实际联合RAM回归：v1/v2 producer及consumer四套共145 passed in 46.27s；
v2两CLI --help退出0。早一轮144通过/1失败为错误消息缺goal_pose字段名，补齐消息后全回归通过，
未修改拒绝条件。v1六文件SHA逐项核验未变。测试不是native/freshness科学证据。
