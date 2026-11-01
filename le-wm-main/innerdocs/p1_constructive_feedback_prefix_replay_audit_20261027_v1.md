# Independent feedback-prefix replay audit

仅NEW同stem source/tests/doc；不修改producer/旧saved auditor/封存报告。独立实现，无producer导入/调用。
只读旧/新physical report外部SHA、execution binding、各自prepare/selection、journal及相关PNG。
**不打开**neural report、latent export、checkpoint、sourcepack或certificate。不是新物理或模型执行。

## Fail-closed比较合同

两份report必须实际terminal且physical_calibration_recorded/errors[]，journal hashchain与run_started prepare、
run_finished/report精确闭合；自己selection digest用默认空格JSON重算。任何输入缺失/部分/重复/损坏或语义不一致均抛错，CLI非零退出。
只对固定2roots×2arms（原名字、原顺序）配对，不按结果/分数排序，不找最佳子集。

- sourcepack SHA、source prepare SHA/semantic ID、native evaluator root mapping与recipe路径一致。
- 每臂root_state+构造goal独立float64字节SHA与manifest rootsha一致；goal/goalRGB SHA一致；reset seed/placement requested一致。
- 每臂plan0..5必须恰有一条fresh_observation与接受guard；每步actual state/returned state/snapshot state一致。
- own step1..5的planned、operation_attempted、returned command逐字精确float32一致；dispatch/completed/returned/frame有序且无native terminal/truncation。
- 两次运行各步0..5 state7与完整block snapshot按float64位模式比较，包含signed zero，非allclose；commands按float32位模式。
- 全部24对prefix PNG比较**完整文件bytes**，不是仅hash或pixel相等；同时验证各文件SHA、pixel SHA、RGB224/guard。
- 另12对warmup0/5/10 PNG/state/snapshot一致；plan0与warmup10 byte/snapshot一致以核无target附加根。

tail允许不同，脚本不评价新tail效能，也不重审模型选优或完整tail计账。应先对新output运行既有独立saved physical auditor，
再做本前缀审计。这是保存的source/recipe/重现证据核验，不是重新sample/reset/simulator replay证明。
报告中的`distinct_report_inputs=false`仅表示同报告自比控制，不能当第二次实际执行重现证据。
每个实际step的执行仅由保存dispatch/returned/frame receipts证明，不声称离线软件能观察未保存的物理过程。

## MAIN命令（CPU saved only）

工作目录`/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_feedback_prefix_replay_audit_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_feedback_prefix_replay_audit_20261027_v1 --old-report results/p1_constructive_plan_physical_20261027_v1_main01/report.json --old-sha d205850e95f639b649fba9b5c6ee8059f2507431753a538b1ce30eccbd234300 --new-report results/p1_constructive_observed_suffix_physical_20261027_v1_main01/report.json --new-sha f864cc80a88eb289b578d271679d0bc895043a3b7bc798ae141097797a555b5e
```

默认stdout，无落盘。可选`--output /ABS/NEW_prefix_audit.json`，<=64KiB、200MiB floor、xb独占，禁止写入两份sealed physical输出目录。
测试RAM反例无tmp文件：缺帧、state最低位/signed zero、同像素不同PNG编码、commands、串臂、源/goal/reset绑定、partial、hashchain等。

## 已完成实际只读核验

独立计算新report SHA：`f864cc80a88eb289b578d271679d0bc895043a3b7bc798ae141097797a555b5e`。
MAIN后续外部确认前，此SHA来源明确是本审计独立读取计算，不能冒称MAIN提供。
新execution binding prepare SHA：`3c1fcd64f87801533b93cf8d7f2a1477240e7065371b063239b73d4ef85f7bce`。

先运行原封存独立saved physical auditor对新output：complete_saved_protocol_verified/findings=[]，
8reset/80public/88explicit/8goalrender/16setter/816physics/4goalbind，四臂完整10own steps，native success全false。
之后本prefix auditor实际完成exact_saved_prefix_replay_verified：4臂、24对prefix PNG与12对warmup PNG文件字节精确相同，
全部0..5 state/snapshot和前5commands一致。没有将旧报告自比当实际重现。

以上只认证保存的prefix精确重现，不意味着suffix效果提升、P1通过、plateau或独立样本增加。
本轮最小测试命令实际完成23 passed，全部saved读取/内存反例；未生成tmp、prepare或结果文件。
