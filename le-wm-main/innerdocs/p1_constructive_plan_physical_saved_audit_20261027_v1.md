# Independent constructive-plan physical saved audit

独占NEW source/test/doc。只读已存档证据与CPU数组；不导入任何producer/planner/consumer模块，不调用
branch/execute/metric/verify/analyze/consume，不执行神经、物理或重新render。指标、root identity、hashchain、PNG计账独立实现。

## 输入与范围

必须外部提供真实report SHA和实际physical prepare SHA，检查execution_binding；逐文件校验prepare bindings。
selection digest 使用默认空格JSON；逐指针绑定own rawcommands/source_slot/blind SHA/arm。handoff外部SHA绑定原planner report及task/bridge文件SHA，
但本审计**不重新证明planner候选选择数学或盲性**，也不拿hand-off consume冒充独立证明。
原source prepare envelope独立解压、semantic hash与sourcepack绑定；source journal独立hashchain；
证书NPZ仅读goal/goal_rgb/observed_states/root_history，不读取witness动作/参数/future_states。
blind只读合法三字段并核实真实zero raw history。原placement requested state与reset seeds按source已存recipe逐项比较，
不执行sample_state/模拟器；所以这是source recipe证据一致性，不是重新物理再现的证明。

## 检查

- journal索引/previous/hash逐条重建，run_started完整prepare，终态run_finished/report一致；nested操作LIFO，attempt与completed分别重计。
- 全root×arm分母，own预选动作、连续step序列，returned commands/state与PNG观测链，原生停止后无下一dispatch。
- 每个PNG真实file SHA、解码RGB uint8/224、pixel SHA；0/5/10三帧精确blind state/RGB/dynamics；goal附加前后root不变。
- RGB/state/snapshot先有fresh_observation，再guard_receipt；guard像素/画布/returned state独立重算，拒绝后无下一dispatch。
- goal与goalRGB SHA、root state+goal float64 identity；绑定区间无日志物理/setter/reset/render；source原reset recipe。
- 每returned state独立计算原native位置/角度/全7维距离/reward/terminal，并列wrapped/object metrics，不偷改native缺mod行为。
- 独立full census与预声明cap：每臂2reset/20public/22explicit/2goalrender/4setter/204physics/1goalbind；
  逐public dispatch必须10physics、逐reset必须2setter/2physics/1goalrender。总数不变但9/11错分步也拒绝。
- 精确artifact census和实际8MiB输出cap；不把未命名旁文件忽略成成功。

完整状态为`complete_saved_protocol_verified`；partial即使可解释也只能是`partial_saved_no_success_claim`；发现不一致为`saved_audit_failed`。
内部exception可能使journal receipt落后于真实执行；发现计账缺证据时不能“修补”或退款。

## 故障现场边界（必须保留）

`env.step`内部抛异常时，producer没有强保证在异常后再render。即使row.last_observation存在，也可能只是前一步/plan0。
审计对attempt后缺returned_transition的step输出：

```text
post_step_scene = unknown_after_incomplete_step
last_observation_role = pre_failure_last_observation_only
```

绝不将旧帧称为故障后现场。已经returned且有新fresh_observation+拒绝guard的情况才可标`saved_rejected_observation`。
render本身抛异常同样无新RGB保证；日志snapshot与PNG仅证明实际保存的观测，不推测未观测状态。
已returned但缺新观测时标`unknown_after_unobserved_transition`，旧last_observation仍只属故障前最后观测。
无独立pre-render snapshot存档，`render_preserved_dynamics`只能作为producer attestation；
hook receipts只能核验保存的调用证据，不能替代真实模拟器重放。

## MAIN最小命令

工作目录`/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_plan_physical_saved_audit_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_plan_physical_saved_audit_20261027_v1 --report /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_plan_physical_20261027_v1_main01/report.json --report-sha d205850e95f639b649fba9b5c6ee8059f2507431753a538b1ce30eccbd234300 --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_plan_physical_20261027_v1_prepare01.json --prepare-sha b717feb4bba220f40f93bb317d1e8894a19b672f54a5942c03f4308253245eb4
```

默认仅stdout，无结果写入。可选`--output /ABS/NEW_audit.json`，<=150KiB、200MiB floor、xb独占；禁止写入sealed physical目录。
测试只读取上述显式授权actual evidence，篡改/step异常均为RAM反例，无tmp文件。
结论严格限程序/协议/保存指标，不宣称P1通过、plateau或任务效能，不增加独立样本数。
输入SHA/损坏hashchain等底层认证失败直接拒绝并报错，不强行生成完整审计结论。

## 本轮实际独立只读结果

指定report核验通过，findings=[]；重计8reset/80public/88explicit/8goalrender/16setter/816physics/4goalbind，88PNG。
四臂均horizon_completed/10own steps，原native success均false。
以下均从保存state独立重算，非调用producer.metric，非新物理实验：

|root|arm|物体位移|object位置误差0→5→10|终态native全7维distance|
|---|---|---:|---|---:|
|0|GN|48.294484|94.668030→49.292891→47.976995|152.662136|
|0|D2P|30.704421|94.668030→63.653480→64.251279|181.747486|
|1|GN|24.027057|149.181105→149.181105→130.929152|492.696370|
|1|D2P|132.970057|149.181105→68.241341→45.398509|453.385088|

局部物体位置改善不等于原native成功，也不证明P1困难/plateau/机制准入。测试包含真实存档只读与RAM篡改/异常反例；无tmp产物。
本轮最小测试命令实际完成21 passed；包括step内部异常、returned后render异常均不得用旧帧冒充新现场。
