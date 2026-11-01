# P5 latest-slot：全164/80 saved-only独立数值汇总

## 合同

只消费Jason实际`agent_p5_latest_slot_factorial_cuda_20261029_v1` schema，
固定164 branches、80 minus/plus对、4 baselines（0/41/82/123），不剔除零响应。
AA为原H1输出；AP/PA/PP为本轮492次新predictor输出，PP绝非原H2输出。
仅global frame15的visual/proprio交叉；frame5/10的原AA输入逐位锁定由上游完整strict验证。
所有cell共同使用原H1 actual20 patch label；H2 actual20 label只计算numeric seam。
保留selected15原编码/rollover数值历史，不将其称为纯物理反事实。

本模块只做saved数组算术，不加载torch、不重新编码、不运行NN/物理/训练。
不是P1/P5准入、fresh验证、跨模型性能排名或控制收益；两known roots不变成164独立样本。
使用data-analysis的完整分母、缺项与分组一致性约束；这里只描述固定有限数据，不做显著性检验/CI。

## 权威与独立性

必填外部output/report SHA、prepare路径/SHA、strict路径/SHA。
固定实际runtime source SHA：
`f5436a9ae922e717920c7662087f8380960c2c6a343503e2b8a7bc750b090ead`。
拒绝whole-history runtime schema；核显式prepare与实际execution_binding一致。

`analyze`真实调用原runtime `consume(output, report_sha)`完整重建source/artifact/journal/
native input/492call/模型未变/预测先封存后解锁label闭包，再要求外部receipt规范JSON完全一致。
必须是完整164/80、492调用和`actual_native_AA_prefix_bitwise_verified=true`，不能status-only。
自身三文件在strict重建前封存；全部prepare bindings、原artifact和实际读取数组在结束时重hash。
上游真实H1 report/strict的固定SHA由runtime闭包继承，结果authority中明确保留。

本模块独立写四cell MSE、factorial及response计算，不调用runtime的
features/row_score/factorial/score_all来得到这些数值。只复用policy-free SHA/JSON工具和完整strict入口。
每个原始NPY独立验证路径/文件SHA/字节/float32/shape/finite/EOF；score规范JSON精确相等，无容差。

## 输出与公式

完整保留164 branch rows（index/root/condition/task/selection identity）：

- 每cell仅取完整native full3输出的末slot；float64 C-order mean square对同H1 label。
- visual=`((PA-AA)+(PP-AP))/2`；proprio=`((AP-AA)+(PP-PA))/2`；interaction=`PP-PA-AP+AA`。
- 分开scalar MSE effects与feature effects L2，二者不可混称。
- H1/H2标签bytes equality/maxabs另列，不换label使结果好看。

完整80 pair rows按原minus/plus/root/condition/h/coord身份：

- 实际label response `label_plus-label_minus`、每cell predicted response及L2。
- cosine仅在两范数都非零时定义，否则null；不造epsilon，不把null当0。
- 每cell plus-minus MSE、response三效应L2；它们是response而非除以h的导数。
- 每对独立重算并逐字段核原runtime pairs，不按结果挑子集。

输出全部/root/condition/root×condition九张描述表：四cell MSE、三效应、response、
cosine正/零/负/null、全部12个有向cell转换×16符号bins。
空集合统计为null并保留available/null计数，不补结果。
另给observed/predicted exact-zero pair-index集合及真正集合相等布尔；绝不因数量同为24声称集合相同。
四baseline仍纳入branch表，但不伪造其对应pair。
实际本轮counts与inherited H1/H2 costs分栏，不称等FLOPs。

本版`whole_history_comparison=null`：不读取旧whole-history summary，不混用其PP或receipt。
将来若比较必须新版本显式绑定另一实际source/report/strict权威、重建并精确连接80身份，不能仅复用摘要数量。

## MAIN运行命令

在`/home/gbk/code/LeWorldModel/le-wm-main`执行。report/strict SHA为MAIN提供的实际权威；
prepare SHA仍由MAIN填其真实值。本worker没有生产读取/消费。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p5_latest_slot_factorial_summary_20261029_v1.py

.venv/bin/python -m experiments.spectral_reachability.agent_p5_latest_slot_factorial_summary_20261029_v1 analyze \
  --factorial-output results/agent_p5_latest_slot_factorial_cuda_20261029_v1_cuda_MAIN01 \
  --report-sha a9479f81d87b1b70ebd1f64db0289d9d2c6a5ae57932cd021f85cc5ecfd47316 \
  --prepared results/agent_p5_latest_slot_factorial_cuda_20261029_v1_prepare_MAIN01.json \
  --prepare-sha "$P5_LATEST_PREPARE_SHA" \
  --strict-receipt results/agent_p5_latest_slot_factorial_cuda_20261029_v1_consume_MAIN01.json \
  --strict-sha c636f8d9c6624317d4da4f3ddb127b4f20710f9a5c02c212872d0d70c9cab642 \
  --output results/agent_p5_latest_slot_factorial_summary_20261029_v1_MAIN01.json

.venv/bin/python -m experiments.spectral_reachability.agent_p5_latest_slot_factorial_summary_20261029_v1 verify \
  --analysis results/agent_p5_latest_slot_factorial_summary_20261029_v1_MAIN01.json \
  --analysis-sha "$P5_LATEST_SUMMARY_SHA"
```

verify完整重建strict与数值结果，不是只检查summary SHA。
保存仅允许本stem的results新文件：32MiB output、200MiB floor、不覆盖；文件flush/fsync后目录fsync。
900秒覆盖strict、算术和结束rehash（不可中断的上游consume返回后再检查wall）。
仅输出compact路径/SHA/status，不向stdout打印全164数组。

## 测试与未完成

非神经fixture覆盖全部分母/顺序/每对身份、零集合等数不同成员、null与极小非零符号、
四cell公式、真实完整尺寸NPY的terminal-slot语义、NEW PP而非H2 provenance、
文件SHA/bytes/shape/dtype/EOF/nonfinite篡改、score篡改、external SHA/prepare/source漂移、
status-only/错误receipt/whole-history schema拒绝，以及analyze→verify确定性接线。
测试的strict stub只用于fixture，不能被当作真实认证或执行结果；生产无globals patches。

尚未执行MAIN生产analyze/verify，未报告任何真实新数值或效应结论。
首轮测试34passed/1failed：缺尚未创建doc触发三文件闭包拒绝；文档完成后重跑，不放宽门。
最终fixture回归35passed / 1.26s；MAIN生产消费未运行。
