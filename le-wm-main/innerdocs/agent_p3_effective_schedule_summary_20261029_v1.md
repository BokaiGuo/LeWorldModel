# P3 effective schedule：saved-only固定配对汇总

## 边界与输入权威

本模块仅消费已保存数据，不加载模型、不调用神经/物理、不重新编码。
固定两条已知根、8臂、24反馈节点，保留全部分母及失败/早停缺项。
不签P1/P3/C3，不宣称fresh、独立8根或equal FLOPs；预算保持`notmatched`。

外部必须提供output目录、report SHA、prepare路径/SHA及strict路径/SHA。
固定owner源码/tests/doc三SHA；读取prepare全部sources绑定，并在分析前后重hash。
`analyze`实际调用冻结owner的`consume(output, report_sha, prepared, prepare_sha)`，
要求返回完整8臂/24节点receipt与外部strict逐字段规范JSON相等。
不能仅凭status/complete字段通过。原strict负责原生/RGB/模型ledger和数组闭包，
本模块独立计算下列数值，不冒称一次新的神经验证。

`verify`再次完整重建上述strict和汇总；因此可能较慢，不能用fixture耗时估计生产耗时。
worker只运行synthetic fixture，生产analyze/verify由MAIN执行。

## 固定比较及缺项

每根固定4个比较，共8对，差值统一right-minus-left：

- 同bank：commit15 → feedback5（endpoints与midpoints分别列出）。
- 同schedule：endpoints → midpoints（commit15与feedback5分别列出）。

跨bank比较明确是候选bank干预，不能称纯调度因果。
每臂输出step0..20与cp0/5/10/15/20，每对保留21个位置。
仅真实共同时间有差值；早停后为null，不复制终态、不padding。
终态另标eventtime；两侧终止时间不同则终态差值为null。
warmup发生在当前task goal绑定前，不纳入当前goal任务指标。

## 数值与支持语义

独立从真实goal7/state7计算jointXY、agentXY、objectXY、wrapped angle、
未wrap的native7距离/负距离reward、完整7维delta及速度误差。
原native success保留原XY和min-angle表达式，包括超2pi时该表达式的负值；
wrapped angle只是另列诊断，不替换原native门。

独立按原identity192完整维度重算mean squared latent Q；拒绝非identity投影。
重算每个真实编译节点全部4候选terminal Q并核保存cost；保留原winner，不重选策略。
selected predicted Q只对齐同预测endpoint的真实encoder Q，缺项为null。
step0 Q来自封存原root编码（原strict验证实际重放RGB），不是本轮新增encoder；
其他没有实际编码的step Q保持null，不由物理状态或相邻checkpoint插值。

逐时列RGB像素hash/PNG bytes hash相等、latent bits、执行rawfloat32 bits及各轴误差。
候选支持在每boundary列全部4×4 packet raw5 exactbits匹配索引，保留重复匹配。
commit15后续packet来自boundary5原H3 bank切片，记录原context、horizon与origin；
绝不称其为当前boundary真实RGB上重新评估的候选。支持相同不代表预测context相同。

## 实际计账

从native operation intent/return按实际units独立求和；从每node model ledger delta
逐轴求attempted/completed，并核report全局/每臂计数。predict/J/VJP及其他所有原轴分别保留。
encoder/projector、public/physics/render/setup/close等native轴亦分列。
inclusive operation wall为嵌套测量，不相加冒充whole wall或FLOPs；whole-job wall另列。
即使某些调用轴恰好一致，预算仍为notmatched，不推断科学效能。

## MAIN命令

在`/home/gbk/code/LeWorldModel/le-wm-main`执行。以下report/strict SHA须填生产真实值；
当前生产尚排队，未生成或猜测任何未来SHA。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p3_effective_schedule_summary_20261029_v1.py

.venv/bin/python -m experiments.spectral_reachability.agent_p3_effective_schedule_summary_20261029_v1 analyze \
  --output results/agent_p3_effective_schedule_20261029_v1_MAIN01 \
  --report-sha "$P3_REPORT_SHA" \
  --prepared results/agent_p3_effective_schedule_20261029_v1_prepare_MAIN01.json \
  --prepare-sha 8e0a80f9b907e7d8b61c605f28c73f295f53ab1447c973fe8b99ebbd241f01ee \
  --strict results/agent_p3_effective_schedule_20261029_v1_consume_MAIN01.json \
  --strict-sha "$P3_STRICT_SHA" \
  --receipt results/agent_p3_effective_schedule_summary_20261029_v1_MAIN01.json

.venv/bin/python -m experiments.spectral_reachability.agent_p3_effective_schedule_summary_20261029_v1 verify \
  --summary results/agent_p3_effective_schedule_summary_20261029_v1_MAIN01.json \
  --summary-sha "$P3_SUMMARY_SHA"
```

新receipt只能写本stem的results新文件，禁止覆盖；128MiB上限，文件flush/fsync后目录fsync。
stdout仅给路径/SHA/status，不打印全数组。

## fixture覆盖与未完成项

23项非神经fixture覆盖固定分母、earlystop14与cp15缺失、eventtime不等、
候选原context/支持语义、完整192维Q、原native角度和速度、非法向量/raw/Q/资源计数、
外部SHA/源漂移/status-only receipt拒绝，以及完整strict调用与确定性verify接线。
synthetic resource事件是测试算术用聚合记录，不声称真实native writer或producer证书。
测试中替代loader/strict仅限fixture；生产源码不patch原globals。

尚未执行：MAIN生产report/strict绑定、实际analyze及verify。
这些真实数据未齐前不能报告任何观察结果或策略优劣。
