# 官方CEM已交付均值：P2 raw-domain / roundtrip saved桥

只补已评known/off两根root0、root1返回均值的可部署动作前置。无NN/physics/fullstrict，
不改变旧source、原mean cost或旧slice的`final_mean_evaluated=false`。
MAIN已提供临时算术初查；本实现不是盲确认/新科学预注册，公式和精确门沿用此前批准的原P2合同，
不根据临时查值改规则、挑根或生成只含可部署根的新实验。

## 固定authority

- paid mean report：`results/agent_lewm_finalmean_paid_evaluator_20261029_v1_cuda_MAIN01/report.json`，
  SHA `e7332cc12f75a8081405bfeed3da9db038ecab00d9234ebbb24f387c182dbebc`。
- paid mean完整receipt：`results/agent_lewm_finalmean_paid_evaluator_20261029_v1_consume_MAIN01.json`，
  SHA `622754ccbd87a860598040d65af35572bb0acab0ee7be435e3a2604cf6564ee3`。
- paid prepare：`results/agent_lewm_finalmean_paid_evaluator_20261029_v1_prepare_MAIN01.json`，
  SHA `f88321adc69f1175d9a02533867d6107aa49beb9c7c7534871320256b85b1313`。

完整绑定该三文件及其记录的原native CEM report/receipt/prepare：
report `ab3313d24b6efb3539bb1b1a70417f7aa41df1602157465d1da5cd1d26e8b965`，
receipt `85af90a95df1481f1bb2491f1db0adfd307be50d0d7a1f636d3e4b0a712e5ada`，
prepare `f8f773572326ddd52c407d14b850584228f29c5c3a14d0f44262129a676d276c`。
通过原prepare明确slot/task指针，不按文件名猜身份；对应任务为原task00/01压缩导出，
SHA分别 `a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b`、
`101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74`。

不调用owner consume或旧verify/build，不嵌套fullstrict。
以完整持久receipt及外部SHA为原预测真实性的trust起点，核schema/status/模型/来源、全2槽，
重读本桥真正需要的paid动作/emb/warm/goal_emb与原task，逐位核原返回actions/mean/末callback.mean；
其余完整神经输出验证依赖已有receipt，不声称重复独立认证。
继承prepare完整sources并核当前文件SHA，另绑定新增source/tests/doc、domain与scaler源码。
config/weights文件SHA只是当前字节闭包，不是过去同buffer加载或训练来源证明。

## 原P2 normalization，不是Hi scaler

`experiments/run_pusht_counterfactual.py`，源码SHA
`0a4eb4592951139e596b35becede17d175df7b48a2fde16679ab0aac7c4dce34`，
AST读取显式`np.asarray(...,dtype=np.float32)`：mean `[-.0087,.0068]`，std `[.2019,.2002]`。
全部与实际task的`action_mean/action_std`逐位比较后才计算。
这是已认证P2工程协议，**不是Hi sklearn原scaler，不证明训练统计真值**；
`training_provenance_verified=false`保留。

原`pusht_d2p_packet_contract.py::normalize_live_packet`，SHA
`eda631348e929b45591807dac0adcbb72d10d84b7d48a7a558a492eab9c04aef`：
time-major原raw5x2 flatten10，mean/std按xy平铺5次（非repeat-interleave），分步subtract/divide，
不再除100。函数自带[-1,1]拒绝。
工具不导入torch；域内另用原P2 `core.normalized_tokens` NumPy helper逐位交叉核查。
这属于source-bound saved-array算术，不冒充本轮真实HOSTCUDA normalize执行证书。

`packet_commands`纯数组函数实际调用检查原两packet；安装版native env.py SHA
`c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2`，
action_space为float32[-1,1]^2；原relative command在step乘action_scale100后加agent位置，
本桥不得额外缩放或靠环境补clip。只读source，不import/实例化env。

## 数值规则与产物

固定tokens `(1,2,10)`，先float32 `tokens*std10`，再独立float32 `+mean10`；
无FMA合并，得到原顺序raw10x2，不clip/tanh/reselect。
按每个原float32元素精确包含于[-1,1]作domain判定，无epsilon。
保存全部20元素inside mask、packet接受状态、越界索引/packet/offset/axis/值/bits，raw_min/max只是诊断。

无论domain是否通过，分步float32 `(raw-mean10)/std10`计算roundtrip诊断，保存全部中间值。
**域外是代数诊断而非合法normalize_live_packet返回值**，仍标`blocked_raw_domain`，不隐去另一根。
域内但roundtrip不bitexact标`requires_paid_deployment_token_replay`，
`old_mean_cost_certifies_roundtrip_tokens=false`，必须另外付费评估实际部署tokens，禁止沿用旧mean cost。
若域内且bitexact也只标saved前置相同，不是物理授权/科学准入；未来CUDA输入仍需实际绑定。

每个float32数组保存shape/dtype/全values/uint32_bits/完整rawbytes hex/SHA，
包括tokens、mean/std、平铺stats、inverse product、raw packets/raw10、roundtrip subtraction与tokens。
roundtrip差异保存全部20元素bit_equal、ULP、float64 delta及总bitwise/maxabs/max_ULP。
±0 ULP同位但bitwise差异明确报告；非有限值拒绝，不补0。
verify按完整canonical bytes重建，不能把JSON -0/+0相等当逐位通过。

输出一个新独占JSON，fixed2根全部保留；无可部署根shortlist或自动实验授权。
`physical_release=false`、`scientific_admission=false`不论分析状态如何都保留。
工程上限8MiB输出、disk floor200MiB、saved重建wall900s；正常完整报告远低于上限。

## MAIN CLI

从`/home/gbk/code/LeWorldModel/le-wm-main`执行：

```bash
.venv/bin/python -m pytest -q tests/test_agent_lewm_delivered_mean_raw_domain_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_lewm_delivered_mean_raw_domain_20261029_v1 analyze --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_delivered_mean_raw_domain_20261029_v1_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_lewm_delivered_mean_raw_domain_20261029_v1 verify --report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_delivered_mean_raw_domain_20261029_v1_MAIN01.json --report-sha "$ACTUAL_ANALYSIS_SHA"
```

SHA变量取真实analyze输出；不预填未来产物。拒绝覆盖/symlink/非独占prefix。

## 测试与未运行项

实际新33项通过（0.36s）；与两冻结前件联合94项通过（4.27s），CLI help exit0。
RAM测试包含：全20slot time-major、精确边界外1ULP拒绝、域外仍保留代数roundtrip、
域内非bitexact需新replay、±0、全部字节/bits、authority错接/task/scaler/模型/原mean/分母拒绝，
真实独占writer→完整verify及报告破坏拒绝；冷启动CLI不导入torch/pymunk。
无worker实际cohort analyze/verify、NN/physics/fullstrict/下载/后代agent。
MAIN尚需审读并执行真实saved analyze/verify。所得部署blocked或replay需要不是科学负结果，
也不能按其中合法根扩新实验或把两known roots变成fresh。
