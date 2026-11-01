# P2原两根 H3 bounded native CEM：实现交接

仅新stem源码/tests/本文；旧H2/官方slice/paid mean及全部结果不改。
使用experiment-code skill组织实现与fixture验证，不执行其训练/选模建议。
批准协议：`innerdocs/agent_lewm_bounded_cem_horizon_comparison_protocol_20261029_v1.md`，
SHA `354b5e8882989ba1e870a54fde32ad28725582b0ac46f184817b40434f1a6f84`。
本交付不是生产完成、官方复现、Hi训练scaler认证、fresh证据或科学通过。

## 固定输入与算法

所有路径根为 `/home/gbk/code/LeWorldModel/le-wm-main/`。
直接读取 `results/agent_p2_common_resource_grasp_cuda_20261029_v1_prepare_MAIN01.json`，
SHA `7facba500439172ced280ec0b893659715139b1e5e22073ae7313665a1c0faec`。
不通过H2 task改名来构造P2身份，不读取旧效果决定算法/根/配额。

| root/off | P2原task文件 | task SHA |
|---|---|---|
| 0 | results/p1_moving_overlay_recompile_cuda_20261027_v1_cuda_MAIN01/task_00.json.gz | 80f0f4ed14498509e596322b830f0f8d1bedc4d38f9d3b7d64b57da375f9a132 |
| 1 | results/p1_moving_overlay_recompile_cuda_20261027_v1_cuda_MAIN01/task_02.json.gz | 009ed5769fb55513100c1223c5659bd5b34911b5cfe13e9b121117c14a0db2d7 |

原task_id/context3×192/goal192/identityQ/raw warm10及工程normalizer逐项绑定。
warm2保留原时间动作，未来3包到原frame25；不补zero、不换goal、不挑可部署根。
完整blind RGB0/5/10与goal仍保存其真实像素及变换输入，但cached rollout不重新编码。
任务mean/std沿原P2 normalize_live_packet，非Hi原训练scaler。

复用安装CEMSolver，不改其全局变量：N300/I30/K30/batch1/var_scale1，
每root显式MAIN本轮事前seed0（不是用户指定；安装默认1234保留）。
全零初始logit mean，单位std；每轮CUDA f32 tanh一次得到raw全box[-1,1]，
一次normalize后拼warm2，完整action5。原CEM在logit空间更新mean/std(ddof1)。
原生criterion始终float32 terminal sum192，不以elite average冒充mean评分。

每轮全300预测/native IO durable后才criterion/update；final mean再次单次tanh为raw15，
封存后从实际NPY重载，不二次tanh/clip。原normalize生成实际部署token；
whole3与segmented1+1+1各独立预测，后者用自己的pred移窗和原action shift。
两根×两布局全部四路封存后才四次final criterion。全future576及frame15/20/25分别
报告bits/maxabs/ULP；不设置事后神经等价容差，不自动升级parity。
float64 sum192/mean192另算评价，不反向替代规划criterion。

## 独立H3 ABI与实际计量

本stem自有Trace/whole_chain/validate_delivery；不调用旧H2 shape guard，
不monkeypatch任何旧module/global。复用旧loader、codec、NoNeural与算术诊断函数，
其精确源码SHA进入新prepare闭包。

planning每轮AE输入[N,5,10]→[N,5,192]；随后3组predictor/proj，共7个module。
predictor两输入均[N,3,192]，输出同shape；proj输入/输出[3N,192]。
final每root whole7+segmented9=16modules，seg AE始终[1,3,10]。
encoder和image projector hooks主动拒绝；inference/f32/CUDA/no-grad/autocastoff/TF32off逐调用核。

| 分轴，两root总量 | 固定预算 |
|---|---:|
| planning cost oracle / final及planning criterion | 60 / 64 |
| official rollout / segmented predict wrapper | 62 / 6 |
| predictor API / samples / context tokens | 192 / 54012 / 162036 |
| pred_proj API / projection rows | 192 / 162036 |
| AE API / samples / tokens | 68 / 18008 / 90028 |
| tanh / normalize API | 62 / 62 |
| 新encoder/image projector/AD/physics/training | 全0 |

`actual_modules`按actual native IO重建attempted/returned/sample/token；缺输入计null，不填0。
计量不是FLOPs证明。继承编码单列：原
`results/p1_moving_overlay_recompile_cuda_20261027_v1_cuda_MAIN01/report.json`，
SHA `2b2e5064f74f8aca712ae0d0415fcbfb64a2c8e6325bdffe24ae4fdc3b373381`，
四off/on task原encoder/projector各16API，不是新job调用；off-root成本归属unknown。
不复用H2的8次编码声明来抵扣P2来源。

## 完整shape census与保护

纯整数shape枚举每次Store.array（包括重复mapping/norm/native IO/回调副本）：

| 项 | 字节/数量 |
|---|---:|
| NPY数量 | 2170 |
| 原f32/int64 payload | 804556336 |
| 最大单数组payload | 1806336 |
| 每NPY header保守限 / 每数组元数据额度 | 256 / 65536 |
| 每journal event额度 | 2048 |
| working总上界 | 950741040 |
| 2GiB减128MiB可用工作区 | 2013265920 |
| working加128MiB保留总上界 | 1084958768 |
| 整个目录cap | 2147483648 |

真实临时Store→NPY→strict reader fixture逐项核数组数量/payload/header和journal行限；
生产shape census无模型执行。此为磁盘上界，不是假称CPU/GPU峰值已认证。
wall900s含verify/load；写入/module前协作guard，不能强杀单个卡住kernel。
GPU allocated3GiB/reserved4GiB、floor200MiB；失败finally保留partial report/完整两根槽位，
partial不允许strict通过，不缩分母/精度/预算补过。不会自动重试。

## MAIN完整CLI

以下命令仅交接；worker未执行生产prepare/verify/run/consume。输出必须不存在。
`PREP_SHA`/`REPORT_SHA`须来自MAIN实际写出文件；不预填未来receipt。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m pytest -q tests/test_agent_lewm_bounded_native_cem_h3_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_lewm_bounded_native_cem_h3_20261029_v1 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_bounded_native_cem_h3_20261029_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_lewm_bounded_native_cem_h3_20261029_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_bounded_native_cem_h3_20261029_v1_prepare_MAIN01.json --prepare-sha "$PREP_SHA"
.venv/bin/python -m experiments.spectral_reachability.agent_lewm_bounded_native_cem_h3_20261029_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_bounded_native_cem_h3_20261029_v1_prepare_MAIN01.json --prepare-sha "$PREP_SHA" --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_bounded_native_cem_h3_20261029_v1_cuda_MAIN01 --authorization MAIN_BOUNDED_NATIVE_CEM_H3_HOSTCUDA_ONLY
.venv/bin/python -m experiments.spectral_reachability.agent_lewm_bounded_native_cem_h3_20261029_v1 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_bounded_native_cem_h3_20261029_v1_cuda_MAIN01 --report-sha "$REPORT_SHA" --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_bounded_native_cem_h3_20261029_v1_consume_MAIN01.json
```

prepare/verify只重建saved输入/source闭包，不递归上游fullstrict。
consume完整核本job canonical单换行hashchain、每轮/全final barrier、全NPY SHA/shape/dtype、
原任务/上下文、每module因果链、topk/mean/std、actualraw重载/归一化、资源与全部artifact引用。
它不能用saved数组独立重新执行CUDA tanh：NumPy tanh只作diagnostic，不是CUDA逐位证书。
normalizer CPU f32 mirror要求exact，真实CUDA若不同则fail closed，不能追加容差。
source/model文件hash+加载后tensorhash不是同buffer loaded-byte/training-lineage完整证明；
requested MATH及观察guard不是实际kernel认证。实际CUDA ABI/内存峰值/规约仍待MAIN生产。

## 比较与未完成边界

新H3与P2只对齐原任务/时距，**非公平排名**：CEM全box vs P2 joint球、f32sum vs
f64mean、batch300/54012 samples vs P2 batch1预算、AD/wall scope皆不同。
原H2及其它baseline保留各自栏，不将H3结果回证H2。
本runner没有新动作physics/RGB；真实native成功、same-model RGB Q均unknown。
后续必须另获MAIN授权、用原recipe真实warm10+raw15及独立资源账，不能挪用P2现成终点。
不改变P1/P2/gate，不宣称控制收益或freshness。RAM fixtures通过仅证明工程接线，不等于科学通过。

## 实际执行的轻量验证

H3独立79passed/13.34s；以下联合214passed/33.61s（仅fixture/临时文件，非生产strict）：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m pytest -q tests/test_agent_lewm_bounded_native_cem_h3_20261029_v1.py tests/test_agent_lewm_bounded_native_cem_20261029_v1.py tests/test_agent_lewm_bounded_native_cem_20261029_v1_seams.py tests/test_agent_lewm_native_cem_slice_20261029_v1.py tests/test_agent_lewm_finalmean_paid_evaluator_20261029_v1.py
```

覆盖真实Store→NPY重载→完整saved reader、N300/final packet reshape、再次tanh拒绝、
重算合法hashchain的planning/final barrier篡改拒绝、缺槽/身份/计量/模型/限额拒绝、
新hooks完整7/16模块shape及encoder禁用/异常清理。fake tensor仅检接线，不冒充CUDA认证。
冷CLI与shape census确认未导入torch/pymunk；旧H2源码/tests/doc、seams和批准协议SHA复核未变。
本交付无生产输出/report/receipt，不预填未来SHA。
