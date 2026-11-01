# Moving teacher horizon localization：固定自身动作H1/H2诊断

## 实际来源与合法时间尺度

本实现只占新stem source/tests/innerdoc，不修改封存文件或Nash误差分解文件，不执行模型或物理。
使用experiment-code技能做窄实现和RAM/mock验证；用户的无训练/无拟合/无额外实验要求优先，不采用训练/扫参/画图模板。

已纯saved读取原live report/journal/PNG：report SHA
`990843e8bba15955db5163b9b5d77c1f8eecaa1870d3745b49d46cc22ca895e3`，
8334条journal事件，12 episode各22PNG：warmup0..10及plan0..10，合计264PNG的文件/pixel SHA、uint8[224,224,3]均核对。
每episode warmup/control实际各10条raw float32 bits完整，plan checkpoints为0/5/10。
这不是新增采集，未运行reset/render/encoder。build再次强制上述全量来源核对，不只相信本doc计数。

原训练/冻结接口的action_dim10是一组5×2 raw动作；**5 raw steps=1模型packet**。
plan raw6/7/8/9虽有真实PNG，但没有授权的stride1模型；不得将单raw动作填充/重复为一个packet或把4个中间帧造为四个模型successor。
无需新编码raw6–9。现有真实packet边界plan0/5/10与原source context足够构成合法H2对照，最大可验证未来horizon为2；更远终点缺实际动作/观测，本文不补造。

原teacher的三个输出token也不是三步自由rollout：token0是source_z5→source_z10，token1是source_z10→ownprefix5，
token2才是ownprefix5→terminal10。当前原R−O三层排序反转本来已经是**真实prefix条件下suffix H1**证据，
不能称由三token或三packet开放滚动累积导致。新H2实验只定位改变prefix context为预测值后的敏感性；原H1证据单列保留。

## authority与明确复用范围

原live report/prepare/analysis/task/model沿冻结teacher v2 `build`的完整外部SHA和因果joins。
核心source SHA固定`492ff318c83a312be5f5f6efd85e49ad06eb56457d5c0e92c3b92ee617a38a1b`。
原teacher actual report固定路径
`results/p1_moving_live_feedback_teacher_fidelity_cuda_20261027_v2_cuda_main01/report.json`，
SHA `7323347a9e74c8f1751550642503426d07ddce55549ddaef4331f3c362567128`；
新build只hash它作解释authority，不读取其预测/效果来构造或筛选新输入。

显式复用冻结core的Store、Native原模型模块hooks、AD/image guards、纯saved strict joins和原normalizer。
新Ledger明确覆盖begin的固定36上限；不更改core全局，不冒用其execute/run/consume。
内部单次forward请求明确沿用core `.forward_input`白名单schema以使用同一冻结Native验证器；
本实验prepare/envelope/report/consume全部是新schema，不冒充原teacher12次实验。
新source/tests/doc全部进入源闭包，core及其原文件绑定仍保留。不得在prepare后修改其中任何文件。

原两个moving任务/root/goal/identity192 Q/normalizer/合法原raw动作均不变。
原模型eval/frozen、tensor SHA前后相同、HOST CUDA-only、MATH、TF32off/autocastoff/inference_mode，
无CPU neural fallback、训练、AD/J/VJP/backward/SVD/solve、encoder/image_projector或physics。

## 固定12条、同动作三次forward

cohort不增删：2source_slot×GN/D2P×O/I/R，仍2个原known roots，不是12个独立任务。
每条用自己的实际prefix P5和suffix S5，禁止用另模式动作或标签拼接；没有候选/选择/zero nominal/retry。
I也以actual prefix做本诊断，不倒写成I部署时观察了actual prefix。

记原root context为Z=[source_z0,z5,z10]、实际warmup raw10为W，actual prefix latent为y5，terminal为y10。
绝对时间：plan0=source10，plan5=source15，plan10=source20。归一化只作原float32 subtract/divide，time-major。
每条固定执行顺序（batch1）：

| stage | emb3 | outgoing3动作tokens | 使用最后输出 |
| --- | --- | --- | --- |
| prefix_H1 | Z | normalize(W[0:5]), normalize(W[5:10]), normalize(P5) | p5 |
| open_H2 | [z5,z10,p5] | normalize(W[5:10]), normalize(P5), normalize(S5) | p10_open |
| teacher_H1 | [z5,z10,y5] | 同open_H2，逐位相同 | p10_teacher |

open_H2只有第二次forward，第一步prefix_H1已实际计算并复用其输出，不偷偷重算第一步凑预算。
H2是shift原context并加入最后预测successor，不把prefix forward的三个输出全当新的三帧历史。
源动作历史完整归一化2token，含实际warmup与prefix，绝不zero pad/除100/双normalization/tanh历史。
open_request全量重建[z5,z10,p5]，不能残留actual y5；consume从已保存p5独立重建并逐请求核对。
同suffix的teacher与open差异只在prefix latent，history/raw5不变；但这仍不是独立干预/通用因果证明。

## 预测/标签隔离

prepare可以像原teacher一样读取actual prefix作为teacher_H1的合法已知过去；root/open请求完全不含该观测值。
actual prefix也是prefix_H1评分目标，因此不能夸称“整个进程从未见过prefix标签”：
隔离保证是每次forward只得到自身窄request，root/open不含actual prefix，teacher只含合法过去。
终态y10的NPY在prepare只hash封存receipt，不解码、不进入任何请求。
goal/state/native outcome/未来path不进入forward白名单；goal仅评分域可读。

全部36预测与原始module trace先durable保存，并写predictions_seal和all_predictions_sealed事件，
之后才由label_loader读取自身prefix/terminal数组评分；无模型调用或label解码回流selector。
prefix评分域的首次label_loader调用也在全36 seal之后，不把其先前conditioning权限描述成完全盲标签。
跨episode source/identity/goal/task/selection join失败则拒绝，不用相同数值的他人文件顶替。

## 预算与工程合同

新增恰36 predict API、36 action_encoder、36 predictor、36 pred_proj，144个API/module entries；
API与子模块不是相加为72次forward。12prefix +12open suffix +12teacher suffix，每次context3输出3token。
encoder/projector图像编码=0，AD/J/SVD/physics/reset/render/publicsteps=0；0新图/标签/训练。
保存各module全部float32输入输出，CUDA同步后才记completed；AE先执行，API随后，不能虚报AE失败时的API attempt。
actual attempt/completion不退款，异常保全部12行/36固定预测分母及实际未完成阶段。

完整输出≤8MiB＋200MiB floor，普通写留256KiB final reserve；独占NEW目录，存在则拒绝，无删除/重试。
prepare磁盘≤256KiB，单gzip/base64 envelope；semantic decoded canonical≤1MiB，CRC/尾随/双gzip/semanticSHA/全build重建严格验证。
完整bindings不裁减，若实际超cap则拒绝，不静默提高cap。
whole-run300s从run入口含verify/load/forward/score/final；CUDA allocated≤3GiB/reserved≤4GiB。
source verify失败在模型加载前拒绝；运行中异常保first/secondary和已发生证据，cleanup/hash/report分别尽力保存。
pre-verify未成功时没有已启动模型的虚假report；最终report写盘失败明确stderr，不冒称完成。

## 输出与纯saved consume

schema为本stem.report，正常status=`horizon_localization_recorded_not_admitted`；partial为`partial_or_failed`。
rows固定12，每行identity、status、predictions三stage receipts、labels(prefix/terminal)、metrics。
每stage指标沿原teacher mean192 float64：factual_MSE、predicted_Q、actual_Q、predicted_Q_minus_actual_Q、actual_prefix_terminal_mse。
prefix_H1的最后一个变化量是root→actual prefix；两个suffix stage则是actual prefix→terminal。

额外原数组描述：

```text
d = p10_open - p10_teacher
e = p10_teacher - actual_terminal
propagation_gap_mse = mean(d²)
propagation_cross_term = 2*mean(e*d)
open_minus_teacher_factual_mse = Mopen-Mteacher = mean(d²)+2*mean(e*d)
```

交叉项可负，H2误差不必大于H1，不能把gap直接解释成全误差中“累积占比”。
若H1原排序已反转，新H2改变排序也不能抹去该H1证据；若H2恶化，只能描述本轨迹预测context替换后的差异。
无拟合、误差通过阈值、显著性、独立泛化或P1/P4科学准入。

consume只读：外部report/prepare/source/model绑定，严格artifact集合/全SHA、事件hashchain、
全部36实际prediction_saved在seal前、12评分在seal后、144 attempts/returns与原ledger精确闭合，
自身标签/原数组/指标复算、3token module桥、最终192预测来自真实pred_proj、open上下文重建。
额外预测比较/原teacher bitwise对照如需解释，只能使用已保存数组，不触发新推理或补跑；本实现不将旧teacher当标签。

## MAIN命令与测试范围

cwd `/home/gbk/code/LeWorldModel/le-wm-main`；本交付不prepare/run实际作业。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_moving_teacher_horizon_localization_cuda_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_teacher_horizon_localization_cuda_20261027_v1 prepare --output results/p1_moving_teacher_horizon_localization_cuda_20261027_v1_prepare_main01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_teacher_horizon_localization_cuda_20261027_v1 verify --prepared /ABS/NEW_PREPARE --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_teacher_horizon_localization_cuda_20261027_v1 run --prepared /ABS/NEW_PREPARE --prepare-sha ACTUAL_PREPARE_SHA --output results/p1_moving_teacher_horizon_localization_cuda_20261027_v1_cuda_main01
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_teacher_horizon_localization_cuda_20261027_v1 consume --output results/p1_moving_teacher_horizon_localization_cuda_20261027_v1_cuda_main01 --report-sha ACTUAL_REPORT_SHA
```

RAM tests只用人工数组、原mock module/tensor及内存PNG，不生成pytest tmp/prepare/研究结果。
覆盖36次全seal后评分、open/teacher输入隔离、实际raw chronology、全12/36失败分母、部分失败不退款、
AD/image/CPU拒绝与hooks恢复、264帧来源缺失/坏bits/guard/cp拒绝、单gzip上限、完整saved consume和误差交叉项。
纯saved build smoke若由本agent执行，只说明实际字段接线可读，不能当CUDA实验成功。
