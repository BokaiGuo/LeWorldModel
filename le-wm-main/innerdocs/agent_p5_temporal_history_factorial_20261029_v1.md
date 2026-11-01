# Temporal early-history：事前492模式 v1

独占source/tests/doc；不改任何旧交付，不运行worker生产prepare/NN/physics/fullstrict。MAIN在新结果前选择328两个新角+164 RR bridge，固定492call，不是328后补RR。绑定协议`innerdocs/agent_p5_factorial_next_identifiability_20261029_v1.md` SHA `8726514b653d85f4736102e06cc81b8652e46f9ebcab6c0875df2fe0141aee7d`。

## 固定输入与完整分母

原164枝、80signed pairs、四root×off/on层（各41/20），原两h×10坐标±及nominal全保留。temporal字母按时间5、10排序，不是旧视觉/proprio模态排序。每时槽visual与proprio成对取actual A或causal-rolled R，所有cell最新槽15固定short pred15，原actions packet[1,2,3]、raw6..10/11..15/16..20全buffer不变。额外agent position/velocity权限保留披露，非RGB-only、非独立fresh根。

| cell | slot5 visual+proprio | slot10 visual+proprio | slot15 | 执行 |
| --- | --- | --- | --- | --- |
| T_AA | A | A | 固定short pred15 | 继承old early AA输出，不新增 |
| T_AR | A | R | 同上 | 新164 |
| T_RA | R | A | 同上 | 新164 |
| T_RR | R | R | 同上 | 新164，所有native IO逐位核旧early RR |

新顺序严格branch-major T_AR→T_RA→T_RR；492predict API=492batch samples=1476frame samples，0encoder/proprio_encoder/AD/physics/training。旧656四角、之前short/H2成本单列inherited，不重复计成新调用，不称等FLOPs。T_AR/T_RA不能借用旧模态AR/RA；新prepare记录每枝两个新角与两个旧endpoint的exact input差异，保留相同枝；若全164某新角等于某旧endpoint或两个新角全同，fail closed，不重命名冒充新干预。

## 原材料身份与验证边界

ROOT=`/home/gbk/code/LeWorldModel/le-wm-main`。来源固定，不glob其他run，不按分数选枝：

- old early source `experiments/spectral_reachability/agent_p5_early_history_factorial_20261029_v1.py` SHA `39dbb6d61121ea9dbd6763be9b0104c45a8b3dbcf20ad3b090f21bed5a5645b7`；原prepare SHA `c8376a96a8320d4f147bf5e526954fbe8e31d37db565428ad82916dcc9d0f9c0`。
- report `results/agent_p5_early_history_factorial_20261029_v1_cuda_MAIN01/report.json` SHA `4fc86a7fdc69fbe95003e417ee38e3a4d430360a8eb5bd0c89f73ee845926d0f`；完整strict `results/agent_p5_early_history_factorial_20261029_v1_consume_MAIN01.json` SHA `d9461f7a21fa13454eca1f4371488431cf833a08ee758990c2dcb999b342600e`。
- 完整独立标量结果`results/agent_p5_factorial_independent_scalars_20261029_v1_MAIN01.json` SHA `ba881dc05522683fe44491a04e028c27dd66e20ae1bb4cd65be9a8cb22ee212e`，MAIN已报告actual analyze+verify完成；本层固定此分析文件及其arguments/schema/status/bindings，不伪造另外的verify receipt，也不重跑其analyze/verify。

prepare/verify正常复用纯codec/数学与已审loader类；无旧globals monkeypatch。`prior`只读完整持久receipt逐字段核schema/status/reportSHA/164/80/predictor计数/RR证据/闭门，核原report/prepare协议、模型before/mid/after/strict_loads、完整旧artifact目录SHA/size与canonical journal hashchain、原prepare全部bindings以及独立标量结果bindings。不调用旧build/verify/consume/fullstrict，也不反复递归读取并重评分旧结果。继承原完整strict的输入因果/label时序/模型加载语义，不把metadata/SHA核查当重新认证全部历史。缺项/变字节即拒绝，不猜loaded bytes。

每branch旧AA/RR真实inputs分别取原report `/calls/{4*i+0或3}/inputs/$tuple/0/$tuple/{0,1,2}/$array`，与row predictions/outputs交叉核；descriptor包含directory/file/SHA/shape/dtype/nbytes/file_bytes/report pointer。完整source NPY验证后才slice。还将AA/RR native inputs逐位接回原prepare requests的actual/rolled输入与旧纯splice规则，保留从H2实际5/10和short causal5/10/15的封存来源闭包；不会把旧full3早输出当新的pred5/10。

input shapes固定visual(1,3,1,16,16,384)、action(1,3,10)、proprio(1,3,256,16)，全部float32 finite contiguous。splice只成对替换槽0/1，槽2/action必须旧AA/RR本来全byte同且新native inputs逐位核对。native outputs完整visual(1,3,256,384)、None、proprio(1,3,256,16)全保存。T_RR三个inputs及全部full3 outputs必须逐位等旧early RR，无allclose/tolerance；差异保partial，不重试。

## label、seal与source/model闭包

共同label取old early已认证batch2实际15/20之slot1（与原prepare原H2 label descriptor SHA相等）。prepare只核未来label文件SHA/size/metadata，绝不decode未来label数值。全部492返回/validated/full native IO保存、全部164RR核对通过、model unchanged后durable全预测seal，才逐枝读取/复制batch2并算四cell20MSE。继承T_AA完整输出在seal内绑定，不放入predictor。

backend继承frozen probe/full-roll加载行为：同buffer实际state/load receipt与expected_model/strict schema对照、所有loaded模型hooks，禁止encoder/CPUfallback/grad/TF32、错误shape/device/dtype、额外predict，observed fused attention拒绝；requested MATH仍不冒充实际kernel证明。新自有Meter只准492个T3，旧656 guard不变。实际imported source必须属于原bindings且前后hash不变。wall包括verify/load/dispatch/终态hash；完整partial rows/pairs/失败不筛选，不称科学负结果。

## 预算、算术与严格消费

native IO=492×2,457,720=1,209,198,240B；共同batch2副本164×786,432=128,974,848B；合1,338,173,088B<2GiB tensor cap。prepare存refs，不复制旧inputs/outputs。total output cap3GiB包含本prepare实际bytes、NPY headers/journal/report、label副本；metadata32MiB/array4MiB，floor200MiB/reserve64MiB/wall900s，全部float32保精度。若资源不足fail closed/partial，不缩分母、截full3、降精度。RSS/GPU峰值是actual receipt，非tensor IO上界；RR只证明该输入在本job复现，不解决任意旧AA跨job数值全域确定性。

report为rows164（3新predictions+inherited_T_AA）、calls492、pairs80、summaries5、完整inherited/counters/模型/source/journal/inventory。score四cell共同H2 label20，float64 C-order all256×384 MSE。slot5=`((T_RA-T_AA)+(T_RR-T_AR))/2`；slot10=`((T_AR-T_AA)+(T_RR-T_RA))/2`；interaction=`T_RR-T_RA-T_AR+T_AA`；joint=`T_RR-T_AA`。纯数学复用旧effects/pair时显式将模态键映射回slot5/slot10，不误标visual/proprio。保所有vector-effect L2、80pair norm/dot/cosine/error、/2h central差分、全/四层分母与全部六cell-pair符号转移；exact零范数→null，无epsilon、无按56非零筛选、无择优角。单时槽内两模态不再分解，不声称完整高阶16角/物理因果/任务成功。

新consume全量验证本job inventory/NPY、完整journal每字段与chronology、每call native input/全输出/metadata、三新槽顺序/RR bitwise、seal-before-label、共同label字节、继承AA身份、所有MSE/效应/pair/四层/计数/实际字节/模型与source不变。不重跑上游fullstrict。complete status=`all164_temporal_two_corners_RR_bridge_and80_pairs_verified`，partial只验证inventory/hashchain不作完整认证。P1/P5 false，不以工程完成宣称实验改善或独立确认。

## MAIN审读后的命令

在ROOT执行；worker不运行这些生产命令。SHA参数取实际返回，不猜未来report：

```bash
M=experiments.spectral_reachability.agent_p5_temporal_history_factorial_20261029_v1
P=results/agent_p5_temporal_history_factorial_20261029_v1_prepare_MAIN01.json
O=results/agent_p5_temporal_history_factorial_20261029_v1_cuda_MAIN01
.venv/bin/python -m "$M" prepare --output "$P"
.venv/bin/python -m "$M" verify --prepared "$P" --prepare-sha PREP_SHA
.venv/bin/python -m "$M" run --prepared "$P" --prepare-sha PREP_SHA --output "$O" \
  --authorization MAIN_P5_TEMPORAL_HISTORY_FACTORIAL_HOSTCUDA_ONLY
.venv/bin/python -m "$M" consume --output "$O" --report-sha REPORT_SHA \
  --receipt results/agent_p5_temporal_history_factorial_20261029_v1_consume_MAIN01.json
.venv/bin/python -m pytest -q tests/test_agent_p5_temporal_history_factorial_20261029_v1.py
```

测试只synthetic native-shaped temp files和fake tensors运行真实Store/Trace/本consume，N=1只改本NEW模块；原164/80/492guard及fourstrata另有RAM全量fixture。真实MAIN prepare/CUDA/strict尚未运行，fixture通过不是科学通过。需原全部source/model/config、report/prepare/strict/独立分析文件可读且SHA不变。独立根确认、采集预算和科学准入均另依原协议，不在此新增或放宽。
