# 七signal / 五U：saved-only完整谱与目标方向算术

独占NEW source/tests/doc；不改Stage B、原solver、chart、normalizer、gate或任何封存产物。
worker仅RAM与真实字段/shape检查，不运行本版生产build/verify、NN、AD、physics或上游fullstrict。
MAIN完整审读及独立测试后运行saved build/verify。本版不是新controller，也不授权阶段C。

## 仅两个数据输入与继承边界

全部CLI文件路径必须canonical absolute。两个输入均按完整原始字节外部SHA读取：

- Stage B实际完整MAIN receipt：
  `/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_capability_probe_cuda_20261029_v1_consume_MAIN01.json`
  SHA `682a78366b10a60895c4530f2fc6cc401e6439b625f4ba0a8aa996dc2fed79a7`。
- 原同时间native endpoint独立summary：
  `/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_secant_endpoint_summary_20261029_v1_MAIN01.json`
  SHA `6a6513ebfa68f331bbbecac139af84b0614446e3df739b3d00959a3119f743a0`，本版固定该已完成输入，不切换对照。

Stage B receipt继承的实际权威为prepare SHA
`90b3a1a6284c5605594d4341a2f994870e4b8db9c1721ba40f0160244082e870`、
report SHA `a64fbf31cf2601cf6ea26d100ef5cb5c01efbe8b63be6cf33bc4d91d735c25fc`、
completion SHA `2d4f8364867ccfba4bb15f4794f0f665af34483cf4bdb1adf57af4efb1f27574`。
本工具从实际receipt读取/交叉连接这些域，不硬编未来权威、不打开这些祖先文件、不重新消费其IO日志。
完整native17、7J/1344VJP、模型/精度/源码认证来自MAIN已经完成的strict，非本summary重新证明。
拒绝RAM/partial receipt、错误固定分母、未闭合native验证、资源向量不足或科学标记非false；不凭status单项认领完整。

除两个JSON数据输入外，只读本三件套及显式SHA绑定的冻结源码/协议字节用于source closure，
不import任何producer，不访问权重、NPY、PNG、运行目录或其bindings里的祖先数据。
cp10完整task/goal/context在五plans中，逐位核验；cp5原task本体不在receipt，
故cp5 task与实际历史认证明确继承原strict的signal binding/parent-context，
其goal只使用同root五plan所认证不变的原goal，不冒称独立重建cp5 prepare。

## 固定分母与接线

7行signal顺序：cp5/root0、cp5/root1、cp10/(r0,C)、(r0,Z)、(r0,P)、(r1,Z)、(r1,P)。
全部192×10 J共13440元素、完整70 singular values均输出，不按强弱择行/择根。
五probe原始次序不变；whole/segmented各1包H1，从同一actual cp10上下文开始。
全部task SHA、goal/identityQ/normalizer/context digest、原signal provenance、完整witness digest、
native call IDs、原params→float64 tanh→一次float32 cast的raw位模式均核验。
incoming raw10按原f32 mean/std重建**两个**时间token；outgoing只normalize一次。
核完整三latent/twoaction的shift、whole/segmented逐字节JSON一致；不只核terminal latent。
实际endpoint按base_task_id、control10、完整snapshot、state、PNG字节SHA、pixel SHA与原summary连接；
最后一history帧和原C或Z/P物理receipt/report域也须一致。state-only相等不替代snapshot。

原4逻辑C槽×Z/P=8比较全保留，后三个C no_witness产生6个null，不填0/tie。
另事前完整列同root同cp10所有实际角色对：r0 C/Z、C/P、Z/P，r1 Z/P，共4行。
不是新增candidate或选择；仅比较已固定5端点。两个known roots不变，独立新根数0。

## 独立目标方向与谱分解

不import原bounded_ridge或任何producer数学。每个J只新做1次本地CPU NumPy thin SVD，
7次**saved数组运算**，不是NN/AD。令J=L diag(s) Vᵀ、r=goal−nominal、t=Lᵀr，
全部10个方向的dot使用192项math.fsum；保留r⊥=r−Lt、其非负平方范数以及分解舍入差。
不设数值秩阈值，不丢微小/零singular，不改原DAMPING=.002、CAP=sqrt(5)*.75。

原solve给定的非负ball_multiplier为μ；列每一方向：

- bounded coordinate `c=s*t/(s²+.002+μ)`；unconstrained ridge coordinate使用μ=0，仅描述，不生成动作。
- target energy `t²/192`、remaining coordinate `t−s*c`、linear residual平方贡献。
- ridge penalty `.002*c²/192`、合计ridge objective贡献、linear gain贡献。
- cap相对于unconstrained ridge的objective贡献，不把cap增量或谱大小当物理能力。

“截断”明确为**全部k=0..10**原完整解方向的retained-prefix描述表，未保留方向回填原target energy，
保留r⊥；乘子仍用原全10方向μ，不对每个k重优化、不选k、不执行任何截断动作。
原solver本身没有硬截断秩，不能据此称其拒绝了某阈值下的方向。
完整谱重构参数、singular、目标能量、谱objective与原saved参数直接Jp的差全部另栏，
基变号/重复奇异值子空间非唯一，跨行单个mode方向不可当成相同机制；只报告差异，不新增容差门。

直接原saved params的192维残差和penalty独立fsum重算，与saved original solve值并排及差值。
本工具不重新执行/替换原solver，不用这份诊断重新编译U。

## 原Q控制与同时间模型能力（不是实际收益）

全部22原Q槽连接：5 actualcp10、7 nominal、5whole、5segmented。
先用原精确float64 NumPy mean route重算并要求与原metric一致；结构或数值路由不匹配失败，不放宽。
再独立scalar math.fsum逐项计算192维Q，保存fsum−原NumPy归约差，不以末位舍入重新判原算法选择。
每个U列同一个真实输入下的 `Q(nominal zero)−Q(U)`，与 `Q(actualcp10)−Q(U)` 分开；
两者均仅是预测，不混入actualcp15，也不把drift误作控制收益。
同一个U的whole/segmented比较严格独立保存；五个角色各自编译U，不声称五者raw相同。

所有cp10相对同job同root cp5的R/nominalQ/10 singular差逐行报告；这是跨时间变化，
不是同时间基线。全部4实际角色对及8逻辑C比较另列**同cp10**R、谱及预测零相对改善。
目标残差和上下文均随角色变化；R/J差不能唯一归因于某个动力学机制，更不能由局部SVD保证真实收益。

## 真实即时代价与待测阶段C

原endpoint summary按state7/goal7独立fsum重算native7、jointXY、velocity、raw angle、wrapped angle；
原min(abs(delta_angle),2π−abs(delta_angle))可能负值的历史行为保留，wrapped仅诊断，成功门不换。
全部8比较列原D10(C)>D10(B)是否严格成立，ties不算通过。
`C_dual_control_conditions`逐逻辑槽输出对Z/P**同时**更差的合取；任何一边不成立就false，
缺C为null。不会只引用有利一边；这能明确区分“C双对照即时代价前提未通过”和未完成probe验证。

本版不消费未来阶段C，固定五端点各U/Z2共10实际probe槽待测，
actualcp15_U、actualcp15_Z2、G及最终反超均null。
不把模型R改善/预测Δ当G，不把一个即时代价分支成立写成协议已满足。
所有P1/P2/P3/C3/scientific/fresh/release/优势标记false；不签原门，不选winner。

## MAIN CLI（worker不执行生产）

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p3_capability_signal_summary_20261029_v1.py
M=experiments.spectral_reachability.agent_p3_capability_signal_summary_20261029_v1
S=/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_capability_signal_summary_20261029_v1_MAIN01.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m "$M" build \
 --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_capability_probe_cuda_20261029_v1_consume_MAIN01.json \
 --receipt-sha256 682a78366b10a60895c4530f2fc6cc401e6439b625f4ba0a8aa996dc2fed79a7 \
 --endpoint-summary /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_secant_endpoint_summary_20261029_v1_MAIN01.json \
 --endpoint-summary-sha256 6a6513ebfa68f331bbbecac139af84b0614446e3df739b3d00959a3119f743a0 \
 --output "$S"
SUMMARY_SHA=$(sha256sum "$S" | cut -d' ' -f1)
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m "$M" verify --summary "$S" --summary-sha256 "$SUMMARY_SHA"
```

单输入/source上限128MiB，输出16MiB，free floor200MiB，read/math/fsync整体180s；
新文件排他创建/fsync目录，前后核源与两输入SHA。verify完整重建所有输出字段，非只核status。
CLI只输出路径/SHA或简短verify状态。若最终wall失败，文件保留但不能签本次完整成功，不覆盖重试。

## 给Pauli的Stage C交接（消息工具不可用，不冒称已直发）

原Stage B `.consume`已有signals7/probes5/plans5，五个`.witness`文件由MAIN实际导出。
顺序固定r0C/r0Z/r0P/r1Z/r1P；每个`plan`包含actualcp10 task/endpoint_identity/source、
causal_raw10及bits、raw5/bits/SHA、signal/solve、whole/segmented完整context、Q、native_call_ids、
prepare/report/completion authority、bridge_authority、native_seal_sha256、witness_sha256。
Pauli应按其冻结阶段C合同使用真实witness，不把本summary或RAM当release authority。
本版不更改/重复其physical adapter，不调用生产消费。
MAIN已提供实际导出文件（以下SHA是MAIN交付记录，本worker未运行其witness/fullstrict）：
绝对目录 `/home/gbk/code/LeWorldModel/le-wm-main/results/`，文件名
`agent_p3_same_time_capability_probe_cuda_20261029_v1_witness{0..4}_MAIN01.json`。
对应index顺序SHA：

- 0：`6029b37c7a02543a58ef962c4369002939239d3f34bda34bced155204eb3124a`
- 1：`011d086c47cfd2ffd33fc16a895960d83e6cb0308c63168ecfd9bfe41abb8b21`
- 2：`5cd3933ffab7ed42b0124f26414bcba8de3c15075f6fe0d27815e5a4b8c3caa3`
- 3：`e1e2bf9124070256da21b064e2c1316045c9b3dc22190c665f44b9c9a3146422`
- 4：`ebf2b284e92c80905c2da561fc7b69f669d4cfd69bc5cd6f213ef34627ab7c55`

## RAM验收记录

测试使用独立构造的对角J/闭式ridge与最终schema形状，不读真实结果，不import producer数学。
覆盖全7/5/8/4分母、all11 rank-prefix、active ball/零singular、strict tie/subnormal/null、
goal/raw/history/whole/source/fullsnapshot/authority篡改、原native angle行为、
临时磁盘完整build→write→verify（真实source pins不patch）、整份结果及输入篡改、
外部SHA/重复key/nonfinite/path/排他创建/本地wall、cold import无torch/physics。
fixture仅临时文件，不是实际HOST证据；生产实际build/verify由MAIN运行。
首次回归38 passed /2 failed（2.18s）：一项fixture把分项除192的谱能量要求等于先求和再除192，
误把末位归约差当严格相等；另一项fixture两份权威共享Python引用，mutation同时改变两份输入。
仅修正RAM断言/fixture独立JSON序列化，不放宽运行时exact Q或任何科学门。
随后40 passed /2.26s；补齐全部22Q不同区域的值篡改（含同步修改plan/re-hash）、
重建raw全部hash后仍不匹配incoming token的拒绝、错误外部字节SHA测试后，
最终 **47 passed /2.94s**。临时磁盘整链通过，实际分析结果尚由MAIN生成。
三件套至此冻结；没有NN/physics/fullstrict或本版生产build/verify。
