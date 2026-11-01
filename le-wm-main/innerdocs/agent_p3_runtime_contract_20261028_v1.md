# P3/C3 同任务程序 runtime 合同 v1

独占三文件：experiments/spectral_reachability/agent_p3_runtime_contract_20261028_v1.py、tests/agent_p3_runtime_contract_20261028_v1_test.py、本文。只创建本stem结果；不改旧sealed源码、MAIN文档、共享总账或dirty改动，不派生下级，不训练/拟合/Q或normalizer修改，不调floor/删winner，不启动GPU/physics。experiment-code只用于固定输入、资源日志和失败变异测试，训练/强制非零效果模板不适用。

个人完成条件是可运行合同与RAM回归、真实saved输入接线、说明/SHA/命令/精确缺项交付，不是P1/P2/P3或研究goal完成。

## 原Proposal与已有实现复用

已读原附件§4.1–4.3/§5/C3：同一任务中冻结动力学决定progress/reconfiguration生成、组合与深度/horizon，而不是仅在根之间分预算。原附件路径及字节SHA进入prepare，非重写原方案。

1. p3_adaptive_runtime_20261027：已有任务内逐层beam与局部信号，不重复声称首次runtime。其未中心化有限响应span含漂移，horizon_gain是单步进展，不等于原提案Jacobian不可达残差。此处不复用该估计器。
2. d2p_adaptive_budget_v2：复用FrozenSignals/FrozenSchedule的严格类型、相同seal及阈值/count验证。工程常量沿RuntimePolicy原值：chi=.5、gain=.05、SVD relative cutoff=1e-6、max_depth3、base4、factor2、max_expansions144。不校准、不放宽，不将沿用阈值在新信号下称最优或科学通过。
3. p3_joint_horizon_goal_metric_scheduler：已有同根H2/H3 full-J/ridge/独立回放和多轴账，不冒称它没做horizon，也不把它等同完整程序类型/组合runtime。其固定Q/银行/预算合同不能直接套成本模块H1/H3。
4. d2p_kinodynamic_program_search与d2p_composition_witness_v2：复用严格full-context校验；保留“显式动作见证+逐边精确身份闭合”的原则。本核不把几何近邻或任意latent当边，不从图推测未执行结果。避免为了复用compose再增加第三次未记账回放；直接独立校验全path和两次回放。
5. 上轮agent_p3_resume_20261027_next只作真实已付资产与两个固定known roots输入来源，不能拿新winner重调候选流或minimum。

## 已真实saved核对的边界（不是新实验）

MAIN冷流report的实际SHA：`5bb52423c796713899a4947f9c0167ad9dfa32d1b336b7abf858d17dd45f8462`；strictconsume实际SHA：`0403affb31fa137308502f5875a05960222c60540b9cf28b91012984310cd65c`。两文件已只读核SHA与receipt字段：512 predict/action_encoder/predictor/pred_proj attempted=returned；wall42.92500251997262s；两根各臂winner12/1，全在共同前15，D−U/S/N均[0,0]。这些旧负结果及阈值/floor/赢家不改变，零差不证明原C3已测全或理论失败。MAIN的terminal0是旧作业退出状态，不在本位伪造新GPU退出记录。

prepare会重建旧prepare闭包，原样抽取两根context/goal/projection/normalizer与模型身份，不读取downstream physical/RGB outcome、不把旧cost/weight/winner交给新调度器。此前520panel、编码和anchor账全保留，另绑定新cold512的完整report/strictconsume引用；复用不等于历史成本为零。PushT原生位置成功同时含agent+object，本模块不由object-only投影判断完整失败，不改原生7维标签或阈值。

## 真正的任务内因果流程

`run_task(task, backend, complete_caps, mode, permit_ram=False)` 是可执行的callback runtime，不只是静态gate清单。task只允许binding、root_context、goal、projection、action_mean/std；严格拒绝reward/physical_state/success/teacher rank等附加字段。goal/Q/normalizer/root context及action domain逐项绑定SHA，任务/模型资产不变；每个子节点另绑定完整parent context，不能拿同root旧信号或近邻替代。

每个实际访问节点先发H1与H3两个compile_finite_horizon_J请求，模型/目标/度量/动作域固定。nominal和AD forward都要付费；完整J为[d,10H]，不是读取外部chi分数。输入响应必须携带精确binding/parent/hash/horizon/producer SHA，拒绝额外outcome字段。producer SHA只绑定声明来源，不自行证明真实模型/独立性。

两次实际NumPy SVD均计attempt-return，计算K=projection@J。用**同一个短时域nominal目标残差**比较两个响应空间，避免把nominal漂移误作horizon子空间收益。保留两个空间的完整谱/rank、同参考残差chi和signed_gain=chi_short−chi_long（可负）；调度只取max(0,signed_gain)，不改原始诊断。零能量chi/gain为0是已知算术情形，不是缺失值补零。沿用SVD截断不是安全/可达证明。

adaptive：高chi扩展reconfiguration，否则progress；gain达沿用阈值则选H3，否则H1；高chi或gain达阈值允许最多3个程序chunk深度，否则至1。每次重新编译子节点信号，允许任务内变化，不是一次root路由。FrozenSignals的seal同时包含task binding和**当前parent context**，同任务旧父信号也被拒绝。

程序候选数由沿用schedule定为4或8，节点队列在任何该节点candidate score之前固定。root请求类型是progress/reconfiguration；非root类型是composition，并显式携带extension_type与已验证prefix动作/边。backend只生成一个后缀edge，不能替换prefix、改变task/model或返回无动作latent waypoint。高chi仅请求reconfiguration生成器，不证明动作真的改变未来能力；原Proposal的c_stage+beta Vreach/阻尼与动作chart必须由MAIN另封存真实compiler spec，当前为null，不能默认0或冒充已有证书。

返回的raw必须是完整float32[-1,1]、5H×2动作；逐包以原float32 subtract/divide重建action token，校验所有仍在窗口的latent顺序、两段between-action history与父子身份。组合仅连接已返回且闭合的edge，保存完整拼接动作和每个预测boundary。它是模型预测见证，不是物理可达见证。

候选按固定model-Q、depth、lexicographic index path稳定排序；未按立即改善过滤reconfiguration，best跨全部已访问深度保留。beam上限adaptive8/fixed4是显式近似，不声称最优。程序chunk深度≤3，局部horizon∈{1,3}，累计动作最多9packet/45步，不能把chunk深度3误报成实际预测总horizon3。winner必须从root完整whole与独立segmented回放，所有boundary context精确一致；不只比terminal。完整候选payload/diagnostic_best即使失败也保留；不完整时model_plan=None。

fixed对照在自身同任务节点也支付H1/H3信号计算，但固定progress/H1/4候选/3chunk深度，不使用chi/gain路由。两臂共用完整caps profile，不重置phase预算、不填充调用凑同数。相同caps不是相同实际花费，当前无质量—成本结论，也不是零谱成本N基线。

## 资源与失败分母

Ledger覆盖signal/generate/metric/schedule/verification全phase，不以扩展次数冒充predict samples。每次真实adapter entry必须回调，计predict API、batch samples、context tokens、AE/predictor/pred_proj calls/rows、encoder/projector、Jacobian、VJP方向、grad API、raw callback、model/array SVD、solve、compiler/replay/metric/schedule。每个轴均必须提供cap，缺项/null/bool拒绝；本位不猜真实compiler预算，实际prepare.resource_caps=null并阻断。RAM的500000只是测试计数上限，绝不写成真实作业预算。

debit在delegate前，异常不退款；cap拒绝记录blocked_before_dispatch且不调用delegate。attempt/return均保留scope/shape/方向数/时间，audit_ledger逐项重数，reject伪samples/总数；失败的SVD等仍计attempt而不completed。signal还要求实际nominal+AD forward、2J、至少2d VJP方向及grad/raw-hook记录，不能用未计导数的数组冒充免费信号。实际backward engine calls、CUDA ns、FLOPs未知保持null；观测到raw hook次数不等于engine次数。known no-physics为0，禁止physics operation入口。

HOST_CUDA adapter必须提供predispatch durable event_sink；RAM可用内存sink。journal失败保留第一错误/后续journal错误，不调用尚未派发delegate；已debit不退款，不冒称已执行成功。hook实际传入shape是外部可信输入，本核不单凭metadata证明GPU tensor或无hidden call；MAIN还必须接入真实冻结模型前后hash/eval检查、native完整数组journal和独立consumer/cleanup。sink的fsync/模型冻结来源真实性由审读后的adapter负责，当前没有虚构它们。

完整成功时predict API必须与三个native模块call数一致，实际samples/context tokens/flatten rows逐轴一致，全部paid attempts已return；被backend吞掉的失败不能释放plan。whole/segmented回放付费不退款，无预留则budget failure会failclosed；真实profile必须事前保留足够回放预算，不能事后追加或调宽。

每task/arm始终保留144个扩展cap槽位和2个回放槽位：not_requested/queued/attempted/returned/failed/blocked_after_failure明确区分。144是上限，不称每个槽位都应执行；节点队列放不下时整节点cap-stop，保留原请求和诊断，不缩候选凑预算、不宣称完整成功。run_panel先固定所有task×fixed/adaptive，setup失败/执行失败也保留分母，不按成功过滤、不重试。未知outcome不填0。wall_ns是真实runtime，operation wall为nested inclusive，不能求和冒充total/FLOPs；模型加载/源重建/历史成本不能被本核局部timer吞成免费。

## 准备/核验与精确缺项

CLI只有prepare/verify/queue，无GPU或physics开关。prepare封存自身三文件、复用源文件、原proposal、前序完整闭包、MAIN真实cold引用和两task数据。queue在缺真实局部信号时只列根的H1/H3 compile请求；program_queue=null，不给默认progress、不把旧H2 J填作H1/H3。完整source/task canonical重建与外部prepare SHA不一致即拒绝。输出仅本stem独占新文件、2MiB metadata cap/200MiB floor、exclusive/fsync，不覆盖旧产物。

尚缺的执行前置：

- MAIN审读后的同节点H1/H3 signal、progress、真正reconfiguration生成器与whole/segmented replay adapter；H2旧consumer/跨root原型不能冒充这个adapter。
- 实际compiler objective/action chart参数、完整冻结模型/source guard、native全数组即时保存和独立saved consumer/cleanup。声明的SHA或backend.kind不是认证。
- 事前封存的完整多轴operation caps、wall/memory/output profile与独立回放reserve；未运行前不虚构future report SHA。
- 独立P1机制准入、适用P2组合增量、认证fresh有序root/task/goal及完整历史排重/acquisition账。当前两known roots不能追认fresh。
- 真实HOST CUDA任务内trace、相同预算口径的fixed/噪声等对照和真实任务后果；当前均未执行，原C3/实验族未关闭。

任何RAM通过、已封存prepare或旧跨root 0差异都不改变P1/P2/P3/freshness/full_C3/physical_release=false。没有新科学结论。

## 命令（cwd=/home/gbk/code/LeWorldModel/le-wm-main）

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_p3_runtime_contract_20261028_v1_test.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_runtime_contract_20261028_v1 prepare --output results/agent_p3_runtime_contract_20261028_v1_prepare01.json
# PREPARE_SHA取上条实际输出；已经存在的文件不覆盖。
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_runtime_contract_20261028_v1 verify --prepared results/agent_p3_runtime_contract_20261028_v1_prepare01.json --sha256 "$PREPARE_SHA"
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_runtime_contract_20261028_v1 queue --prepared results/agent_p3_runtime_contract_20261028_v1_prepare01.json --sha256 "$PREPARE_SHA"
```

MAIN后续API接线：用prepare.tasks固定全体名单，为每task/mode创建绑定相同冻结资产的reviewed backend（signal(request,ledger)、generate(request,ledger)、replay(request,ledger)、kind=HOST_CUDA、durable event_sink），提供同一事前完整caps，调用run_panel。当前backend/profile/spec均null，执行不授权，不提供会默默填补依赖的示例模型或伪研究资产。

首轮26项RAM通过0.81s；补强父节点signal seal、journal与多轴审计后30项通过0.96s。最终回归、真实saved prepare/verify和冻结SHA另列；这些不是神经/物理证据。
