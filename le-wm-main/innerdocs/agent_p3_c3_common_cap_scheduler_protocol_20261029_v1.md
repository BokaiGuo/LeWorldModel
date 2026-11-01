# C3下一可执行切片：不全枚举的同cap原生调度对照（提案，待MAIN批准）

2026-10-29。只读既有来源、写此NEW独占doc、运行文末纯RAM整数公式；不实现runner，不运行NN/physics/prepare/fullstrict，不开子代，不修改secant四件套或其他封存结果。experiment-design技能只用于可识别对照设计，不采用调参/训练步骤。本协议不授权生产。

## 1. 现状：选择确实不同，但尚无动态配额有效性的证据

本次只读取结构、决策、资源和状态，不读取reward/success/endpoint收益决定新配置。结果路径均在`results/`，源码在`experiments/spectral_reachability/`。

|只读权威|实际SHA256|
|---|---|
|agent_p3_horizon_activation_summary_20261029_v1_MAIN01.json|c2e933932b2b6a924f565295b72e676849470c7ac21f98dbb21a1086acf9e2ab|
|agent_p3_parent_tree_summary_20261029_v1_MAIN01.json|00ebd93b13353dfc54d45263575bf0d16a642d8985e773281aea265b94d81467|
|agent_p3_parent_native_cuda_20261029_v1_consume_MAIN01.json|622d1cdaf8d1ed69c4e78f8d1f5c7c6114285db6121f0021fbbaf6573a8a76ad|
|innerdocs/agent_p3_resource_matched_followup_20261029_v1.md|03aa37fcdd878ab854a0495f494652c80a917a45544b56722f5f62e3cb9cd194|

原tree receipt真实schema `.consume`、status=`saved_four_slot_engineering_verified`；绑定prepare `deb87295f3984929f4734ece0b73770220ec59ac3b621a97e85e2bade60e3599`、report `75f4c162aaa75223dfd8523381e7614c8a35637afd52278df8a80d6affa34e40`。仅读取，不再次执行producer/strict；继承权威不等于本doc独立认证了原NN。

shared-panel summary固定24节点：chosen及dispatched均H1=12/H3=12，refused/not_reached均0。每个fixed臂三个节点全H1，每个adaptive臂全H3；每个选定首包均返回5步。**臂间有路由差异，臂内未展示horizon切换**。源码`shared_panel.plan_node`先完整生成并支付两个4候选面板、whole/segmented和独立winner，再读policy选择执行首5。每完整episode132predict；所以这是共享付费的horizon使用消融，不是选择H之后节省/重新分配面板计算，也不执行policy的width/depth/type。

原parent tree不是两horizon候选全枚举：每parent付H1/H3信号，`choose`后只生成所选H的完整队列。真实结果的结构如下（不是物理效果）：

|槽|实际生成H|调度parent/完整progress队列|生成宽度/beam|深度/状态|predict/J|
|---|---|---|---|---|---|
|root0 fixed|全H1|9/9；层数1+4+4|每队列4，beam4|3，plan available|114/18|
|root0 adaptive|已生成H3；另有H3 R请求|2/1|progress4；child R8拒绝|1，refused，无plan|28/4|
|root1 fixed|全H1|9/9；1+4+4|每队列4，beam4|3，plan available|114/18|
|root1 adaptive|全H3|13/13；1+4+8|每队列4，beam8|3，plan available|278/26|

四槽都没有原child-policy主动停止，也没有cap触发；adaptive两槽内也未出现H1/H3切换。root1的13不是13个width8队列：它是全width4、beam8保留更多parent。root0原8请求是真拒绝，不能算R已执行、width8激活或budget主动优化。fixed/adaptive原caps114/562不相同；该receipt不能提供同cap对照。

结论：有真实parent J和原始队列路由，不是所有候选穷举；但目前horizon表现可由固定H3解释，宽8未执行、深度自适应未激活，也没有稀缺预算的收益证据。不能把“choose存在”或“请求不同”写成C3成立；未来重跑同样常量选择也必须报告策略退化，不强求出现切换。

## 2. 唯一推荐的下一实验：common-vector-cap full-typed tree，而非再加搜索bank

下一代码slice复用原R-aware tree的`Session/run_tree/evaluate_R/validate_tree`，新增own-schema common-cap调度入口，不改旧runner/profile。两known原root tasks保持原始完整context；**不是actual-cp5，不偷换起点**。原goal/Q/normalizer/model/precision、progress原fraction bank、R原spectral8 bank、R严格资格均冻结；本轮secant/goalbank不接入。无物理release，不把model children冒充真实观测反馈。

固定三个scheduler全保留：

1. 原fixedH1：progress、width4、beam4、depth3。
2. 新名staticH3-progress：只作预定义静态反事实，H3、width4、beam4、depth3；原fixed代码不改，不能冒称这是原policy。它补足“adaptive实际上只是固定长horizon”的替代解释。
3. 原adaptive：每个真实model-parent重新付信号并调用完整原`core.choose`；原阈值、4/8请求、beam8、depth1/3、progress/R及composition链保持，不把R映射progress，不只消费horizon字段。

两个常量horizon都来自原支持集{1,3}，不是按已见收益选择H3当唯一baseline。三个scheduler均付同一类型的H1/H3信号，不生成未选horizon的候选，不为填账shadow replay；信号属于这份受控对照，不能称最廉价自然baseline。所有eligible R edges进入原(modelQ,depth,indices)beam，no_witness停该branch，无fallback。

本提案保留两个**由调用图定义而非效果选取**的budget层：K=1和K=17，见第3节。固定2roots×3scheduler×2caps=12槽。K=1检查真正预算稀缺时的拒绝/partial；K=17是结构饱和上界对照，用于区分“多给资源才可运行”与调度本身。若只先实施一个工程切片，先做共同cap的完整RAM callback/strict；不擅自只运行结果好的一层。两个预算层的质量不混算成一个提升。

严格措辞仅为**同27轴cap envelope**，不是同actual spend、同wall、同FLOPs。原完整4/8队列原子派发、144请求及最大depth3不变；不为耗满cap补调用，也不把未用额度算执行。cap不足按原语义保留partial/diagnostic_best，**不放行model_plan**。低cap可能所有臂无完整plan，这是有效工程分母，不是可比较的0成功，也不能因而事后调大cap。完整plan与partial不可混报best-Q。

这个设计检验完整policy bundle在共同机会预算下的选择/完成情况，不单独识别width、depth、R哪一项造成差异。若两常量与adaptive的真实队列完全相同，结论必须是本cohort无调度激活；若高cap不bind，也不能从高cap单独证明有效配额优化。后续独立因子消融应另立协议，不能在本批按表现选择子策略。

## 3. RAM预算公式：逐轴上界，不是免费R或预测FLOPs

令P(n)为原predict_axes(n)：predict API/sample、AE/predictor/pred_proj调用及sample轴=n，context/token/投影行轴=3n，其余0。以下是成功完整调用图；失败只记真实attempt/return前缀。

- 每parent signal+schedule `S=P(8)+{J=2,signal_compile=2,VJP/grad/rawback各384,arraySVD=2,schedule=1}`。包含独立nominal和AD forward，不能漏AD原始forward。
- progress队列 `G(h,w)=P(hw)+{programcompile=w,metric=w,solve=1,arraySVD=1}`，h∈{1,3}、w∈{4,8}。完整parent只生成一次bank，候选路径按原已生成链保存，最后whole/segmented另付。
- R队列 `T(h)=bank_spectral8+9×item(h)`：8proposal+独立paid-zero；每item为(h+3)predict、1post-J、192VJP/grad/rawback、2replay、1solve/arraySVD、1programcompile、2metric。bank另1programcompile/1arraySVD。故T(3)=P(54)+{J=9,signal_compile=9,VJP/grad/rawback各1728,replay=18,solve=9,arraySVD=10,programcompile=10,metric=18}`。suffix仅诊断不拼edge。
- 最长3个H3 edge=9token，最终whole+segmented reserve `V=P(18)+{replay=2}`，不是物理45步。真实短链只付实际长度，不为了18填充。
- 逐分量有T(3)≥G(h,w)、T(3)≥T(1)。最大17个paid parent来自1+8+8，不是1+8+64（原beam最多8）；全部外部请求最多136≤144，R zero不占第9个候选identity，但收费。

先定义 `B0(K)=K*(S+T(3))+V`，再将schedule_decisions上限统一设17，得到所有方法共同预算B(K)。这是显式新cap，不把便宜的policy调用数K当成额外parent硬限；K=1时便宜progress可以跑多个parent，其余各轴仍逐项受限。这不是允许跳过原子预检：每次仍核真实queue最低量+该链最终验证reserve；下一parent信号派发前也必须预检全pair+schedule。原depth/144依然限制。

|轴|K=1，同cap稀缺层|K=17，同cap饱和层|
|---|---:|---:|
|predict API/sample；AE/predictor/pred_proj相关单行轴|80|1072|
|predict/AE/predictor context token；pred_proj rows|240|3216|
|J / signal_compile|11|187|
|VJP / grad / raw_backward各|2112|35904|
|arraySVD / solve / programcompile|12 / 9 / 10|204 / 153 / 170|
|replay / metric / schedule|20 / 18 / 17|308 / 306 / 17|
|encoder/projector及sample、modelSVD|0|0|

上述是全typed安全分量上界，不是原114/562的重标。旧R-aware默认replay cap=2，无法容纳单R的18+最终2；**不能沿旧cap拒绝后当R负结果**。新schema显式绑定B(K)，旧caps不改。AD-primal和J元素另按实际每个signal的H重算：parent pair有4个AD-primal predict及7680 J元素，每R post-H1各1/1920；它们不和inclusive predict重复相加。FLOPs、scalar backward、算子耗时仍unknown。

下面代码只验证整数预算/支配关系；不import任何producer、NumPy、torch、weights或saved NN artifact。

```python
from itertools import product
fields = ('predict', 'J', 'VJP', 'svd', 'solve', 'compile', 'replay', 'metric', 'schedule')
def add(*xs): return tuple(sum(z) for z in zip(*xs))
S = (8, 2, 384, 2, 0, 0, 0, 0, 1)
def G(h, w): return (h*w, 0, 0, 1, 1, w, 0, w, 0)
def T(h): return (9*(h+3), 9, 1728, 10, 9, 10, 18, 18, 0)
V = (18, 0, 0, 0, 0, 0, 2, 0, 0)
def B(k):
    v = list(add(tuple(k*x for x in add(S, T(3))), V))
    v[fields.index('schedule')] = 17
    return tuple(v)
assert B(1) == (80, 11, 2112, 12, 9, 10, 20, 18, 17)
assert B(17) == (1072, 187, 35904, 204, 153, 170, 308, 306, 17)
for h, w in product((1, 3), (4, 8)):
    assert all(a <= b for a, b in zip(G(h, w), T(3)))
assert all(a <= b for a, b in zip(T(1), T(3)))
assert 1+8+8 == 17 and 17*8 == 136 < 144
assert 17*18+2 == B(17)[fields.index('replay')]
print('RAM static budget identities: PASS; no execution certification')
```

## 4. 精确复用与未闭合前置（不能靠本doc签过）

源码本次只读快照：core `72f076106fb8f6c2aa73386c06fecb912f033e81a5108aa1f36885bd0342c2bc`；parent adapter `150a1c4189a68f9b7c71479e5aa6980cea490fdeed0d5524b7bfad0cc78c2558`；R orchestrator `da00e09c89ed8b5be5399952c2ec328a98e036db08f764b61342d69a18f5e6a6`；spectral bank `e254f08db87b1e8b6985b6afd6ba3f3ed8f98e8f1af0d87f9edcd14021f2786d`；RAM evaluator `71e4e178eb180a0d9b740697c98538d6f9afd0ba50d35efb03aa23f91378e04d`。它们的读取不代表新的运行认证。

最小实现缺口是NEW common-cap producer/consumer，而非再写bank：复用`core.choose/request/checked_path/model_cost`、原完整parent信号和source capsule、R orchestrator的`evaluate_R`及全eligible/modelQbeam、`reserve/Session`和saved重建。新增staticH3自己的明确决策ABI、三scheduler共用caps、27轴账及source闭包。不能调用旧run_tree后在receipt换mode或source identity；更不能用monkeypatch替换choose。source/installed class/model/precision前后及原3latent/2token/raw链必须照实保存。

接线前需MAIN批准本预算与staticH3新variant，并冻结当前R谱bank及资格/全部eligible语义的完整source绑定；不得将近期goalbank/secant成功与否作为换bank依据。R真实单parent诊断不自动认证完整树的child composition、全eligible beam、最终whole/segmented；这些生产闭包尚需新runner的RAM+MAIN实跑strict。旧receipt已证明progress child，不证明R child，更不证明P1。

RAM验收必须覆盖：三policy不读取candidate cost来选horizon；同cap真实选择分歧与常量退化都可保存；R8全队列/zero实付；no_witness只停branch；depth1真实child stop；宽8和beam8不混同；边界cap只容纳signal而容不下整队列；不能先付半队列；failure不退款；完整12槽/144identity分母；全context/source/事件/数组篡改拒绝。低cap不放行diagnostic_best、高cap不强制补调用。

资源之外还缺P1原机制准入、真实控制任务/反馈/endpoint指标合同、独立fresh证书等；这些不是NN接口成功就成立。第一阶段仅model planning机制/资源证据，三policy plan长度可不同，modelQ不是同物理终点收益。不得以不同预测终点的Q直接签控制提升。后续真正物理C3须另固定相同观测频率、FIRST5执行和总20步等合同（晚H3明确lookahead），MAIN审读后另行授权；本提案不执行它、不签P3/full_C3。

工程wall/disk/CUDA memory及总12槽新output allowance还需MAIN在实施前固定，不能因为本doc给出NN静态调用cap就默认3600s或2GiB足够。等27轴cap本身不保证equalFLOPs；即使H计数/shape也相同，仍须报告实际wall/memory与未知算子成本，不伪造同算力。

交付只此doc与文内RAM公式。无新生产CLI、无运行授权、无NN结果；不是又一份可冒充可执行runner的metadata包装。
