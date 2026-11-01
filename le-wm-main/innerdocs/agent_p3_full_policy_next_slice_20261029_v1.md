# P3 horizon-only之后的最小真实policy代码slice

2026-10-29，仅设计新文档；不改shared_panel三件套，不读物理outcome选择配置，不建runner、不运行prepare/模型/物理/full strict。MAIN已报告shared_panel联合104tests及prepare/verify通过、生产运行中；这里不等待或利用其结果。使用experiment-design技能明确分阶段实现与预算，不采用按结果调参步骤。

## 1. 精确缺口，而非再跑panel

shared_panel只使用`core.choose`的horizon，完整保留但不执行原candidates/depth/type；两个bank和双付费面板不是完整原policy。原`p3_adaptive_runtime_20261027.RuntimePolicy`固定base_beam4、expansion_factor2、max_depth3、max_expansions144、chi=.5、gain=.05。`agent_p3_runtime_contract_20261028_v1.choose`实际为：

|决策|fixed|adaptive|
|---|---|---|
|root extension type|progress|chi>=.5为reconfiguration，否则progress|
|horizon|H1|按原gain阈值选H1/H3|
|每parent candidates|4|chi>=.5为8，否则4|
|depth limit|3|chi>=.5或gain>=.05为3，否则1|
|非root program_type|composition|composition；extension_type仍由该parent信号选择|

因此真实原policy下“adaptive width8”与reconfiguration关联；实现了progress宽度参数8的RAM测试，不等于原policy宽度8真实运行。depth不是每次固定三层：必须在各parent用新信号重算原choose，以该parent的continue_expanding为准；非root composition标签不能据此把其extension_type=reconfiguration改成progress。

## 2. 建议下一次交付：ParentNativeAdapter + 真progress组合队列

下一独占代码slice不扩当前物理runner，先做一个可调用的HOST CUDA程序搜索backend和自己的prepare/run/saved-consume。固定原两known tasks，不碰reserved根；保持原goal192/identityQ/model/action normalizer及工程阈值。不复做RGB编码或同反馈物理，输入可为合法绑定的原真实root context，结果只是model program，不自动给物理release。

复用`core.run_task`的队列算法（或调用原函数并提供严格适配backend）：每个parent完整fresh H1/H3→原choose→一次原子分配全部4或8请求→真实generate→`checked_path`连接→原(modelQ,depth,indices)排序→原beam4/8→深度最多3、请求最多144→赢家whole/segmented独立重放。不能先只做4再把未做4记成完成；扩展cap若装不下整parent队列，保留`expansion_cap_before_node_queue`且不释放最终program。原`run_task`此状态下只有diagnostic_best，不等完整model_plan。

具体要写的最小接口为：

1. `bind_parent(root_task,parent_context,prefix_edges)`：保留不变root任务authority及单独parent-context SHA；严格校验两token/三latent和原raw动作链。不得只换last latent或拿root J冒充child J。
2. `signal(request,ledger)`：在该完整parent context创建实例级`NativeProgram/NativeSignal`，实际编译H1/H3。现`NativeSignal`接受其构造task的root_context；直接把原root实例用于child会失败，不能删检查。若向`core.run_task`适配，需要显式保存外层root-binding request与内层parent-task/request/response的双向映射和原始native response；只对齐外层接口字段，不伪造旧receipt。也可新orchestrator直接使用这套双identity schema，复用core的choose/request/checked_path，不改其函数。
3. `generate(request,ledger)`：progress用当前parent已付对应H的J、原bounded ridge及既有`live.generate(task,response,count,ledger)`的`i/(count-1)`原fraction规则；支持4/8，不将8个index挤成4。当前shared_panel两bank是独立干预，不冒充原full-policy候选规则；下次优先用原规则，不叠加bank实验。每parent可显式生成一次完整bank并复用已持久bank到该parent各请求，按实际SVD/solve次数收费，不每个candidate暗中重复求解或把cache工作记成新调用。
4. `replay(request,ledger)`：将原`NativeProgram.replay(parent,raw,segmented,ledger)`适配成原`verify_program` ABI。逐edge保留input/output完整context、原raw、path与source；最终从真正root重放拼接的全部raw，whole/segmented每个边界精确相等，不能只比末192。

这个slice能真正执行原progress分支的宽度、depth以及progress→progress的composition，并真实拒绝reconfiguration。它**不是完整full-policy实现**，但比再跑horizon-only panel新增真实child闭包。必须完整保留全部4个task×method槽和144 expansion槽；遇到原policy要求R，不筛root、不fallback，记录明确unsupported及未请求槽。不得把固定三层的RAM树或旧P2组合结果当新policy树。

## 3. 可直接复用的函数与不能直接搬的实现

|现有位置|具体复用/边界|
|---|---|
|agent_p3_runtime_contract_20261028_v1.py|`policy/choose/request/checked_path/model_cost/Ledger/audit_ledger`；`run_task`队列、原子cap、beam与final replay逻辑；后者只要求接口，不证明真实child模型|
|agent_p3_local_signal_cuda_20261028_v1.py|`NativeSignal.signal/spectra`及native IO/VJP持久化；`runtime_signal`只对其显式rebased task使用；不能把`decide_root`的root-only队列改标child|
|agent_p3_native_program_20261028_v1.py|`generate`原ridge/除数/float32一次cast；`NativeProgram.replay`完整3latent/2tokens的原AE/predict/predictor/pred_proj链。旧`panel()`仅root4候选，不直接充当新树|
|agent_p3_fullcommit_live_20261028_v1.py|`generate(...,count,...)`原4/8 fraction bank，`NativeJournal`；不能复用其remaining-fit/commitment dispatcher来截断树|
|d2p_composition_witness_v2.py|`compose`连接原始动作、完整上下文与逐edge replay；需用其精确provenance/domain/edge字段写新适配，不伪造为旧P2receipt|
|agent_p3_shared_panel_20261029_v1.py|只参考已审MATH/TF32/autocast、sources/model before-after、排他Store与saved验证经验；不重新生产、不把其receipt重标fullpolicy|

原P2`agent_p2_contextual_composition_runtime...expanded_bank`是80个两包logit/tanh bank和四槽种子协议，不是原P3 4/8个raw bounded-ridge候选，不能因“有组合源码”直接替换。旧`preflight_pusht_d2p_p1_reconfiguration.compile_reconfiguration`确有“先真实预测prefix、再算post-prefix reachability、比较零控制”的数学结构，但使用随机sketch、logit/tanh参数与旧image/history ABI；不是当前identity192/raw-J的即插即用R compiler。

## 4. 真reconfiguration的最小必要新增，而非给progress换标签

width8生产完整执行必须有独立R generator。它要为原choose给出的8请求生成**实际raw prefix及完整预测context**，对每个prefix之后的同goal/Q后缀响应实际付费，保存零控制参照、immediate原Q、post-prefix nominal/J、原ridge reach值及全部candidate。R的目标是改变后续可达性，不能只用同一个当前节点progress解的8个fraction并按即时Q排序，就宣称R已实现。允许没有非贪心witness；不能事后筛掉这种根。

具体尚缺的权威是：与原P3 raw域/identity192一致的**R proposal bank的逐位来源与8个索引映射、prefix/suffix H的固定定义、post-prefix reach评分/选择规则及其源码版本**。原choose只决定类型/宽度/深度/horizon，不定义这些生成细节；旧preflight的sketch/tanh不得默默成为默认。MAIN下一次需在结果前选择并封存这些generator规则（不是调chi/gain或签科学approval）。可以提取旧preflight的“零控制/immediate/reach并列诊断”形式，但不得默认其旧数值协议合法。

R模块就绪后，将它接同一`generate` dispatch，保持`request.extension_type`原样；composition只是前缀边与该extension的真实串接，可能含progress或R边，不是第三种假action生成器。原`core.run_task`仍用原Q选beam/best；若R生成器另有reach排序，必须明确它在**生成子程序内部**的作用，不能偷偷改全树beam/最终modelQ目标。若生成器需要更多内部proposals，其全部预测/AD也计费，不能声称一个原candidate请求只有一次forward。没有R adapter时停在slice2的partial真实实现，不包装成fullpolicy/P1通过。

## 5. 事前预算：先有树的实际账，不套132或等FLOPs

以下是**单task×method、单次root程序搜索**的完整上界，不是每个反馈边界共享panel的费用；不强制跑满。继承depth3/beam4或8：fixed最多9个被调度parent（1+4+4）、36 edges；adaptive最多17 parents（1+8+8）、136 edges，均小于144原扩展cap。每parent均付H1/H3 pair：8predict（4nominal+4AD primal）、2J、384VJP/grad/raw callback、2signal SVD。

|progress队列预算轴|fixed最坏完整|adaptive含R尚未计R内部工作的树外壳上界|
|---|---:|---:|
|调度parent / edge请求|9 / 36|17 / 136|
|signal predict|72|136|
|edge生成轨迹predict（每edge一条完整path）|36（固定H1）|≤408（每edge≤H3）|
|最终whole+segmented predict|≤6（最多3tokens）|≤18（最多9tokens）|
|inclusive predict API/samples batch1|≤114|≤562，**不是完整R预算**|
|J / VJP / grad / raw callback|18 / 各3456|34 / 各6528|
|J元素（各pair192×10+192×30）|69120|130560|
|program compile / metric|≤36 / ≤36|≤136 / ≤136|
|solve（一次bank/parent缓存实现）|≤9|progress部分≤17；R内部另计|
|array SVD（诊断+progress solve）|≤27|progress外壳≤51；R内部另计|

`run_task`要求每edge至少H次已付预测；本预算选最小一条真实生成path+最终两次完整replay，不额外伪称shared_panel逐候选双重重放。若新增每edge独立segmented核验，额外fixed≤36/adaptive≤408predict必须加cap，不能混用原114/562。AE/predictor/pred_proj逐实际predict调用各收费；predict上下文token和pred_proj行=3×实际samples。0新encoder/physics仅限输入已有认证context的这一代码slice。

为使后续R预算可落地：若封存规则恰好是“每个edge proposal额外一对post-prefix H1/H3 J”，在最多136 proposals下增加≤1088predict、272J、52224VJP/grad/raw callback、1044480J元素；连树外壳上界为≤1650predict、306J、各58752VJP/grad/raw callback、1175040J元素。这里尚未含R内部proposal生成超出一条edge轨迹的工作，不可把1650当任意R算法通用cap。精确同context缓存须绑定source/task/raw/path，实际少做就少计，不为凑数重复调用；未计量scalar_backward/总FLOPs仍unknown，不用VJP数代替。

后续MAIN按选定slice封存wall/output/CUDA memory cap，不从当前panel速度反推已认证上限；slice2可提议3600s/1GiB含128MiBreserve、200MiBfloor、CUDA allocated3GiB/reserved4GiB，属事前安全cap不是性能测量。未绑定R内部计费规则前不授权其生产run。任务固定4个task×method槽，不能为匹配cap删根或少跑adaptive。所有attempt失败不退费，超cap保存partial而非按已生成best偷偷放行。

## 6. 下一代码slice的明确验收与边界

验收不是泛泛审计：RAM构造根chi低/gain高→progress H3 depth3，再在一个真实预测child使gain低→原depth1停止，验证没有强制三层；构造chi高→原8请求/R明确unsupported，不能只执行4；逐条验证parent完整context及真实归一化action history、每个J来源、原beam tie-break、144 cap整parent拒绝、whole/segmented全raw链、失败attempt不退费及全槽null。随后由MAIN审读源，单独HOST CUDA run/strict证明真实child计算与程序闭包；worker不执行。

搜索depth3最长9tokens=45raw，不等同20步物理commitment；此slice不接物理，不暗改late-H3 FIRST5执行合同。未来若将树放入真实MPC，每个实际节点重做本树并只释放FIRST5，需要新的完整运行计量；不能拿一次root树冒充每个真实child都反馈。P1机制、P2组合增量与fresh独立准入仍独立；工程实现可以保存known2诊断，但不能凭实现/测试/优化后的modelQ授予P1、full C3或物理收益。

本次仅新增本文。未运行测试（无新代码）或生产；不重跑当前panel。源码事实已只读定位，尚未新增ParentNativeAdapter/R compiler；最小下一实施是第2节，R必要生成协议缺项按第4节明确保留。
