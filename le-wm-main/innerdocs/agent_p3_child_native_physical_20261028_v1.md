# P3 child native physical bridge：先协议，未授权执行

拟独占新stem `agent_p3_child_native_physical_20261028_v1`。本轮仅新增本文；尚未实现source/tests、未prepare、未run、未启动任何环境或模型。后续源码/测试必须MAIN明确授权后另行交付审读。不得修改child compiler、signal v2、旧physical/saved consumer、封存产物或MAIN文档；不得接触reserved roots。

## 1. 已知终态与尚缺前置

MAIN已报告compiler HOST CUDA MAIN01 exit0：report SHA `885f69f6de5ce9b78b8f3acdc84f14ef8891ba3ecd79912fc9af4e1920fc7ffb`，657predict/41compile/41solve/41SVD/369replay/164metric，全41/5/2。报告精确绝对路径需MAIN提供或由其输入manifest明确指定；不能仅凭MAIN01名称猜路径。本文引用MAIN终态，不冒称worker独立消费。

compiler完整strict当前仍在MAIN运行，尚无已知持久回执SHA；新桥不得预造或把CUDA exit0当strict通过。未来prepare必须完整调用冻结compiler.consume，并以实际外部SHA核完整持久strict回执与重新消费返回逐字段一致，再复核其全部prepare/report/输入/source闭包。若重核失败，不进入环境构造。上游signal完整回执已知，不可代替compiler回执。

另外必须取得原P4真实physical轨迹的完整native/source/budget权威consumer及持久回执，补充现有P4 rolling视觉seam consumer。后者明确 `full_native_or_budget_reaudit=False`，**单靠视觉seam及commands相同不能认证physical起点**。若尚无可调用完整消费者，则新桥保留blocked_missing_native_source_authority，MAIN先指定完整权威接口，不绕过。

## 2. 固定48槽及配对单位

固定slot-major child顺序保持8episodes×b5/b10/b15，每child fixed/adaptive，24child/48method，known独立roots仍为2。本批41selected可请求physical raw5；5个adaptive原H3×8整节点拒绝、1child对应2method未到达原样保留。不得删除拒绝、从41中按预测Q/动作是否变化/物理成功挑子集或把仅fixed可执行的5child伪装成完整配对。

主要配对单位是同一原episode、同一boundary的两方法，满足二者exact physical起点/任务/goal上下文均认证后才有paired difference。本批至多18个双可执行child；另5个只有fixed可执行、1个均not_reached。这不是18个独立root，统计必须保2root层级、episode/prefix_arm/source_mode/boundary依赖，不伪造独立样本量。

每槽必备身份：root_slot、sourcepack原slot/recipe、task contract及原codec hash、episode_id、prefix_arm（GN/D2P）、source_mode（I/R）、condition若原source明确提供则逐位继承、boundary、compiler mode、child/context/ancestry SHA、signal group/receipt SHA、compiler report/strict/prepare SHA、selection_digest、winner原index、selected raw5与独立prefix path。缺condition不能从文件名或source_mode推断，更不能填off/on；明确unknown或在该任务不适用时标not_applicable。

## 3. 只执行selected raw5，不执行规划H或拼接新suffix

唯一动作权威为compiler strict rows.selected.raw5，必须与winner完整actions的前5及独立selected-prefix replay逐位一致，float32 (5,2)、原raw domain、原normalizer，不除100、不round、不zero padding。H3规划15步不是本次执行15步，不接后10步或旧teacher suffix。每槽最多5次新的public step，原native terminated/truncated立即停止且不padding、不重置继续；实际returned可0..5，失败未知不填0成功。

这是一组**原observed boundary反事实分支**，不是把b5新动作接入b10再接b15的在线闭环。b10/b15起点仍为原P4 source prefix，不用本次b5物理终态替换。若以后要online rollout，必须重新观察/编码/付child信号/编译并另立因果合同，不得把当前saved计划离线串接成闭环。

## 4. 精确恢复原physical boundary，拒绝仅latent/7state恢复

优先采用同原生环境版本/seed/reset/placement recipe→原真实非零warmup10→原source已实际returned raw前缀至boundary的deterministic replay。每个方法独立恢复，task/goal/渲染配置按原source时间线设置；所有恢复步实际收费。不可只set七维obs或复制latent，因为body速度、unwrapped angle、agent状态、solver/contact缓存等未由latent或7state完整指定。

原source所有goal绑定/goal_pose/goalRGB、renderer/overlay变化与reset时序必须沿明确输入边继承；不能默认goal_pose代表构造真实goal，不能为像素相同改图。原warmup10精确float32权威不能假设全零。原P3 root physical helper只会root warmup，不等于已支持P4 b5/b10/b15恢复；只能显式参数化复用冻结pure函数，不monkeypatch globals或借旧schema白名单。

到达boundary后必须逐位核对：完整native snapshot、七维state、历史raw10及原executed prefix、目标/renderer不变量、对应真实PNG file/pixel SHA、source observed/encoded时序及child latent history的明确旧输入闭包。恢复过程保存每个public-step返回、原始action info与全部render调用。不能因最终latent相等就认证body或任务等价。

原生angle边界沿已修复saved规则：obs block.angle等于requested unwrapped angle `%2pi`；snapshot block.unwrapped_angle及float64_bits仍精确等于原requested，其他六维exact。不得宽泛容忍2pi扰动、改变原recipe或给不同snapshot建立等价。任何恢复/PNG/上下文不一致都标恢复失败并保留48分母，不容差修复、不重新选根。

若native因接触历史不确定导致exact replay无法复现，必须unknown/blocked；不能增加一个“忽略snapshot差异”分支。直接snapshot restore只有原environment存在完整、已审核且覆盖相关隐状态的原生restore ABI才可另合同批准，不能在本桥临时发明。

## 5. 复用旧physical证据的严格边界

默认new execution不假设存在旧等价结果。未来prepare若提供明确旧physical输入，必须先完整oldphysical.consume，再验证完整task/source/raw ancestry、original prefix/context/native snapshot/goal及renderer逐位一致，selected raw5及停止语义一致，并把旧report/prepare/PNG/日志/源SHA全绑定。commands相同、同root、同Q或同终态均不足以复用。

route需在新physical结果出现前封存为execute/inherit/blocked；不得按old成功筛reuse、不按new失败回退旧结果。inherit明确historical、freshness=False，新物理工作0但旧实际成本独立保留；不能新旧各步拼接成为一个执行receipt。缺旧权威或identity不全时保unknown；不要为凑“完成率”强行复用。

## 6. 资源上界与提前停止

若全41槽均new，单槽恢复public steps上界为10warmup＋boundary（5/10/15），执行最多再5。精确总恢复预算需从41槽原boundary roster求和，不按结果挑槽。已知完整23 observed child的boundary和为225（完整24为240、未到达b15减15）；fixed全23可执行恢复455、执行≤115；adaptive18可执行具体boundary和须prepare从已严格消费的5拒绝身份扣除，不能猜。

保守所有41 boundary≤15：恢复≤1025 public steps，执行≤205，总≤1230。若复用明确旧证据，按sealed route分别计new/reused，不按实际成功补预算。该上界不是批准的物理cap；constructor/reset/set_state、goal setter/bind/render、public step、native内部physics step、PNG及overlay额外渲染、close必须逐项从原ABI和sealed roster推导，另报实际attempted/returned。native内部step的reset代价也计入，不仅把public×10当全部physics。

output/disk/wall必须在后续prepare前以真实group样本或RAM lossless容量估计给出完整41新分支+48槽报告的保守余量，由MAIN批准。不得照搬4root的616physics/204render或凭空给本41分支套同cap。whole-job wall从完整输入verify前到最终receipt完成独立测量；inclusive嵌套资源/时间不能加总，失败constructor内部工作未知时保null/scope，不制造0。

native终止起点不得执行新raw；source恢复前缀若提前native结束而权威称boundary存在则恢复失败。requested5、attempted、returned及first failure分开记，stop后后续步骤未dispatch；0returned合法工程失败receipt不等于成功5steps。不能通过notreached/earlystop少步宣称优化省算力。

## 7. 原生指标和严格消费者

执行前与每个实际returned状态记录原生指标：native jointXY（agentXY＋objectXY的四维范数）与原angle成功阈值组合的真实success；native7 reward为包含速度的七维Euclidean距离负值，不能当任务成功。另分栏objectXY误差、agentXY误差、wrapped角度、native angle term、velocity误差、native7距离、原reward/terminated/truncated。阈值不改，角度距离定义不混用。latent Q、predicted first-boundary误差需另saved分析，不能用Q改善代替物理任务改善。

checkpoint为真实step0与实际terminal（若完整则5）；缺step5不能补图。保存每次真实PNG的path/fileSHA/pixelSHA/state/snapshot，render前后不改变physics；当前compiler latent prefix不是新physical checkpoint。若需要后续RGB encoder评估，由MAIN另授权CUDA adapter，不在physical桥中加载模型。

strict consumer须重建完整producer及source权威闭包，核48槽/route/请求raw5、每次恢复与execution的真实attempt-return、动作info、goal和snapshot/PNG、原生earlystop和全部资源、原生指标独立复算、最终body/源码漂移和实际report/所有artifact hash。failure/unknown不能被summary覆盖；无实际持久strict报告不称已消费完成。saved consumer不执行physics或模型。

最低RAM负向测试：旧root代替child、跨episode/prefix/condition、raw时序/尺度/zero变化、丢5拒绝/2notreached、仅state恢复缺snapshot、2pi snapshot篡改、wrong goal/render/context、恢复提前终止、新执行0return错误当成功、未full consume复用、commands相同却不同task、旧新日志拼接、earlystop padding、native7成功混淆、资源/文件/链篡改和输出cap失败保全分母。RAM通过不是物理/科学证据。

## 8. 下一次明确授权所需交付

MAIN提供：compiler report绝对路径＋持久strict路径/SHA；P4完整native/source权威consumer准确接口与回执；是否允许new execution/明确旧physical reuse输入；预算/输出/执行authorization合同。然后独占本stem source/tests/doc实施prepare/verify/run/consume，默认run在环境import/构造之前拒绝无授权。所有输出NEW排他、外部实际SHA，未来prepare/report SHA不预填。

在这些前置未齐之前仅协议，不自动prepare/run、不自行采集目录或调用physics/neural。P1/P2/P3/P4/freshness/fullC3/physical release/compute/wall advantage均继续false；局部391对634即便实际完成也不是等算力。当前只比较已知两根历史起点反事实raw5后果，不声称fresh root、完整组合树或在线调度科学成立。
