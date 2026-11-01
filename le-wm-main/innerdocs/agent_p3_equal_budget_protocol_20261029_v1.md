# P3 资源可识别对照：共同向量预算与实际匹配实施协议

2026-10-29，仅新增本文。依据已冻结parent源码/已完成报告及既有预算设计，使用experiment-design技能组织实施前置与验收；不训练、不调阈值/候选，不改旧封存，不运行生产/NN/physics/fullstrict，不开agent。本文写于原parent结果之后，**不是原实验预注册**；以下新比较须在新运行之前由MAIN封存实现、预算与输入身份，不据未来结果选规则。本文不授权运行、不签P1/P3/C3/freshness，也不使用任何未来物理结果确定预算。

## 1. 旧证据的准确位置

路径相对 `/home/gbk/code/LeWorldModel/le-wm-main`，下列SHA已只读核验：

|对象|SHA256|
|---|---|
|`experiments/spectral_reachability/agent_p3_parent_native_adapter_20261029_v1.py`|`150a1c4189a68f9b7c71479e5aa6980cea490fdeed0d5524b7bfad0cc78c2558`|
|`experiments/spectral_reachability/agent_p3_parent_native_cuda_20261029_v1.py`|`eeea5710d4171b7c43f99d1f2b5ad2412e03a25ba14e8fc5589ea5ef73f57b3b`|
|`results/agent_p3_parent_native_cuda_20261029_v1_cuda_MAIN01/report.json`|`75f4c162aaa75223dfd8523381e7614c8a35637afd52278df8a80d6affa34e40`|
|`results/agent_p3_parent_native_cuda_20261029_v1_consume_MAIN01.json`|`622d1cdaf8d1ed69c4e78f8d1f5c7c6114285db6121f0021fbbaf6573a8a76ad`|

本次只读继承外部strict，不重新运行它；结果用于确认不平衡/覆盖缺口，不据Q或任务收益挑新预算。既有 `agent_p3_equal_actual_budget_tree_protocol_20261028_v1.md` 已区分equal-cap与matched-actual；旧151/971预测上界含另一种逐候选双replay合同，**不能移植成当前114/562实现的预算**。既有 `agent_p3_resource_matched_followup_20261029_v1.md` 的双面板对照也不是当前parent树的自然资源匹配。

|旧槽|实际predict / J / VJP|实际搜索行为|可下结论|
|---|---|---|---|
|root0 fixed|114 /18 /3456|H1、9个扩展parent、36个progress候选，depth3，完整双replay|真实有界progress树完成，不是完整P3证明|
|root0 adaptive|28 /4 /768|先root H3×4 progress，再在depth1请求composition/extension=R、width8被拒；无最终model_plan|不是“根一开始就R失败”，不是R8实际执行后无益；已付前缀/信号保留|
|root1 fixed|114 /18 /3456|同fixed结构|同上|
|root1 adaptive|278 /26 /4992|H3、13个扩展parent、52个progress候选，depth3；逐parent width全4|更多搜索和更长H并存；并未生成R8，亦不是等资源比较|

原adaptive frontier beam上限8与实际每parent候选4不同；原root1访问parent为1+4+8，fixed为1+4+4。两者实际最大depth均3，不能说本批验证了“深度被信号改变”；原代码允许的child stop/depth变化与真实出现须分开。

## 2. 先固定声明：三种公平性不能混名

对同一任务root，方法m的实际付费向量记为 `A_m`（attempted）、`C_m`（completed），共同预注册上限为 `B`；每轴 `C_m≤A_m≤B`。J按原shape/H保留元数据，模型forward按batch/完整context-token/精度保留，不能只比predict API。

|声明|事前定义与可检验条件|不保证|
|---|---|---|
|共同预算可行性 / equal-cap|两方法同一逐轴B、同一同步elapsed上限W、同验证/输出/内存合同；实际向量全部报告|不保证实际花费相同，不代表强/最优fixed基线|
|指定轴matched actual|事前列出集合K，逐轴核A和C相同，另核shape/H/精度；其余轴不隐藏|例如predict/J相同仍不代表其余资源相同|
|完整已测向量matched actual|所有事前声明的已测轴A/C及相应工作shape逐项相同；缺项则该匹配声明unknown/未满足|仍不是equalFLOPs/等墙时/等能耗，也不保证所有未计量计算相同|

FLOPs、engine backward、scalar backward、独立AD-primal计量和J元素如未由真实instrumentation提供，保持null/unknown。不能把J×192当真实engine调用，把API、token、VJP简单加权相加当“总算力”。即使有事前硬件基准得到固定权重的scalar work proxy，也只能另报预注册proxy，不能签FLOPs；本协议最小版**不使用这种权重**。

自然策略的离散parent/4或8整队列/不同H/不同最终replay长度/原child stop使精确实际向量相等通常无保证。没有事前可行构造时仍可运行equal-cap比较，不因不可能保证精确相等而停住所有工程研究；但事后不能改口“等实际预算”。

## 3. 最小可实施主线A：共同上限的progress-only有限对照

目的：比较原fixed/adaptive规则在**同一可用资源包**下的完整执行、覆盖和模型程序输出，不把实际278比114的差距归零。既有 `parent.run_core(..., mode=..., caps=..., max_requests=144)` 已有显式caps接缝；新独占driver传同一B，不修改旧源码/default_caps，不伪造旧报告schema。保持原choose、beam、深度、稳定排序、生成器、模型/goal192/identityQ/normalizer/raw chart/完整parent context、whole+segmented最终验证。

### A的事前预算：结构推导而非按实际278回填

最小建议B**逐轴复制当前源码 `default_caps('adaptive')`**给两策略，来自最多17parent/136候选/最长9packet双replay的progress外壳；不是用观察到的最大实际278作裁剪。prepare封存完整27键及其来源SHA：

|共同轴组|每arm上限B|
|---|---:|
|predict API / samples|562 /562|
|predict context tokens|1686|
|action_encoder calls/samples/context tokens|562 /562 /1686|
|predictor calls/samples/context tokens|562 /562 /1686|
|pred_proj calls/rows|562 /1686|
|J / VJP / grad / raw_backward|34 /6528 /6528 /6528|
|signal compile / schedule|34 /17|
|array SVD / solve|51 /17|
|program compile / candidate metric|136 /136|
|完整最终replay API|2|
|model SVD / 新encoder calls,samples / 新projector calls,samples|各0，限当前真实实现无这些调用|

这里的0是当前program search不执行视觉编码等工作的已定义增量，不是历史获取免费，也不把未知axis填0。所有keys在prepare与Ledger逐项核对；若新增实现出现这些工作，不能继续用本B伪称覆盖。history acquisition/source/model加载与hash开销另列；shape约束保持原batch1/3latent/2history-token，H1/H3的J宽与反传图长度仍保存。

为避免墙时逐位相等的不可实施条件，A建议每arm同步elapsed cap `W=3600s`，采用现有whole-job安全上限量级作为新的共同arm上限，并非根据本批实测速度拟合；新全四槽job安全cap建议18000s、输出1GiB含256MiB final reserve、disk floor200MiB、CUDA allocated3GiB/reserved4GiB。**这些是本新协议建议值，须在新prepare前由MAIN接受并写死，现有job合同不因此更改或自动授权支出。** 输出可先于W耗尽，照样保留失败，不按剩余空间删证据。

同步elapsed起点是该arm首个signal前CUDA同步完成，终点是最后一次实际工作/异常展开后的CUDA同步；包含其间CPU policy/IO等待，名称是host-synchronized elapsed，非纯GPU kernel time。每次dispatch前及返回/同步后检查，完整记录started/ended/超时overrun及同步失败；不能承诺阻断已经在运行的kernel。加载/prepare重建/跨arm来源核验、最终报告保存时间另计whole-job wall，不藏在arm外声称端到端更快。两方法不并发、无其他GPU job，顺序固定root0 fixed/adaptive、root1 fixed/adaptive；固定顺序的warm/cache效应为限制，不冒称随机化wall因果效应。

### A的运行与停止合同

1. 固定四方法槽、每槽144请求位，共576；原两known roots全部保留，不触reserved。每parent使用完整自己的三latent/两history-token，新付H1/H3信号；root/兄弟J不得替代。原模型输入不接native state、reward、物理outcome；这些不参与预算路由或选择。
2. `choose`完整decision先durable封存；整4/8候选队列身份原子分配，不能拆队列凑预算。严格调用当前 `generation_minimum + verification_reserve` 预检，并给每个真实native调用逐轴guard；预留不记实际工作、未dispatch不记付费attempt。
3. 未用预算允许保留。fixed正常完成114后停止，不再加种子、重启、补重复零动作/诊断、强迫H3来凑562/34J；adaptive按原continue/beam/cap/type停止，不为匹配fixed改回H1或删parent。
4. 合法R请求在当前progress-only adapter仍整8拒绝；先前已付signal/候选、未请求位、diagnostic_best均保存。unsupported/cap/timeout/runtime failure时不释放partial model_plan；原core只有正常完整验证成功才有model_plan，不另以anytime名义偷放diagnostic_best。失败attempt不退款、不重试、不跨root/跨arm借用剩余额度。
5. 保留旧driver真实运行故障会block后续槽的语义；新四槽中未执行者为blocked/unknown，不记0工作或0收益。共同B不是“每槽一定完成”的保证，global source/模型守卫失败也不以重复运行筛掉。

A可报告：同cap下完整计划覆盖/unsupported/cap/runtime failure、真实资源与whole验证、输出模型Q/动作长度/H/实现宽深。自然停止后实际资源不匹配照实输出。A仍不是强fixed成本最优性测试，不证明adaptive在等实际计算下优于搜索更充分的fixed；要加强fixed必须另外事前定义有用搜索算法，不能在本批结果后为某根加算。

## 4. 若研究必须要求“真正实际匹配”：单独预注册B线，不能伪造

严格匹配不是A运行后的筛子。最小新增是**RAM预算可行性编译**，不先跑CUDA找碰巧匹配的根：输入冻结policy/生成器版本、全工作成本映射、H/width/depth域、队列原子性、最终replay与固定四槽；输出能否对所有合法决策路径保证事前声明K内实际匹配，以及不可保证的路径/离散余量。遇未知R内部成本或未计量轴，输出unknown而非算0。

当前progress成本结构可静态推导：一个实际扩展parent信号为8predict/2J/384VJP、2signal SVD；候选生成再付 `H×width` predict、width个program compile与metric、一次solve和一次array SVD；一次schedule。最终两replay按所选完整链的真实packet长度收费。原child-policy停止parent也已支付信号/schedule但不支付候选；R拒绝只支付到拒绝前实际工作。已知shape及真实调用数量映射，不能冒称FLOPs成本。

如不能构造保证匹配的合法有用调度，B线结论就是“当前约束下不保证matched-actual”，**不阻塞A继续研究**。若MAIN另行选择新的budget-neutral allocator，必须先封存：每策略全候选工作单元库及成本、完整工作块入队规则、剩余预算处理、稳定tie-break、plan提交/验证规则、失败保留。新allocator是新算法，不倒签原choose；不能用实际native标签选任务/路径，不为凑数读outcome、padding或过滤资源不匹配样本。即使某次四槽恰好全部匹配，也仅按事前K声明，未知FLOPs仍null。

另一可实施但**不同问题**的matched-work对照是共同固定完整面板/有限树工作带：两规则面对同一已付库，只改变采用哪个封存程序。必须明确shadow计算与共享缓存来源、所有候选完整性及模型only选择；它识别的是“利用同一信息/库的决策差异”，不是adaptive动态分配计算的节省或效率。既有shared-panel已经覆盖部分此问题，不能再包装为新完整tree/R证据，也不在本文重复要求生产。

## 5. 调度激活必须逐项证实，不能只看policy名字

|所称机制|必须事前固定并记录|未出现/未实现时的准确口径|
|---|---|---|
|H自适应|逐parent原decision.H、实际生成edge长度5H、J的10/30 raw维度、释放raw与whole路径|只运行H3可证H1→H3干预，但不是在一条树上动态切H的证据；H不是物理执行时长|
|候选width4/8|requested、allocated、attempted、returned四种宽度及整队列状态，候选索引全保留|R8被拒只叫requested8，不叫width8实际执行|
|frontier beam4/8|排序前后node身份、pruned完整候选、下一层真实parent数|与每parent候选宽度不同；更多parent可能单由beam上限造成|
|depth/continue|parent depth、原program_depth_limit、continue、实际returned最大depth、child stop原因|两策略都depth3不能证明动态深度产生差异；H3不等于depth3|
|程序type/组合|原program/extension type、真实edge内容、独立whole/segmented全context/raw闭包|progress链仅支持progress组合；composition标签不表示R已实现|

观察不到某机制激活，保留全分母并报告未测试到，不称机制负结果。激活后模型Q下降但无物理观测，仅是model-only输出；若最终program长度不同，Q对应不同终点/动作程序，不能不加说明做“同时间真实收益”比较。搜索树不自动获得固定20步物理准入；任何新physical比较须先固定完整raw执行时域/早停/检查点与native各轴，不能从树best截FIRST5冒充已验证整段物理结果。

## 6. 可执行R是完整P3比较的硬前置，不以加预算修复

当前A可先做progress-only有限合同，但root0 R拒绝保留，**不称完整P3等预算对照**。要完整研究原type/width/depth，必须先另版补齐R：

- 在任何新结果前固定8请求逐索引的真实raw prefix proposal来源、prefix/suffix H、box/约束、零控制参照、immediate-Q与post-prefix reach评分/选择、stable排序；不能把当前progress方向8个fraction重标R，也不能以物理收益选prefix。
- 每prefix真实预测完整后继context，后继所需nominal/J实际付费且绑定该context/goal/normalizer，保留完整post-prefix动作见证和非贪心标识；允许全体都没有非贪心witness，不筛根/改阈值。
- R可能含内部更多proposals、derivatives与验证；必须给出其逐轴成本及最坏有界数量，再对fixed/adaptive共同预算重新封存。**562/34J是progress外壳，不是完整R cap**；不得在run中抬cap或用剩余额度绕过预检。
- 原请求types原样dispatch，parent→R→progress/composition的全部raw/context闭包、最终whole/segmented和失败账闭合；RAM测试验证R确被调用、不能fallback，随后由MAIN审读与独立HOST验证。

R尚无确定proposal/成本协议时，完整P3对照状态是“合同未实现/成本未知”，不是代码故障或R科学负结果。合法拒绝也是真实覆盖不足，不能从四槽分母删除。此处没有批准任何具体R算法或新阈值。

## 7. 最小新实现ABI与审读判别

下一实现不改旧source，复用显式caps的`run_core`/真实NativeTree低层，单独schema prepare/run/consume：

- prepare：四槽固定顺序、144×4分母、完整task/model/goal/Q/normalizer/history/rawdomain身份SHA、原choose/generator/source闭包；`comparison=equal_cap_NOT_equal_actual_NOT_equalFLOPs`、完整B、W及timer scope、failure/release规则、K（A为空，不发matched-actual签名）、R支持表与工作成本版本。
- run：每个slot实际cap、所有parent/context/J mapping/完整decision、队列/beam/prune/childstop、attempt/return/failure和资源元数据、同步elapsed/overrun、模型/precision before-after、持久partial、whole/segmented双replay。不得重标旧114/562报告。
- consume：由MAIN执行真实完整saved闭包，独立复建原policy和精确资源；summary只汇总指定SHA产物。输出equal_cap_contract、各轴actual differences、K匹配/unknown、覆盖、计划可用性与未执行槽，不凭进程exit0签fullpolicy。

|审读观测|解释|禁止|
|---|---|---|
|同B/W但actual不同|共同上限成立，实际匹配未满足/未声明|称equalFLOPs或按实际资源差直接校正收益|
|R请求被原样拒绝|当前有限合同正确执行、完整type覆盖未实现|给R填0成本/0收益或筛掉root0|
|同支持谓词但错误拒绝、用旧parent J、半队列偷偷放行|具体实现/身份问题，先定位修复，受影响科学比较不可解释|叫P3理论失败，或放宽原gate|
|完整支持与身份闭合、等cap下输出无优势或更差|这些已执行固定任务的有限model-only不利证据；资源差异仍报告|推广任意任务、替代native物理收益、签P1/C3|

RAM验收至少包括：两方法B不同拒绝；仅predict相等但J/VJP/shape不同不得matched；不同实际elapsed不误拒同cap；同cap越界/同步overrun；failed父/returned子账不退款；未知轴不补0；整队列装不下；最终verification reserve不当实际工作；R8整拒绝与无fallback；root/child/兄弟身份替换；child真实stop；beam与candidate width混淆；partial best不得release；blocked槽保持unknown；全四槽/576请求位不删。测试不替代实际CUDA/物理证据。

## 8. 交付与下一步

仅本文，无新代码/测试/生产/fullstrict；未修改旧结果或合同。最小下一实施是A的**新schema共同B注入＋同步wall/完整saved账**，先RAM全边界回归，再MAIN审读冻结后执行；完整R线先冻结生成/成本协议，不能暗中并入A。新实际report/receipt/source SHA尚未产生，不编造生产CLI或未来SHA。所有 known-root 工程证据和任何新的equal-cap结果，均不能自动签发P1、完整C3/P3或独立fresh确认。
