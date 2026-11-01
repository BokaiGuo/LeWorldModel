# 提请 MAIN 审读：actual-cp5 二维 I/R 中心割线共同方向

本文件仅为独占最小设计协议，不实现、不授权生产，不修改任何既有bank、模型、科学gate或封存结果。
唯一推荐路线：已付小半径四项 → 二维中心割线 surrogate → Gram 最小范数共同方向 → 固定四个缩半半径的完整真实模型重评。
这是基于已见known roots提出的新探索性方法，不是事前盲确认、完整梯度、原R等价、equal-budget或物理收益证据。

## 1. 固定输入、评分出处和分母

沿用冻结 `agent_p3_goal_conditioned_R_actual_cp5_cuda_20261029_v1` 全四组，顺序root0/H1、root0/H3、root1/H1、root1/H3，不增删/交换根。
真实task builder SHA `ce6670d84f6f0f712bdfe5e1b7471be65b0441338c18010f79b3967ef28397c7`；actual abs5/10/15、原goal/identity192 Q/normalizer/raw域与双任务身份保持不变。
沿用其prepare SHA `d54438930818af9bfd65c0f23a0666a19b320d3a62fb05256d5ac05beb798771`。
MAIN提供的真实report SHA `9bedec46c5c0e74b73c145935280b2efe400f617f8bbe7b72205fedb60a20d4f`、strict receipt SHA `b50268a2018e6da1fbc46a4d19d701cd4a9a575cfaf581807cf29c01861d7ba8` 作为后续实现应精确绑定的authority；本次未重跑consume或读取评分选参数。
生产own-schema flags使用真实 `full_C3_demonstrated=false`；核内 `C3=false` 不能替代生产字段。所有physical release、P1/P3/full-C3、freshness、公平性声明仍false。

继承四组的原0+8项全部保留，包括原大半径，不因为本方法只用小半径就抹去其成本/分母。新试探另有固定4组×4半径=16槽；无效组仍保存四个null及明确原因，不把null填成零分或已执行零动作。
使用的已付评分必须来自同一组small-radius索引0..3：d0负、d0正、d1负、d1正。绑定完整bank.local_task/native_response/mapping/source capsule、两个方向、params/raw字节、每项真实edge/postcontext H1 J/原bounded_ridge/suffix与原zero。
禁止跨root/horizon/方向/半径拼合，禁止将未完成或退化项当作有效四项。未来实现需要认证真实同local task，而非只接受outer/root同名hash。

## 2. 割线在什么坐标中成立

令n=10h，D=[d0,d1]∈R^(n×2)，两列取冻结bank中float64原方向，不重新SVD或旋转原bank。
chart参数v∈R^n对应动作 `raw(v)=float32(tanh(v)).reshape(5h,2)`，仍由原normalizer恰好转换一次。
令I(v)为该完整edge终点的原immediate；R(v)为该终点后重新编译H1 J并调用原bounded_ridge得到的own_reach_value。R仍是原有界线性reach指标，不是suffix真实非线性终态cost；后者另存为bounded_suffix_cost，不混作gate。必须完整重评原R，不能用新割线sR预测替代它。
原小半径 r0=sqrt(5)/4；四个评分确实对应 v=±r0*d0、±r0*d1，而非raw空间的±r0*d。

分别计算二维中心割线系数（j=0,1）：

    sI[j] = (I(+r0*dj) - I(-r0*dj)) / (2*r0)
    sR[j] = (R(+r0*dj) - R(-r0*dj)) / (2*r0)

这是沿现有tanh/float32调用链、固定非零半径的二维割线 surrogate，不是∇I、∇R的精确值，也不是完整n维梯度或二阶AD。
只有额外光滑/精确算术等条件下才能讨论中心差分偏差；本实验不假定这些条件已经被模型或cast满足。缩小新候选半径不会自动消除取自固定r0的割线偏差。
sR来自完整postcontext-J/solve评分差，包含这些模型机制的有限幅度影响；不能据此声称已求出dJ/dv或envelope精确导数。

## 3. Gram 最小范数方向：确定性数学合同

G=DᵀD，一般不等于I；不得把d0/d1误当正交基。求共同方向的目标是原n维chart中最小范数，而非系数c的欧氏最小范数。
两列及四评分必须有限且非null。用一次float64 G经济SVD记录两个奇异值；若smax=0或smin<=64*eps64*smax，标记 `gram_rank_unresolved`，保留四新null槽，不换成单方向、不补单位基。
该64eps只作新surrogate数值可解析性判定，不是对原资格增加epsilon，也不能将拒绝称为精确不可行证明。
通过后另付一次真实 `solve(G,I2)` 得M=G^-1；不使用未记账的QP求解器/多轮solve。

    normI = sqrt(sIᵀ M sI), normR = sqrt(sRᵀ M sR)
    a = sI / normI, b = -sR / normR

normI或normR为0、非有限或无可表示正值时标记 `secant_zero_or_unresolved`，不以另一评分代替；负根号量等不一致标记数值失败。
归一化只使两个约束无量纲，并保持割线的原符号；不训练、拟合统计或改变原I/R评分。
定义严格限定在该二维surrogate上的凸问题：

    minimize  (1/2) cᵀ G c
    subject to aᵀc >= 1, bᵀc >= 1.

令κ=aᵀMb。精确算术下|κ|<=1；当κ>-1时最小范数解为

    c* = M(a+b)/(1+κ).

理由：两约束的dual norm均1，解的两个约束同时活跃；κ=1时公式仍给同一最小范数方向。κ=-1是surrogate相反法向，约束不可同时满足。
浮点实现若κ不在[-1,1]、1+κ<=64*eps64、c*非有限，保留 `secant_cone_unresolved`，不偷偷clip κ、加ridge或换目标；这也可能只是数值未解析，不能宣称真实非线性可行域为空。
必须记录G/M/κ/c*、两个约束残差与未归一化sIᵀc*/sRᵀc*；后两者必须严格>0/<0，否则数值拒绝，不把容差用作资格。
最后 d=Dc*/sqrt(c*ᵀGc*)，float64单位chart方向；记录实算norm与全部线性surrogate预测。公式的精确约束推导不等于对有限精度实现或真实模型的精确证明。

## 4. 唯一新候选序列与原 exact gate

统一固定四个正向半径，按k=1,2,3,4递减：

    rho[k] = r0 / 2^k
    v[k] = rho[k] * d
    raw[k] = float32(tanh(v[k])).reshape(5h,2)

不加负向、旋转、重估割线或逐槽自适应半径；H3不乘sqrt(h)。所有有效组全部四半径都真实评估，不能首个成功即停或只报告成功半径。新方向/半径仅依赖上述固定模型算法，不读reward、native success、state outcome或物理结果。
每项从同一actual-cp5完整context出发，执行完整h包edge，真实终点后付H1 J，原bounded_ridge、原suffix诊断与两次metric全部保留。不得只凭割线预测放行、不缓存成旧候选终点、不把suffix拼入edge。
新项与同组原paid zero比较：I_new>I_zero 且 R_new<R_zero，严格saved原值，无epsilon、无相对改善阈值，不改gate。
旧zero可以复用的前提是同task、goal/Q、model/modules/source、normalizer、完整context和评分语义认证完全一致；完整原zero费用仍在继承账中。无法证明时refuse该来源，不默默重算/补零，须MAIN另行固定资源。
本最小切片只输出完整16槽评分/资格和原0+8并栏，不发布model delivery、commit或物理动作；故不需要新whole/segmented验证。本协议不授权下一物理线使用eligible。
新候选若用于未来交付，必须另协议/另付whole+segmented与来源认证；本轮不以省略验证冒称已可交付。

## 5. 预算：继承费用不是免费，不冒称等算力

原全四组完整goalbank实际费按其真实receipt继承；下表的196/40等对应原全8有效且no_witness的完整阶段。实现必须核对receipt实际向量，不把表当作缺字段补0的来源。
新阶段最多4组×4候选；每个新候选原item(h)需(h+3) predict、1 J、2 replay、1 signal_compile、1 program_compile、1 solve、1 arraySVD、2 metric。
四组每组另有surrogate 1 arraySVD(G)、1 solve(G,I2)、1 program_compile；这些也不是免费。

|资源轴|继承原完整阶段|新阶段上限|累计上限|
|---|---:|---:|---:|
|predict_API / predict_samples|196|80|276|
|action_encoder_calls / samples、predictor_calls / samples、pred_proj_calls|各196|各80|各276|
|predict_context_tokens、action_encoder_context_tokens、predictor_context_tokens、pred_proj_rows|各588|各240|各828|
|jacobian_API / signal_compile_calls|各40|各16|各56|
|VJP_directions / grad_API / raw_backward_callbacks|各7680|各3072|各10752|
|replay_calls|72|32|104|
|solve_calls / array_SVD_calls|各40|各20|各60|
|program_compile_calls|40|20|60|
|metric_calls|72|32|104|
|encoder_calls / encoder_samples / projector_calls / projector_samples / model_SVD_calls / schedule_decisions|各0|各0|各0|

H1每组4新项=16predict/4J/8replay；H3每组=24predict/4J/8replay。四组另有4次2×2求解/分解，已计入20solve/20SVD。
原AD-primal_predict_API共44（parent各H合计8，加36个post-J），新16，累计60；原J_elements=1920×44=84480，新30720，累计115200。这两个描述轴不与predict_API重复求和，标量backward_engine_calls/FLOPs/CUDA时间不得由API数推造。
割线/Gram构造、范数、矩阵乘加、源认证和saved验证CPU开销记录wall/实际调用，不称0FLOPs；非同构的分解大小/NN样本/上下文轴分开报告。
若来源已完整认证并复用，本轮实际新派发最多80predict/16J，不能声称本轮重新执行276predict；276是含继承的累计成本。若重新生成原阶段，则196也必须真实重新付费。
预算cap仍为上述全组上限；退化只减少真实派发，保留attempted/returned/failed，不padding不refund。每组bank surrogate+四完整item应原子预检，任何运行异常停止后续派发且保持四组/16槽分母。
建议工程上限沿原新runner：2GiB新output含512MiB收尾reserve、diskfloor200MiB、总wall3600s/组1800s、CUDA allocated3GiB/reserved4GiB；旧封存字节不复制、不计作本轮新增空间，也不删除。正式生产前由MAIN固定新实现及cap映射。

## 6. 与完整双梯度/二阶AD路线的比较及唯一推荐

完整路线需对edge动作求∇I和∇R。reach的R依赖postcontext nominal、post-J和bounded optimum，不能把goal梯度或固定J残差梯度当作∇R；完整推导至少涉及dJ/da、原bounded约束/active-set与非唯一点的可微性条件。
其必要前置包括：原192方向Jacobian构建支持并审计create_graph、SDPA/算子二阶导可用、冻结权重但输入可导、bounded solve/约束导数语义、原生二阶IO/图/精度认证及新增AD/memory预算。现有一阶signal receipt不能为这些背书。
有限差分完整∇R也不是免费替代：H1/H3的n=10/30维若用逐坐标中心差分，仅2n个新R点评分每组就需20/60次完整edge+post-J，不是本切片的4次；必须单独预注册预算。

唯一建议先做本二维中心割线切片：只增加固定16个完整候选评分，能直接检查“有限维reach-aware共同方向是否比goal-only方向更贴合实际资格”。
限制必须并列：二维以外方向不可识别、固定r0割线有偏、Gram/共同锥可能退化、缩半不能保证真实符号、post-J仍只局部有效、gate无交集不等于全空间无解；也不保证比旧bank更好。
后续如失败，不在本批加方向/改半径/调64eps或gate；需先完成全槽解释，再由MAIN决定另立新协议。此文不授权该后续，也不预言P1/P3机制或物理收益。

## 7. 将来实现的最小验收（本次不实现/不执行）

- 四组/原32+4zero/新16槽的真实source身份；索引/正负/半径错误、坏local配对、缺评分必须拒绝。
- 非正交D下Gram最小范数闭式与独立小型RAM凸问题对照；同向/相反约束、zero slope、null方向、病态Gram与浮点越界保留原因。
- 新半径顺序/参数→tanh→f32→原normalizer一次、H1/H3 edge/suffix分离、所有候选实际post-J/原ridge；原exactgate的tie/近tie不受surrogate数值规则影响。
- 完整27轴及继承/新付费分栏、派发前原子cap、失败不退款、null不派发、不能早停挑成功；完整nativeIO/source/model/精度/文件census/saved消费由MAIN审读生产代码后执行。
