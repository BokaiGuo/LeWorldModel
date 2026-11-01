# Moving U/D/S/N：整数配额重合的读表解释与下一轮候选预注册

2026-10-27。**本文件是新分析/设计文档，不是对已执行协议的修订，也不是新的已批准实验。** 不修改旧source/tests/prepare/report或共享账，不运行模型、物理、训练、重新编码或候选评价。采用experiment-design的机制检查与配对对照拆分思路，不采用看结果调参、训练或挑最好根。P1/P3与完整C3科学准入保持false。

## 1. 本轮确切发生了什么

证据以 `/home/gbk/code/LeWorldModel/le-wm-main` 为相对路径根：

| 素材 | 外部SHA256 |
| --- | --- |
| 已批准外层 `../docs/D2P_MOVING_BUDGET_SCHEDULER_PREREG_20261027.md` | `8e8cd4091dbaed5991dc119638ffdc00c8aed0852d0ed4c54cc6b1b3a82009bf` |
| scheduler `results/moving_history_budget_scheduler_cuda_20261027_v1_cuda_main01/report.json` | `710892008bbd47aa188a9b9391972976d3a94d2a0067e46911ff37f05be8369b` |
| physical `results/moving_budget_scheduler_physical_20261027_v1_main01/report.json` | `c9169b685e4e7f7d79ea2101aa81ad55b95593f96d6158ed8ff0815de136bd70` |
| comparison `results/moving_budget_scheduler_comparison_20261027_v1_saved01/report.json` | `1eb2c51d6d4459b4aa6723fafed40ca8034729aa4f27d4359c28964125840b64` |

本轮并非未计算信号、零rank、零residual或fallback均分。saved signal如下；这些是原off-Q、零动作H2局部几何的模型内量，不是物理可达性证书：

| 固定slot | rank | residual energy `E` | `chi` | `w=(1-chi)E` |
| --- | ---: | ---: | ---: | ---: |
| 0 | 20 | 234.04133745316253 | 0.3064299945198928 | 162.32405169996156 |
| 1 | 20 | 265.81984108629655 | 0.4120577530171025 | 156.2867146609139 |

两根残差能量与投影外比例不同，但两项相乘后的w接近；不能仅因chi相差约0.106便推断分配权重差很大。此w优先可解释的绝对残差能量，也不等于每多评价一个随机候选的期望改善。

U/N本来就固定10/10；U虽共享谱成本也不按w分配，N完全不使用w。D/S真正经过权重公式后也落到10/10，继而四策略各根都评价同一个bank的index0..29，独立执行相同顺序并稳定argmin。每策略完整120search+8replay；共享8只计一次，全panel确为520predict/AE、2J、384方向/grad、2SVD。不能因命令重复追认去重或退款。

实际slot0八臂表内四个winner都是index1（原GN锚点），model Q `0.6003295827228584`；slot1四个winner都是index2（原D2P锚点），Q `0.1660668051225332`。四种策略各自付费评价/双重replay后没有找到胜过这些共同锚点的候选。每根四分支的literal commands与真实终态逐位相同，8个物理分支确实独立执行，各full10；不是少派发。0/8 native success不构成P3或组合理论负结果。

## 2. 取整为什么消除了这次调度差异

令 `p=w0/(w0+w1)`。实际：

```text
w0/w1 = 1.0386298800390457
p      = 0.5094744711674455
1-p    = 0.4905255288325545
(w0-w1)/(w0+w1) = 0.018948942334890993
```

原20-block保底/可分配公式的连续值：

```text
xD0 = 20*(.25+.5*p) = 5+10*p = 10.094744711674455
xD1 = 20-xD0                 =  9.905255288325545
xS0 = 5+10*(1-p)             =  9.905255288325545
xS1 = 20-xS0                 = 10.094744711674455
b0  = floor(x0+.5), b1=20-b0
```

D与S都变成 `[10,10]`。解析地，D均分当且仅当

```text
9.5 <= 5+10*p < 10.5
 .45 <= p < .55
9/11 <= w0/w1 < 11/9       （两w均正）
```

S均分要求 `.45 <= 1-p < .55`，所以**D与S同时均分的区间是 `.45 < p < .55`**。端点不能偷写成闭区间：p=.45时D均分而S取11/9；p=.55时D取11/9而S均分。half-up加“第二根取余”的固定规则不完全交换对称，恰在tie处不能简单把两根都各自round后相加。

实际p在开区间内部，离最近边界 `.55` 仍有 `0.0405255288325545`；连续block偏移只有约0.0947，远不到0.5个block的取整门槛。这是明确的**有限分辨率/小权重反差造成的策略碰撞**，不需要诉诸CUDA漏算或物理不确定性来解释。它也不是精度误差：相差约4个百分点才触边，不是1e-15级浮点抖动。

由于每block含3候选，本次连续理想候选数为 `[30.284234135023365,29.715765864976635]`。仅将每block改成1候选、总候选仍60，并保留相同连续比例及half-up：

```text
y0=60*(.25+.5*p)=15+30*p=30.284234135023365
n0=floor(y0+.5)=30, n1=30
```

仍然均分。单候选粒度下D/S同时均分区间收窄为 `29/60 < p < 31/60`，实际p依然位于其中。因此“减小block就会在这两根分出不同预算”不成立。不能看到当前结果后提高gain、改温度、改Q、放大w差或强制31/29，再把新结果追认成原预注册。

## 3. 当前证据能识别与不能识别的量

能识别：本实现正确支付共享谱开销；信号不是零；原整数分配对这两根恰好不改变搜索内容；已实施四策略在本轮给出相同winner和相同物理后果。D−U/S/N的真实描述差为0，native metrics不是model Q的量纲。相同任务与相同输出下，N逻辑独立部署128predict，U/D/S各136，且后者另有AD/SVD；这些成本轴不能免单。这只是当前已用两根、已实施策略的事实，不是总体期望优势/劣势。

不能识别：如果真的把更多候选分给某根会怎样；w能否预测候选边际价值；排序不同的前景任务上D是否优于swap；是否有算力/壁钟优势；是否完成原C3的任务内progress/reconfigure/horizon调度。不能用本次“没有实际处理差异”的对照去估计“处理差异有效时”的物理效果，也不能把它从总体分析中删掉以制造效果。

即使下一轮产生29/31，两根仍共享index1/2强锚点；减少的一条尾候选未必是winner，增加的候选也未必胜过锚点。**配额不同不蕴含winner不同，winner不同不蕴含真实任务收益。** 已保存bank中的未评价位置不得看未来成本来挑配额；本文件未评价任何新候选。

当前comparison v1的actual same-Q/RGB列仍为missing；后续独立RGB adapter必须有自己的明确schema/source/prepare/report SHA，新版本比较器才能消费，不能在旧报告中回填。原off目标也不能根据overlay诊断结果事后切换。

## 4. 下一轮最小候选协议：单候选粒度、无偏随机整数化

**下面仅供MAIN选择并另行预注册，不是已经批准、实现或执行的v2。** 其动机来自已知离散化机制，所以当前两根再跑只能是探索/工程校准，不可称blind prospective复验。

保留原off任务/Q、包域、原normalizer、完整H2、两根全部纳入、原zero/GN/D2P锚点与固定raw bank规则；不训练、不拟合Q、不用新物理/latent outcome配额。只改变整数化接口和调度单位：

```text
每方法仍60个fullH2 candidates，root候选索引是固定bank前缀。
U/N固定30/30；N仍无信号入口。
D: y0=30 if sumw==0 else 15+30*w0/(w0+w1)
   a=floor(y0), t=y0-a
   nD0=a+I(u<t), nD1=60-nD0
S: nS0=60-nD0, nS1=nD0   # 预声明配对反向整数化，不交换任何root/bank/goal
```

`u`由单独PCG64流产生，建议在**新的预注册**中固定 `seed=20261028+104729*panel_id`、每panel恰好一次 `random()`；所有panel_id、seed与该u的float64 bit先封存，绝不看u能否造成差异再换seed。u与候选bank流分离。不因sumw0或整数y而重抽，u照样保留；t0自然无额外候选，root候选数恒在15..45、两根总60。D的期望候选数为y0，S为60-y0。S明确使用互补整数配额这一**新耦合规则**，不能冒充原v1在half-up tie时也如此。

执行固定U/D/S/N顺序；每方法先交替评价slot0/1各15个候选，再轮询剩余整数配额，耗尽即跳过，不按cost/物理质量更改顺序。全部H2完整支付，不按第一包剪枝。各方法独立评分/argmin与两次fresh replay；相同raw也不去重。曲线只在已付费30/60个candidate边界报两根best-so-far，不用插值冒充新调用。

对当前保存p做**解析演示**，该新规则产生D `[31,29]`、S `[29,31]` 的概率约0.284234135；其余约0.715765865仍全部30/30。本文件没有抽这个u，更没有挑一个恰好造成差异的seed。允许并完整报告碰撞，不能“如果碰撞就再跑/换根”。若要求每次强制非均分，那是另一个rank-only/强制扰动算法，不是无偏整数化，需要单独理论与全额付费对照，不能偷塞进本方案。

## 5. 预算、对照、冻结与可执行前置

每个新panel若获批，仍是原完整资源上限：

| 路径 | 每panel实际完整预算 |
| --- | ---: |
| 两根nominal H2 + fresh full joint AD H2 | 8predict/AE |
| 四策略各60candidate H2 | 480predict/AE |
| 四策略各两根whole+segmented独立replay | 32predict/AE |
| 合计 | **520predict/AE/predictor/pred_proj；1560 action context tokens** |
| shared AD/SVD/new solve | **2J、384数学方向/grad、2SVD、0solve** |

per-method128，per-root `2*n+4`（最大94）、shared每根另4；U/D/S独立部署逻辑136与N128仍不等总资源向量。原编码8/8和GN/D2P锚点288predict/完整原AD账均须有实际receipt；新任务上重新生成锚点属于**新取得资产成本**，不能套用当前两根“历史已付”标签免单。600s/3GiB allocated/4GiB reserved、16MiB output/200MiB floor保持候选上限；超限保留失败，不能通过减调用、降精度或padding伪造match。

MAIN若要把此问题从工程校准推进到一个可解释的前景比较，建议另行固定**8个全新two-root panels**作为最小候选队列（数量是本次提案，不是事后已批准额度）：独立root recipe/seed列表、目标构造规则、表示、锚点生产器、bank seeds和rounding seeds全部在模型/物理结果前封存；同根不同rounding seed不是独立任务样本。保持全部panel、不得按w反差筛入，也不得因碰撞/native失败补根；若source无法生成某固定panel，登记失败而非补到8个有利panel。

这个队列仅cached planner部分的完整上限是4160predict/AE、16J、3072方向/grad、16SVD；fresh编码/锚点/物理/RGB另封完整向量，不能把4160当端到端总成本。该额度不是执行授权；当前没有为其创建runner、source资产或manifest。

预声明的主要科学终点应是另行批准的真实native成功/真实终态指标，按panel保留全8臂分母，并将model Q单列为机制量；原native reward含velocity但success无velocity门，wrapped指标不替换native。D−U为主，D−S、D−N为次；先报逐panel/根结果，再报固定队列平均，不以只保留“有效操纵”panel作为主分析。描述性机制终点固定为：非均分率、D/S实际配额差、winner碰撞率、literal commands碰撞率、full10率、完整成本向量。若前景来源/独立性或P1/适用P2前置尚不足，只允许明确授权的工程test，不能开放P3/full C3 gate。

执行前尚缺且不得假装具备的材料：MAIN批准新整数化/队列协议；新盲task/goal来源及独立性说明；全部前景source/encoder/anchor receipt与外部SHA；新scheduler source/tests/doc/prepare封存；按新schema的physical及RGB消费者；实际设备/磁盘/墙钟授权。当前只有这个可实施数学和接口提案，不是这些资产已经存在。

## 6. 保留的v1交接

comparison source/tests/innerdoc的冻结SHA仍分别为：

```text
977cb1e991ecf95752ea9c7d568b53f076c116ad9639be6faec657338989a55b
25abcf8d00847452859c7ba92319480dba21207686ec1a211691f4fef52c7c41
c3b913991f12ca0ba50a7f0899122a859b0024335ff59f2eaafdaf22a22abda8
```

交付时38项RAM/mock测试通过（不是本文件新增实验）；完整CLI在 `innerdocs/moving_budget_scheduler_comparison_20261027_v1.md`。saved01已存在且不可覆盖，复算输出应选全新saved02等目录。本文只新增这个Markdown并做saved算术复核，没有重跑神经/物理或修改任何冻结字节。
