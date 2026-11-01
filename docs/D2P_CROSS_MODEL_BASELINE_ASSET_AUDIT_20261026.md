# D2P跨模型/分层基线：身份、协议与资产审计

日期：2026-10-26；总体状态：partial。新增覆盖为Hierarchical Planning with Latent World Models、Learning Goal-Conditioned Subgoals through Frozen Hierarchical World Models、Control-Geometry Straightening、Dual-WM、Slot-JEPA。本轮只新增本文件，不改其他worker的基线报告、runner、预算准入或阶段代码；不拉大权重、不启动GPU。

## 来源与阶段编号合同

新附件 `/home/gbk/.codex/attachments/380fd905-cc8f-4924-a507-d9292f925635/已粘贴的文本.txt` 是检索清单与待证主张，不是外部数值的一手出处。原提案C1–C3、实验1–3、P0–P5范围保留。附件末尾“最小预算方案P0/P1/P2/P3”仅表示整理文献、同模型对照、公开协议与后续跨模型的工作排序，本文称“预算路线B0–B3”；绝不替换原研究阶段定义。

尤其：B0整理公开结果不等于原P0通过；B1比较composition不能绕过原P1非贪心重配置kill experiment；B2复用SAGE/AP协议不自动开启原P2；B3跨模型效能仍受原P1及各模型自己的前置准入约束。只读资产审计可以独立并行，模型试跑/正式跨模型P5不能偷跑。

证据标签：`cached_primary_text`表示已读取带论文身份的本地原文转存；`official_search_metadata`表示本轮官方arXiv检索返回身份/摘要，但不是已核验全文表；`advertised_link_only`表示论文提供链接但未证实可下载；`unverified`不是0、无资产或论文不存在。

## 身份核验矩阵

| 候选 | 已核实的身份/一手定位 | 版本、表页、作者边界 |
| --- | --- | --- |
| HWM | Wancong Zhang、Basile Terver、Artem Zholus等，*Hierarchical Planning with Latent World Models*；[arXiv:2604.03208v2](https://arxiv.org/abs/2604.03208v2)、[HTML](https://arxiv.org/html/2604.03208v2)、[PDF](https://arxiv.org/pdf/2604.03208v2)；读取 `/home/gbk/.local/share/arxiv-mcp/2604.03208v2.md` | `cached_primary_text`；转存头部标v2、16 Jun 2026，正文date另写August 24, 2026，两者并存，不据此猜测最新修订。PDF页码未核实；以下表号按实际转存正文 |
| Frozen HWM Subgoal | *Learning Goal-Conditioned Subgoals through Frozen Hierarchical World Models*，官方检索作者写Xiang Kaibo、Liu Qingfeng；[arXiv:2610.26179](https://arxiv.org/abs/2610.26179) | `official_search_metadata`；检索页显示v1提交19 Oct、更新20 Oct 2026；具体当前版本号及PDF全文未核验。附件称Table 2及误差定义均待原文确认 |
| CGS | Ziang Fu、Ning Ning，*Control-Geometry Straightening for Sampling-Based Latent Planning*；[arXiv:2609.35603](https://arxiv.org/abs/2609.35603) | 官方检索：v1 28 Sep、更新7 Oct 2026。版本号、正文表页、协议全文仍`unverified`；不将试探的v2 URL当作已确认版本 |
| Dual-WM | Delin Zhao、Zhengrong Yue、Shaobin Zhuang等，*Beyond a single latent space: a dual-latent world model for long-horizon planning*；[arXiv:2609.37644](https://arxiv.org/abs/2609.37644) | 官方检索显示29 Sep 2026提交；完整版本、表页、PDF未核实。Dual-WM是该文方法名，不以简称定位其他同名模型 |
| Slot-JEPA | Peisong Wen、Qianqian Xu、Minye Lei、Siran Dai、Yang Liu、Qingming Huang，*Slot-JEPA: Object-Structured Prediction Interfaces for Latent World Models*；[arXiv:2610.28074](https://arxiv.org/abs/2610.28074) | 官方检索显示20 Oct 2026提交；全文、版本号及表页未核实 |

四项新ID来自本轮成功访问的[官方组合检索](https://arxiv.org/search/?query=%22Learning+Goal-Conditioned+Subgoals%22+OR+%22Control-Geometry+Straightening%22+OR+%22Dual-WM%22+OR+%22Slot-JEPA%22&searchtype=all&abstracts=show&order=-announced_date_first&size=50)，不是猜测或附件补值。后续全文和metadata请求的自动审批超时；不能把已确认身份扩张为已确认所有实验或资产。

## HWM：已核读的协议和原文数值

这是一项跨backbone的分层训练/规划方案，不是同一LeWM上只换planner。正文说明为不同时间尺度学习world models和macro-action encoder；不会因低层encoder冻结就自动符合D2P“不额外训练模块”的约束。

| 原文位置 | 任务与划分 | 原报告对照/结果 | 与附件的区别 |
| --- | --- | --- | --- |
| §3.2 Table 2 | Push-T；validation trajectory随机抽取start–goal pair，间隔d=25/50/75 | DINO-WM 84/55/17%；DINO-WM (hier.) 89/78/61% | d为任务起终间距，不等于模型预测horizon；这些值可作`paper_reported/context_only`，不与LeWM结果计算跨协议增益 |
| §3.3 Table 3 | Diverse Maze；25训练maps、20 held-out maps；shortest-path distance D=[5,8]/[9,12]/[13,16] | PLDM 100/84/44%；PLDM (hier.) 100/95/83% | 44→83来自Table 3，不是附件所称两项都在Table 2；maze difficulty也不是PushT goal offset |

没有把没有误差条的百分比补成SD/SEM；未核实完整evaluation seeds、各难度分母和success predicate实现，不据此宣布统计显著。本地原文中的“up to更少compute”也不等于在当前D2P实现上的等预算优势。

| 协议字段 | HWM原文已核验信息 | 缺失/风险 |
| --- | --- | --- |
| backbone/checkpoint | PushT为DINO-WM，低层25M causal ViT、冻结DINOv2 encoder；高层75M、macro-action维度4。Diverse Maze为PLDM；高层复用冻结低层encoder，macro-action维度8、stride10 | 具体发布权重、revision、任务manifest、normalizer未取得；不与本地LeWM checkpoint混同 |
| observation/action | PushT附录A.3明确RGB、2D action及agent位置/速度proprioception；5个primitive actions拼成10维packet。Maze为top-down RGB，action repeat4 | 不能只按“视觉world model”一词标为与本地RGB-only相同信息合同；动作物理单位/normalizer仍需源码核实 |
| training/split | PushT使用DINO-WM发布数据、validation trajectory目标；高层额外学习。Maze 25训练/20 held-out布局，训练每map2000条100步轨迹、5M transitions | Maze正文含10×10 grid描述，附录另写8×8 excluding outer wall；保留原文口径，不擅自消除。seed列表与精确数据revision未核验 |
| horizon/feedback | 两层receding-horizon MPC，PushT用CEM，Maze用MPPI；task gap d/D与pred H/h分开 | k的部署时间/低层执行频率应由最终源码明确；不只凭同名horizon判兼容 |
| metric/seed | Tables 2/3报告success rate (%)，各任务按其原协议 | 本轮没有确认评价seed列表、bootstrap/误差条或统一阈值，不能补造 |

原文附录B给出预算，不是只用“候选数相同”判断公平。以PushT d=75为例：

| 原表/层级 | elites | iterations | samples/update | prediction horizon | k |
| --- | ---: | ---: | ---: | ---: | ---: |
| Table 11，flat | 20 | 20 | 1000 | h=15 | 5 |
| Table 12，hier high | 10 | 20 | 1200 | H=5 | 不在该高层列给出 |
| Table 12，hier low | 10 | 20 | 1200 | h=5 | 5 |

这些来自原文转存，PDF页码未核实；未将高低层候选数直接加成D2P的predict samples，未补反向/编码/墙钟数据。Maze困难D=[13,16]的原文Table 13 flat MPPI为samples250、pred h250、k4；Table 14 high为samples4000、pred H47，low为samples1000、pred h15、k4。这是不同潜在对象/预测尺度，不能把数值大小当成调用数或总FLOPs比较。

## 其他四项：协议确认止于官方摘要，附件表值仍待全文

| 工作 | 官方检索摘要确认的方法/任务事实 | 仍未确认的字段与可比性 |
| --- | --- | --- |
| Frozen HWM Subgoal | 冻结两层HWM，通过冻结high-level predictor训练轻量goal-conditioned condition网络；hindsight goals/state-prediction losses，无search-solution/macro-action标签；推理保留高层自回归但无高层候选优化。摘要指H-JEPA共享latent checkpoints、3个backbone seeds×每backbone3个head初始化；任务Cube、PushT、FourRoom、Visual AntMaze | 具体task sets/splits、checkpoint路径、观察额外状态、动作layout、horizon/执行/计数预算、seed ID与误差条定义、Table 2原值均`unverified`。不能把“50-task sets/标准误”从附件当原文事实；训练head与D2P主线不同 |
| CGS | 为representation增加control-geometry auxiliary loss，用局部pixel-action pairs匹配动作和latent differences的pairwise cosine；跨架构、MPPI/CEM/GD，摘要明确涉及128 candidates/update | 修改训练目标，不能作为同一个冻结LeWM上零后训练planner替代；Task/split/checkpoint/observation/action/horizon/iterations/metric/seed及附件Table 1数值均待全文核验。128候选不足以建立等预算 |
| Dual-WM | 双latent分别服务local execution与long-range planning；高层learned macro-actions、LoRe weighted rollout；from-scratch训练、5项goal-conditioned visual tasks、goal offsets50/100；摘要称不使用actor-guided proposals的task-wise strongest比较 | 不是LeWM换planner；具体5任务名、split、额外观察、checkpoint、packet、horizon、执行、预算、seed及success predicate均未核实。摘要宏平均不转录成任务表，不与当前P0/P1均值互减 |
| Slot-JEPA | pretrained decomposition分离稳定semantic slots和time-varying masks并在downstream冻结；TMP用transition memory，UOT用于mask预测监督和计划评分；摘要提6个planar与REALM-JEPA的4个robot-arm任务 | 不是单一dense latent可直接调用的LeWM predictor；任务/split、动作条件与history、slot/mask排列、Jacobian可微链、UOT求导、权重、预算、seed及metric均待全文/代码确认。不能因“对象结构”直接说已解决PushT任务代理错位 |

新附件所有四项百分比、±、表页和性能归因仍保持`attachment_claim_unverified`，未复制进已核验数值表。官方摘要存在某个相同汇总值也不自动核实附件的表号、误差定义或完整协议。

## 资产可用性矩阵

“有论文/有代码链接/可获取权重/可合法使用/接口可加载/已验证效果”是六个不同状态，不能逐级自动放行。

| 工作 | 官方明确给出的资产线索 | 本次实查 | 当前准入状态 |
| --- | --- | --- | --- |
| HWM | 原文Code：[kevinghst/HWM_PLDM](https://github.com/kevinghst/HWM_PLDM)；[项目页](https://kevinghst.github.io/HWM/) | 确认原文链接；GitHub repo元数据请求审批超时。未克隆、未下载权重，未读manifest/license/release | `advertised_link_only`；不能标“权重可用”或“可复现”。PLDM仓库名也不证明DINO-WM/V-JEPA2三套资产全部发布 |
| Frozen HWM Subgoal | 摘要指向H-JEPA分发的shared-latent HWM checkpoints，但未得到唯一官方repo/权重URL | 论文元数据/全文请求审批超时；没有自行猜H-JEPA域名或repo | checkpoint/head权重、dataset manifests、license、代码commit全`unverified`；不能拿另一个HWM架构代替 |
| CGS | 本次成功返回的官方摘要未提供已确认代码/模型链接 | 全文/metadata后续访问未成功，未下载资产 | 官方代码、训练配置、CGS权重和许可`unverified`；不是“无公开资产”的断言 |
| Dual-WM | 官方摘要明确[DeLin1001/Dual-WM-Official](https://github.com/DeLin1001/Dual-WM-Official)，描述为core implementation | repo元数据请求审批超时；没有验证默认分支、release、权重或模型卡 | `advertised_link_only`；core code≠五任务checkpoint齐全 |
| Slot-JEPA | 成功读取的官方摘要没有唯一代码/模型URL | 后续全文/metadata请求审批超时 | slot decomposition、TMP、UOT配置、REALM-JEPA任务资产、checkpoint/许可均`unverified` |

本轮没有下载任何权重、没有导入神经模型、没有CPU神经计算、没有GPU调用。只读检查不存在“顺便试载”的授权；需要下载大权重、接受许可或新增预算时回主线程/用户处理。

## 对当前P0/P1与后续复制的具体意义

1. HWM可作为已确认的**跨backbone分层性能参照**，但其DINO-WM+proprioception / PLDM协议不等于本地LeWM RGB合同；可引用原表，不用于本地配对显著性或等预算排名。
2. Frozen HWM Subgoal属于**额外学习高层模块**的比较候选；CGS/Dual-WM/Slot-JEPA属于不同训练/表示/模型接口的复制候选，不是当前冻结LeWM P1已经运行的强对照。
3. 在任何模型上应用D2P前，至少需要合法资产、精确版本、观察/动作/normalizer与因果context、可微multi-step rollout、AD-FD与真实回放合同、任务度量、冻结参数/buffer及完整资源记账。仅import/fixture/单测成功不证明模型具有所需物理能力。
4. 当前Reacher模型条件队列不足预定8根、PushT suffix known4的物理收益未成立；外部公开成功率不能补齐本地非贪心→未来真实能力改变→任务收益机制链，也不能开启P2/P3/P5。
5. 安全并行工作：核验未取得的原文/资产许可、整理同模型默认planner接口、记录独立root与公开manifest所需字段。暂不创建新实验、不训练condition head、不改变checkpoint、不使用privileged状态挑根或调门。

## 实际验证与剩余阻塞

已完成：全文读取新附件；定向检索已有arXiv缓存；核读HWM v2指定正文与附录表；成功官方组合检索确认其余四项唯一身份/摘要；核对原文链接与尚缺资产状态。未完成：HWM仓库/权重实际可用性、四项全文与表页、各方法完整公开protocol和许可。本页完成的是审计登记，不是全量核验完成或研究完成。

网络按正常审批访问：多项后续请求返回自动审批评审超时，这不是404、撤稿、论文不存在或操作不安全的证据；没有使用替代连接器、代理或下载权重绕过权限。来源核验受限时保留`unverified`，不凭附件/记忆填数。
