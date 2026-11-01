# D2P 原始范围的最短科学关键路径（2026-10-26）

## 结论与审计边界

研究尚未完成。最短路径不是再增加诊断数量或先包装论文，而是先在合格任务族上完成正式 P1：同一根上同时验证非贪心前缀、后继真实任务能力改变、重新编译后缀的真实收益，并在独立 outcome 中战胜资源匹配的强对照。P1 之前不打开 P2/P3/P5。可以并行消除输入合同、基线和统计设计的前置障碍，但不能并行绕过科学依赖。

本页以接近 CVPR 时提高并行效率为目标，不假定任何截稿日期，也不保证录用。原提案的 C1–C3 和实验1–3全部保留，不缩成 paper-only proxy。原附件只明确命名 P0/P1；P2–P5沿用当前 closure audit 的后续阶段定义，不反向声称附件逐项规定了这些编号。预算路线 B0–B3 不覆盖研究阶段 P0–P5。

审计依据：

- 原始方案：`/home/gbk/.codex/attachments/861f6764-1b25-4227-bf15-f1a237ba7d8f/pasted-text-1.txt`，尤其第4–8节。
- 最新累计闭环审计：`docs/D2P_RESEARCH_CLOSURE_AUDIT_20261025.md`（文件名为25日，正文已累计26日更新）；其“尚缺能力探针”等历史段落须由页首较新结果覆盖。
- 当前结果均以 `le-wm-main/` 为 Python 工作目录；下文 `results/` 相对此目录。历史测试数量引用文档，不是本轮重跑证明。
- 当前独占更新仅涉及本文件及新增 `docs/D2P_BASELINE_PROTOCOL_FIELDS_20261026.md`；历史单文件边界见末尾记录。不启动实验、神经计算、训练、adapter/readout拟合或GPU作业；不改其他源码、阈值、根、旧结果，不提交、推送或删除。GPU继续由parent协调。

## 当前实物证据：哪些完成了，哪些没有

| 证据锚点 | 本轮核查/累计审计中的事实 | 可支持与不可支持的结论 |
| --- | --- | --- |
| `results/pusht_d2p_nuisance_suffix_bank4x16_v1_20261026_cuda/report.json` | 当前 `status=completed`；11项gate全true；3968 predict、128 physical branches、64 root/proposal pairs；`P1_admission=false`。parent报告exit0 | CUDA suffix作业已完成，不再标running；这是完整性完成，不是P1成功 |
| `results/pusht_d2p_nuisance_suffix_bank4x16_analysis_v1_20261026.json` | 独立分析已完成，64 pairs/128 branches，`new_inference=0`、`P1_admission=false`。nuisance−identity物理pose cost均差+2.930957：13改善/23恶化/28相同，四根均值全部恶化；actual common-Q cost均差−0.063998 | 本批共同Q代理改善没有转化为物理pose收益，不能称suffix修复有效或晋级P1。四个已知校准根不是64个独立根；这不是理论负结果 |
| 同一suffix报告的protocol | `identity_gn4`与`nuisance_gn4`；全部16前缀固定，只重编译观察RGB后的suffix；显式goal-set输入 | 尚非prefix/stage/own-state全编译器修复；额外goal集合不能冒充单目标RGB合同；两臂各自度量下降不能直接互比为任务收益 |
| 上述独立分析的限制 | 未存Jacobian/SVD/GN中间向量或原参数，不能独立重算方向；tanh饱和限制正则能量重建；未存物理goal state，绝对pose距离仍为收据；CUDA/冻结/计数不是独立运行时trace | 完整性、保存数组可复算和完整算法验证须分账，不能写“所有实现正确” |
| `results/reacher_d2p_model_escape_pool32_analysis_v1_20261026.json` | `eligible_count=1`、`selected_seeds=[]`、149184 predict、0物理未来、`P1_admission=false` | 固定32根模型筛查没达到预定8根；任务前提未建立，不是物理P1反例；不补根、不放宽阈值 |
| `docs/D2P_GOAL_METRIC_PROSPECTIVE_CONFIRMATION_20261026.md`及closure页首 | 完整Q编译器805536预测/912物理分支；三任务模型见证均0/8；能力探针58单元/3116未来及独立拟合已完成，仍无同根完整关节机制合取 | 不能再把“补测能力”泛写成尚未执行；缺的是合格机制证据，不是测量计数 |
| `docs/D2P_MECHANISM_CONTRACT_AND_MODEL_COHORT_20261026.md` | own-state residual与共同残差增益合同确有区别；696新拟合仍未恢复完整见证 | 不把更强诊断的否定等同原提案被否定，也不靠改诊断门放行 |
| `docs/D2P_PUSHT_ACTION_STATS_AND_OBJECT_RESPONSE_20261026.md` | 192分支已复算；三suffix中51/47/40条物体终点精确不变；封存4个selected suffix均未改变物体；动作归档统计已完成 | RGB距离下降可只反映推杆，不能代替物体任务成功；无需把这批旧诊断再排成待跑实验 |
| `docs/D2P_PUSHT_GOAL_AGENT_INTERVENTION_20261026.md` | 目标推杆干预与候选Q敏感性已完成；44编码/9门，物体能量保留最差0.726814，未见推杆位置中位0.002828 | 支持继续检查有限span度量，不证明所有物体控制信息保留或真实效能恢复 |

精确normalizer溯源仍缺官方权重到训练数据/统计的绑定。归档std与公开示例约3.3%差异不是替换常数的授权，也不证明它解释剩余失败。旧Cube/repair/teacher结果不补training-free主线；旧PointMaze-SRA的P1不等于D2P-P1，TwoRoom输入/FD合同通过也不等于物理效能。

已核实的suffix判定：分析器明确按 `nuisance_minus_identity` 计算差值，pose cost越低越好。四个根的pose均差依次为+6.862024、+0.143690、+0.776308、+3.941808，因此不能用64条候选总体common-Q下降掩盖每根平均物理恶化。这是已知校准根、固定前缀、suffix-only、额外goal-set输入条件下的描述性结果，不是独立outcome、完整编译器或D2P理论的否定，也不证明恶化的唯一原因已经定位。物理绝对距离仍受上述goal-state保存限制约束。

## C1/C2/C3与实验1–3的完成矩阵

“实现/局部证据”不记为科学完成；每行都需要自身的独立证据，不能靠另一个同名分支借位。

| 原始目标 | 当前完成层级与证据 | 最小缺失前提/交付 | 依赖与完成判据 |
| --- | --- | --- | --- |
| C1 Dynamics-Native Action Abstractions | action-Jacobian、task-weighted SVD、有界DLS、动作见证及重观察接口已有；运行时/归一化/后缀合同多处已修 | 对最终冻结接口绑定context、目标Q/R、动作尺度、动作见证与预测后继；独立校准/outcome；物理实现误差及强基线 | P0及实验1；不能只给低秩优化收益就声称程序合成优势 |
| C2 Reconfiguration-Aware Program Synthesis | 两段程序、有限动作词、全库物理探针与独立拟合存在；仍无稳定同根完整机制 | 先建立模型具备相应动力学的controllability-gap任务族；再冻结选择器、非贪心/真实能力/后缀终态指标，独立验证全部合取 | P1为kill experiment；效益必须来自实际封存选择，不是事后库中好候选 |
| C3 Model-Native Test-Time Planning | dispatcher与predict/VJP账本存在；当前全bank逐根编译不等于动态预算 | P1后由动力学信号决定progress/reconfigure、深度/horizon/预算；同预算固定策略与随机调度消融 | P2/P3/实验3；不把手工阈值称最优/安全保证，不训练router |
| 实验1 Executable Subgoal Gap | dynamics-generated中间状态已有；未形成三臂正式比较 | arbitrary latent waypoint、已有learned subgoal/proposal、dynamics-compiled program；相同任务、低层controller、信息及资源合同；记录实现误差、偏差、最终success | 合同/资产可独立准备；正式D2P组合效果比较等P1/P2。缺learned prior不能改名为随机基线充数 |
| 实验2 Controllability-Changing Actions | 大量Reacher/PushT诊断和完整能力测量已执行；不构成完整C2 | P1机制与正式等资源强对照：采样、梯度、GN、GRASP、无组合SRA、kinodynamic；独立闭环任务成功/成本 | 最先补的决定性科学证据；只有终态更好或rank变大均不足 |
| 实验3 One Frozen Model, Many Tasks | 同一Reacher checkpoint的全驱动/肩/腕任务与统一目标接口存在，结果不稳定 | 同一模型切换目标、horizon、动作限制；真实success、长时任务、耗时、forward/backward；独立根和失败保留 | 可先冻结任务/评价合同；P1后完成组合、调度和多任务效能。不同checkpoint各跑一个任务不能替代同一模型多任务 |

## P0–P5阶段矩阵

| 阶段 | 已有证据/状态 | 尚缺的最小科学里程碑 | 现在是否可独立推进 |
| --- | --- | --- | --- |
| P0 谱编译与执行前提 | 有历史正回执、source-native/FD/精确回放；不同接口不能共享pass | 最终接口逐项绑定CUDA/冻结、AD-FD、action/context/normalizer及完整执行；原提案随机/CEM/GN/GRASP等预算物理比较仍须补齐 | 可做只读合同、预算及资产审计；实际CUDA由parent决定 |
| P1 非贪心重配置 | 当前未通过；Reacher模型任务准入为空；PushT新suffix完整性与独立分析完成，但pose均差+2.930957、四根平均均恶化，不构成修复 | 合格任务族→封存全部候选/选择→独立calibration→冻结方法→root/source-disjoint outcome；同根机制合取及强基线效益，按原门失败即停 | 当前关键路径；不能重用已看结果的校准根当新outcome |
| P2 多程序组合 | prefix-observe-suffix与动作词原型不是完整组合收益 | P1后最小多程序搜索，逐边动作见证和预测postcondition；超过无组合SRA/同预算平坦搜索 | 阻塞；本轮不开发、不运行 |
| P3 动态horizon/算力 | 旧SRA局部证据不可迁移 | P1及必要组合后，冻结几何调度与固定预算对照；horizon、噪声、开销消融 | 阻塞；可列缺失协议，不打开阶段 |
| P4 RGB闭环多环境 | PointMaze-SRA限定证据，Reacher/PushT已有分支，非完整D2P闭环 | 每环境自身D2P准入和长时RGB闭环；PushT分开接近/接触/推送/闭合及最终成功，规划器不读物理真值 | 现可准备评价/回放合同；正式Full D2P依赖P1及相应P2/P3，不借P4绕门 |
| P5 跨任务/模型复制 | 无完成原范围的training-free回执；repair资产不能填补 | 同一模型多任务与独立冻结模型/架构复制分别报告；每模型先验证自身动力学前提 | 阻塞；只读核查权重、许可、接口可并行，不启动复制实验 |

## 强基线尚欠什么

| 对照 | 当前边界 | 必须补的最小合同/证据 |
| --- | --- | --- |
| 随机、CEM / MPPI | 真正CEM-MPC与更宽CEM已有局部结果；旧只执行首段再公共DLS不算完整CEM | 预算匹配、完整执行tail/同RGB反馈、同目标/动作支持/停止；MPPI尚缺当前正式矩阵接入证据，不能将两者混写已完成 |
| Vanilla gradient MPC | 有重规划/反馈与梯度对照诊断 | 同可行域、初始化、incumbent保留、重启和资源封顶；不靠较弱求解器制造停滞 |
| Local Gauss–Newton | GN4/GN16及DLS广泛存在 | 把普通GN与程序生成/选择的增量分离；同suffix求解强度、同反馈 |
| GRASP | 当前gradient-and-restart仅GRASP-style | 官方方法/实现忠实性与依赖核查；真实GRASP公平接入，不把近似命名成复现 |
| SRA without program composition | 旧SRA结果不同合同 | 最终D2P同一动力学对象、候选支持与资源，关闭程序组合，测组合的增量 |
| Kinodynamic search | closure未有完整挂接回执 | 冻结模型内动力学扩展与可执行边；同RGB部署信息，不能给它额外物理oracle后混作同信息比较 |
| Arbitrary latent waypoint | 有概念/接口不等于实验1完成 | 同低层controller和预算，记录执行可实现性而不只latent距离 |
| Hi-LeWM / SAGE learned subgoal/proposal | 原方案要求的强比较，未有当前正式可比回执 | 优先核查已发布冻结权重、许可与兼容接口；其已有训练成本披露，不把它变成D2P训练。无可用资产且需要新训练时须用户决定，不能静默删掉 |

预算向量至少包含encoder forward、predict样本、输入/动作逻辑VJP、Jacobian/SVD/QP、候选数、horizon、真实执行长度、墙钟和缓存/goal-set构造成本。前向样本相近不等于等算力：closure历史32-root dispatcher/greedy/CEM分别约2062/142/1934 predict，却约24768/192/192逻辑VJP行。应预先冻结资源封顶与比较方式，而不是按结果调预算。物理探针属于评价开销，不能泄漏给规划器；多候选不是独立样本，统计单位按source/root配对，预先固定主要终点、区间和多重比较规则。

## 最小必要里程碑与并行前沿

依赖链：输入/度量合同与任务前提 → 正式P1机制和独立收益 → P2最小组合 → P3动力学调度 → P4长时闭环与实验1–3完整矩阵 → P5独立复制。这里是依赖顺序，不是本轮运行授权。

| 里程碑 | 最小交付/停止点 | 可并行而互不改动的工作 |
| --- | --- | --- |
| M0 suffix审计已完成，不晋级 | 保留全部4×16；common-Q均差−0.063998但物理pose+2.930957，13改善/23恶化/28相同、四根均值均恶化；不设为默认修复，不按此启动正式P1 outcome或整链扩展 | 无需继续等待分析或重复跑同批；parent协调GPU，其他worker只读消费完成产物 |
| M1 优先定位代理—物理任务脱节并明确输入/模型前提 | 先用保存数组按全部根/前缀核查共同Q、物体响应与pose变化的对应关系，区分仍待证的目标度量、动作响应和接触/horizon解释；不新拟合Q、不挑受益候选。同步明确goal-set输入是否允许、normalizer溯源及controllability-gap前提；Reacher空队列不自动扩扫 | 可独立并行做保存数组审计、normalizer证据检索、基线忠实性/资产与预算合同核查。新候选须先有具体可证伪修正及冻结协议，再由parent决定CUDA；不能因suffix失败就认定整链Q也必失败，或未经解释直接扩展 |
| M2 正式P1 | 在最终合同上重新绑定P0，按原门独立校准及outcome；非贪心、真实能力改变、闭环收益同根成立；强基线及不确定性合格 | 基线忠实性/预训练资产核查、计算账本和统计合同可提前准备；任何新CUDA由parent安排 |
| M3–M4 组合与调度 | P1成功后逐步验证无组合消融、固定预算/固定horizon对照；失败就报告局部边界，不跳门 | P1前仅记录缺失接口/评价需求，不提前启动P2/P3 |
| M5 全范围闭环 | 实验1三臂、实验2机制强对照、实验3同模型多任务，加多环境与独立模型复制 | 基线资产/许可、数据来源去重可提前核查；效能矩阵须等待各自准入 |

加速原则：优先复用已经完成的诊断，少开重复分支；每项工作按唯一文件/产物归属并行；训练修复不进入冻结主线；同一GPU只由parent调度。若M1/M2在固定合同下失败，保留研究未完成/模型条件边界结论，不靠切换论文叙事宣布原目标完成。

### 并行扩为6子任务＋主线程

用户后续明确授权：对比实验可并行推进，GPU仍由主线程协调，阶段准入不变。原论文报告数值可以直接引用，但必须标明原文来源、版本、表页与实验设置，明确为paper-reported，不冒充本地复现、相同任务合同或等预算结果。基线真实性审计可先交付原文报告值核验，不必等待本地复现；本页完成后，该新增任务仅写 `docs/D2P_REPORTED_BASELINES_LEWM_PLANWAM_20261026.md`，优先核查已有LeWM/PlanWAM原文，缺失字段显式标缺，不凭记忆填值；网络依正常审批，不启动GPU。

以下旧表保留首次六路分工的历史快照，其“本轮”“待交付”“未见接线”等均不代表最新状态；当前状态以下方“最新六路快照”为准。不是六个worker均已启动/完成的声明，不派发任务或代改实现文件。新增两条审计是子任务5、6；原预算准入、预运行封存、计数独立复核继续。

| 责任槽位 | 独立交付与边界 | 已核对状态；未验证部分 |
| --- | --- | --- |
| 主线程 parent | 合并六路交付、冻结最终合同、处理用户输入选择、独占GPU排程及正式阶段准入；汇总代理—物理脱节解释 | suffix完整性和独立分析已完成但无P1收益；P2/P3/P5仍关闭。不得以子任务测试通过替代正式科学门 |
| 子任务1 关键路径/证据登记 | 唯一写入 `docs/D2P_CVPR_CRITICAL_PATH_20261026.md`，维护C1–C3/实验1–3/P0–P5与依赖，登记完成/缺项 | 本次已更新为六路分工；只登记本轮看到的产物/源码，不推测其他worker运行状态 |
| 子任务2 预算准入继续 | 实际planning predict资源收据逐根验证，拒绝缺失/不匹配/坏单位；不混称总算力 | 已只读见 `experiments/spectral_reachability/d2p_stage_admission.py` 的 `p0_predict_samples_matched`、所需gate `actual_planning_predict_samples_matched_to_cem` 与P1入口拒绝检查；存在 `tests/test_d2p_p0_resource_admission.py`。这是源码接线证据，未在本轮运行测试或确认正式新outcome准入成功；旧文档“待实现”不能再当当前源码状态 |
| 子任务3 预运行封存继续 | 推理前封存明确的小源码清单、参数/协议与外部保留manifest摘要；结束后验证，失败目录不复用；不读大权重/全仓哈希 | 已见 `experiments/spectral_reachability/d2p_prerun_source_seal.py` 与对应测试：最多64文件、单个1MiB、合计8MiB；声明字节封存不证明执行了这些源码。本轮未在 `run_reacher_d2p_p0_goal_progress.py` 见该helper的create/verify调用；故helper存在不能登记为P0 runner已集成或已有正式预运行收据 |
| 子任务4 计数独立复核继续 | 不导入runner/helper、不运行神经计算，独立检查保存planning收据、根/方法/phase/context与单位；揭示未测资源 | 已核查 `results/reacher_d2p_p0_predict_sample_matched_calibration2_v1_20261026_cuda/independent_predict_accounting_analysis.json`：`accounting_receipts_valid`，2根20261040/41、7源gate，两根predict样本匹配但调用数不同；`efficacy_claim=false`、`compute_matched_claim=false`、`P1_admission=false`。仅收据一致性，不是CUDA真实性独立trace或总算力匹配 |
| 子任务5 新增：基线真实性审计 | 逐项核查随机/CEM/MPPI/GD/GN/GRASP、无组合SRA、kinodynamic、latent waypoint及Hi-LeWM/SAGE的算法身份、官方实现/冻结资产、信息访问、反馈与执行tail；给出真实复现/明确近似/缺失证据表 | 当前已验证缺口仍是GRASP-style≠GRASP、随机单程序≠预算匹配随机搜索、learned prior与kinodynamic未有完整挂接回执。`docs/D2P_P0_PREDICT_SAMPLE_MATCH_20261026.md`也明确这些限制；本轮未核实新专属真实性审计交付，不能标完成。只读审计不授权新训练或下载/运行 |
| 子任务6 新增：P1剩余代码假设审计 | 用源码与保存数组逐条列“假设→支持/反证→尚缺观测→最小可证伪检查”：目标Q全链一致性、own/common residual、normalizer溯源、动作单位/布局、因果history、suffix warm-start/incumbent、反馈执行、物理goal/pose指标绑定；不拟合、不调门、不筛根 | 已有归一化/目标拓扑/后缀/渲染等修复及suffix代理下降而pose恶化证据；仍不能证明代码原因穷尽或某个未测假设为根因。本轮未核实新专属剩余假设审计交付，状态为待交付证据，不是模型/理论负结果 |

推进次序：子任务2/3/4并行补齐“资源准入—预运行封存—独立计数”的实现证据；子任务5独立防止弱/伪强基线，子任务6独立缩小P1代码/合同假设，子任务1只登记。主线程先汇总这些缺项，再决定是否已有具体可证伪修正值得冻结新协议；不因六路并行就打开P1 outcome或后续阶段。各路必须区分源码已存在、测试已报告通过、真实运行收据、独立数组复算和科学效能；任何一层不能自动升级下一层。

### 最新六路快照（覆盖上表历史状态）

下表按当前文件证据更新；不是六个进程均在运行的声明。科学 P1 仍未通过，不靠累计测试数晋级。

| 槽位 | 当前已核实状态 | 下一交付与依赖 |
| --- | --- | --- |
| 主线程 parent | 汇总协议、GPU与用户选择；sampling两次host审批超时见 `D2P_PARAMETER_RECEIPT_REPAIR_20261026.md`，无新真实校准结果 | 先验收具体修正及冻结协议，再获得有效执行批准；不能因基线测试通过开启P1 outcome或P2/P3/P5 |
| agent 1 文档/协议字段 | 本页及新增 `D2P_BASELINE_PROTOCOL_FIELDS_20261026.md`；仅文档，不改注册器 | 可独立完成字段建议、原范围矩阵及证据登记；交agent 5/主线程采用 |
| agent 2 预算准入/sampling runner | `run_reacher_d2p_sampling_calibration.py` 已存在但真实runner **未执行**；本轮 `ls -ld` 确认 `results/reacher_d2p_sampling_calibration2_v1_20261026_cuda` 不存在。133/33候选、266/66/256预测样本为fixture/预期而非实测 | 等成本域修复、坐标/反馈/资源合同验收和host批准；不自行GPU运行。该校准不是自动等预算，更不是P1 |
| agent 3 预运行封存 | 新sampling runner已导入并调用 `create_seal/verify_seal`，明确模型分配前封存、结束后校验；不能追认旧P0 runner集成或未运行的收据 | 可独立核对小源码清单和参数绑定；实际封存收据依赖执行。源码修改后必须重封存 |
| agent 4 独立计数 | 新 `audit_reacher_d2p_sampling_calibration.py` 已存在；旧两根P0收据审计只证明一致性。新sampling无真实输出可审 | 可独立准备缺失/失败规则；等真实收据后核对逐根、方法、phase及单位，不把synthetic测试当真实运行 |
| agent 5 基线真实性/文献 | `D2P_BASELINE_FAITHFULNESS_AUDIT_20261026.md`、`D2P_SAMPLING_BASELINE_CONTRACT_REVIEW_20261026.md` 已交付；文献系列与AP/GRASP/Hi-LeWM/SAGE/PRISM/IMWM核验仍 **partial** | 用已有原文完善字段、资产及忠实性，不重试限流网络。random-policy≠shooting，restart-Adam≠GRASP，clipped-Gaussian heuristic≠faithful MPPI；paper-reported不替代本地配对机制比较 |
| agent 6 P1代码假设/修正 | `D2P_P1_REMAINING_CODE_HYPOTHESES_20261026.md` 已交付，其“未证实明确bug”仅限该审计范围。另一sampling review明确的 **complex-cost丢虚部bug已交实施**；当前内核/桥已有实数dtype拒绝逻辑，属静态修复证据，本轮未运行验收。PushT **fixedbank metric caller正在实现**，参数修复记录已登记，未核实完整caller运行收据 | 每prefix自己的预测状态重算suffix nominal/Jacobian，统一stage/own-V/非线性终态度量及参数见证；依赖坐标/输入合同。实现完成不等于完整谱前缀生成或P1 |

新关键路径：具体缺陷修复与fixedbank caller/信息合同 → 源码封存及资源准入 → 主线程获批真实校准 → 独立收据核查 → 合格任务前提及正式P1同根机制合取、强对照、独立outcome。sampling运行只补强基线前提；fixedbank只补已知候选的合同检查，不能替代任务族或新outcome。agent 1/3/5和agent 4审计准备可独立推进；agent 2真实运行等待合同验收及执行批准，agent 6实现可并行但不得修改运行中共享源码。P2/P3/P5科学工作仍P1-blocked，不作理论负面推断。

## 真正需要用户选择的阻塞与无需追问的工作

需要选择，但不应在本轮擅自替用户决定：

1. 正式部署是否允许显式“同物体任务goal图像集合”。当前nuisance Q依赖这项额外输入；若坚持单目标RGB，不能隐藏集合或用模拟器物理坐标生成它。两种合同各自需要冻结且独立验证，不能择优回填。
2. 若Hi-LeWM/SAGE或GRASP缺可用冻结资产，是否取得外部资产/资源，或明确接受哪个比较缺项及其结论限制。当前禁止新训练，不能自动训练补齐，也不能拿GRASP-style替代。
3. 只有在计划新增模型任务池、扩大预定筛查预算或增加独立模型复制时，才需明确额外资源/任务分布授权；现有Reacher固定32根筛查不足8根不能自动续扫。不存在“为赶截稿就放松门槛”的选择。

无需用户再选、可以安全自主推进但不在本轮执行的工作：基于已完成独立分析只读核对全部根/前缀的代理—物理响应脱节及方法局限；检查normalizer出处和公开基线资产/许可；列出最终接口绑定、根去重、算力账本、统计和评价合同的缺项。它们不需要启动神经实验或打开后续阶段。normalizer证据缺失是外部证据缺口，不是让用户凭偏好挑一个常数。

## 本轮验证记录

- 当前独占更新：只读核对sampling review、基线忠实性、剩余代码假设、参数修复记录、注册器和sampling封存接线；目标sampling目录不存在（`ls` exit2），不重试执行批准。两次host超时和caller实现中来自主线程文档/用户通知，不伪称本轮亲历进程。仅更新本页与新增协议字段文档；无测试重跑、网络或GPU。以下单文件表述均为历史轮次记录。

- 完整分段阅读715行原附件；读取累计closure及最新任务/度量/PushT审计，并对关键JSON作定向字段检查，没有加载模型或大数组。
- 定向读取suffix报告：completed、11/11 true、3968 predict、128分支、64 pairs、P1=false；读取独立分析的scope、verification_limits及计数，不据此宣布效能。
- 定向读取Reacher独立准入分析：eligible=1、selected=[]、149184 predict、0物理未来、P1=false。
- 本轮不重跑pytest、不重跑分析器、不进行任何实验；历史1591/1626等测试计数只属原审计记录，不作为本文件的新测试结果。
- 本轮唯一新增路径：`docs/D2P_CVPR_CRITICAL_PATH_20261026.md`。写入后仅作该文档的结构、路径存在性与空白检查；保留所有既有dirty变更。
- 后续本次仅更新本文件：定向读取独立分析的scope、summary、计数与限制，并只读核对分析器的nuisance−identity差值方向；验证pose均差、13/23/28计数、四根均差均正和common-Q均差。移除“等待Turing分析”旧状态，M0标完成，M1前移代理—物理脱节核查。不重跑分析器、测试或实验。
- 六路并行更新：只读核查P0预测样本匹配文档、资源准入源码、预运行封存helper及独立计数JSON；确认两根样本相等/调用不等且无效能、总算力或P1声明。对应测试文件仅确认存在，未运行；新增两条专属审计无已核实完成产物，不登记完成。仅本文件更新，不运行GPU或修改其他文件。
