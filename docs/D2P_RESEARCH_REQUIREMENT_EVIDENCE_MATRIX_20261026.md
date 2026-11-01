# D2P 原研究要求—真实证据—最小决定性实验矩阵

审计日期：2026-10-26。结论：**现有报告不足以完成原研究闭环；正式P1未通过。** 下文不是再排一轮泛化诊断，也不是以测试数衡量进度；每项标明究竟还欠哪一个可改变科学判断的真实比较。本轮只新增本文件，未执行这些实验。

## 1. 要求来源、判定口径与边界

原始依据为 `/home/gbk/.codex/attachments/861f6764-1b25-4227-bf15-f1a237ba7d8f/pasted-text-1.txt`，本轮分段完整阅读715行：§3–4定义同一有限时域动力学对象、progress/reconfiguration/composition、动作见证；§5定义几何驱动计算；§6列C1–C3；§7列E1–E3；§8只命名P0/P1。原文中的外部论文身份/成绩不是本轮自动认可的一手证据。

项目P2–P5定义来自 `docs/D2P_RESEARCH_CLOSURE_AUDIT_20261025.md` 的“原提案要求与当前证据”表：组合、动态horizon/调度、RGB多环境、跨模型/任务。它们是项目闭环要求，不反向冒充附件逐项定义的阶段。**预算路线B0–B3不是研究阶段，不覆盖或替代P0–P5。** 原提案不规定所有当前样本数/CI门；项目已有预声明准入是附加执行合同，保留不放宽。

判定分为：局部足够（只满足一个明确子要求）、不足、阻塞、不适用。没有取得新结果不等于模型负结果；已有负面/不稳定校准不等于理论否定；代码和测试存在不等于真实运行。文中“真实”指冻结checkpoint实际预测及真实环境执行/观测，不是必须新增真实机器人硬件；不新增提案未要求的硬件条件。

下列结果路径均相对 `/home/gbk/code/LeWorldModel/le-wm-main/`。证据分级：A=本轮定向读取实际JSON字段；D=本轮读取已有专属文档、未独立重算其底层数组；S=静态源码；缺失仅指下述审计范围未核实，不宣称穷尽全仓。没有加载模型、大数组/权重、计算全源哈希或访问外网。

## 2. 精确证据索引及其可用范围

| 锚点 | exact文件/字段与本轮核实 | 足够支持什么；不能支持什么 |
| --- | --- | --- |
| A0 P0校准 | `results/reacher_d2p_p0_predict_sample_matched_calibration2_v1_20261026_cuda/report.json`：`status=calibration_only`，7项`gates`均true，`stage_gate.p1_unlocked=false`；protocol为horizon2、history3、action_block5、warmup15，qpos仅评价 | 局部冻结/动作/根身份及D2P=GN实施合同；不是新outcome、P0完整强基线或P1 |
| A1 P0独立计数 | 上目录 `independent_predict_accounting_analysis.json`：`status=accounting_receipts_valid`、root_count2；20261040/41每根D2P/GN调用/样本122/536，CEM8/536；random0/0；`efficacy_claim=false`、`compute_matched_claim=false`、`P1_admission=false` | 预测样本匹配而非调用/总算力匹配；random仅单程序，不是best-of-N搜索。收据一致性不是独立CUDA trace |
| A2 完整Q编译器 | `results/reacher_d2p_full_goal_metric_compiler8_analysis_v2_20261026.json`：`independent_qp_fits=1632`、`physical_branches=912`；`aggregate.{full_drive,shoulder_only,wrist_only}.model_witnesses=0`（各8根）；`all_stage_terminal_checks=0`；相对`original_greedy`关节终距均差+0.0030100142/+0.0018082809/+0.0169331184；`P1_admission=false` | 完整Q/action witness及固定15动作执行有实际审计；不能说“只要把Q接全就成功”。scope明说缺真实能力证明、matched强基线及反馈重规划准入 |
| A3 能力已测 | `results/reacher_d2p_full_goal_metric_capability8_analysis_v1_20261026.json`：`total_independent_fits=1044`；三任务各8根，`aggregate.*.all_diagnostic_ingredients=0`，`P1_admission=false`；scope为保留全部根/角色的局部测量 | 缺的是完整机制合取而非“没做能力探针”；此处共同残差物理诊断不等价于原提案own-state目标 |
| D3 残差合同复核 | `docs/D2P_MECHANISM_CONTRACT_AND_MODEL_COHORT_20261026.md` → `results/reacher_d2p_mechanism_metric_contract8_v1_20261026.json`；696个新拟合区分own/common、RGB/Q-RGB/关节；本轮读了实际JSON顶层/聚合入口，但未重算拟合 | 合同差异明确，未恢复native见证；更强common诊断失败不能单独否定own-state命题。初始RGB上下文向量缺失保持unknown，不能以zero-prefix代初始状态 |
| A4 任务前提 | `results/reacher_d2p_model_escape_pool32_analysis_v1_20261026.json`：`eligible_count=1`、`selected_seeds=[]`、`predict_samples=149184`、`physical_future_branches=0`、`P1_admission=false` | 固定32根模型条件筛查不足预定8根；不是两阶段物理可达的证明，也不是物理P1反例；不得自动扩池/补根/调阈值 |
| A5 PushT suffix运行 | `results/pusht_d2p_nuisance_suffix_bank4x16_v1_20261026_cuda/report.json`：completed，11项gate true，3968 predict/128物理分支，P1=false | 完整性完成。四个已知校准根、16固定前缀；只改suffix，非完整编译器或正式P1 |
| A6 PushT独立分析 | `results/pusht_d2p_nuisance_suffix_bank4x16_analysis_v1_20261026.json`：64pairs/128branches，new_inference0；`summary.pose_cost.mean=+2.9309574228652586`，negative13/positive23/zero28；四根均差全正；`actual_common_nuisance_cost.mean=-0.06399817366655453`，P1=false | 共同Q改善未转为pose收益；64候选不是64独立根；额外goal-set输入必须披露。verification_limits：缺原Jacobian/GN参数/物理goal state，不能独立重算方向或绝对pose，不能宣布算法正确性全覆盖 |
| D4 物体与normalizer | `docs/D2P_PUSHT_ACTION_STATS_AND_OBJECT_RESPONSE_20261026.md`、`docs/D2P_PUSHT_GOAL_AGENT_INTERVENTION_20261026.md`：192分支复算；selected suffix均未改变物体；44编码/9门目标Q检查；官方权重训练统计绑定仍缺 | 支持具体目标/物体响应合同问题，不证明没有接触、不证明Q全局物体保真，也不授权替换统计常数 |
| S1 新fixedbank caller | `experiments/spectral_reachability/pusht_d2p_metric_program_compiler.py::compile_programs` 当前已落盘；逐prefix `_advance` 后重算nominal/Jacobian，调用array scorer并返回非线性预测；显式`P1_admission=False`、`spectral_prefix_generation=False`、bank provenance caller supplied | 比“仍未见caller源码”更进一步，但本轮未核实对应真实运行报告。非线性预测diagnostic only；不能当作已执行物理校准或完整谱前缀合成 |
| S2/D5 新sampling | `run_reacher_d2p_sampling_calibration.py`与独立auditor存在；`docs/D2P_PARAMETER_RECEIPT_REPAIR_20261026.md`记录两次host审批超时未执行。本轮Path.exists确认 `results/reacher_d2p_sampling_calibration2_v1_20261026_cuda` 为false | 真实sampling尚无结果。complex-cost丢虚部已由review交实施；当前实数dtype防护见前轮静态核查，不在本轮验收。fixture候选/计数不填真实结果槽 |
| D6 基线/文献 | `docs/D2P_BASELINE_FAITHFULNESS_AUDIT_20261026.md`、`D2P_SAMPLING_BASELINE_CONTRACT_REVIEW_20261026.md`及reported-baselines系列；核验仍partial | CEM存在，旧P1开环CEM与D2P反馈不对称；其它runner有两阶段CEM-MPC，不能说全仓没有。restart-Adam≠GRASP，clipped Gaussian heuristic≠faithful MPPI。论文报告值可引用但不是本地/同预算/配对证据 |

## 3. C1–C3：科学贡献要求逐项落地

| 要求/原文定位 | 已有证据是否足够 | 最小决定性真实实验（尚未完成，不是本轮运行授权） | 不能拿来替代；依赖/不适用 |
| --- | --- | --- | --- |
| C1a §4.1/§6：同一冻结动力学的task-weighted有限时域响应，产出动作、postcondition、子空间、时域与代价 | A0/A2足够支持特定接口局部实现，**不足**覆盖最终runtime；新PushT只有S1源码 | 在最终固定接口执行一组预声明独立根：保存Q/R、nominal、Jacobian/SVD或可核实中间量、动作与预测postcondition，再真实完整执行并比较实现误差；绑定checkpoint/context/action chart。单根足以检出接线违约，不足以宣称稳定收益 | AD-FD、DLS解析解、freeze hash、AST与mock只排特定bug，不是实际可执行抽象收益；训练技能/adapter对training-free C1不适用。依赖最终输入/normalizer合同和P0 |
| C1b §3–4/§8：不是仅换坐标的优化器，应有程序对象价值 | A2已有动作见证但 **不足** 排除仅数值优化收益 | E1三臂真实低层实现对照，加P2同谱/同solver无组合消融；若只等于GN或低秩搜索，按原文限制主张为优化贡献 | A0的D2P=GN正是等价性证据，不是程序优势；保存Program字典不证明新能力。完整效益依赖P1/E1/P2 |
| C2a §4.2/§8：短期非贪心→后继可达目标改善→真实闭环收益 | A2/A3没有同根全部成立；A4任务队列空；A6仅suffix物理恶化；**不足且前提阻塞** | 先在封存模型条件任务族确立局部直接搜索平台和两阶段可达，再在同一根上执行selected reconfigure/greedy/zero及同强度后缀；评价首段真实任务、预注册own-state能力指标、最终success/cost。模型选择先封存，能力探针仅评价；独立outcome根确认，不用探针挑程序 | 不同根各自满足一个要件、事后oracle候选、rank增加、Q-RGB改善、终距单独获益都不能替代。common响应诊断与own objective分账；不得降低门。依赖P0、合格任务族、目标/反馈合同 |
| C2b §4.3/§6：重配置与progress组合产生增量 | prefix-observe-suffix原型已有但 **不足** 证明程序组合价值 | P1后，冻结可组合候选与每边动作见证，执行最小多程序搜索，对照同库无组合SRA/平坦搜索及去重配置；同根成功和成本报告 | 两段控制流程本身、全库里存在好动作、单后缀更强都不能替代；依赖P1/P2，不先打开阶段 |
| C3a §5/§6：几何决定progress/reconfigure、深度/horizon、预算 | 现有全bank dispatcher/账本 **不足**，全枚举不叫自适应 | P1/P2后固定几何调度规则，在同预算上限下与fixed-depth/fixed-horizon/fixed-budget、随机或打乱几何信号调度比较；记录实际分配与success、完整资源/墙钟曲线，含谱分析成本与噪声消融 | χ阈值fixture、旧SRA horizon结果、仅少调用不证明质量—成本优势；不训练router，不宣称安全/最优。依赖P3 |
| C3b §6/§7 E3：同冻结模型复用多任务 | A2三动作限制任务是局部真实证据，**不足**完成多目标/horizon与长时效能 | 锁同一checkpoint，预声明目标×horizon×动作限制任务矩阵，独立根执行完整闭环；每格success、时长、predict/backward与失败保留 | 三任务共用8根不是24独立根；不同模型各跑一个任务不等于同模型多任务。依赖E3/P4；跨架构另属P5 |

## 4. E1–E3：每组实验需要排除的替代解释

| 原实验 | exact证据/尚缺关键对象 | 最小决定性真实实验与主终点 | 不能替代；任务依赖 |
| --- | --- | --- | --- |
| E1 Executable Subgoal Gap，§7 | A2有动作生成状态；D6指出 `audit_reacher_d2p_waypoint_tracking.py`跟踪既有动作rollout，**不是arbitrary latent优化臂**。未核实已有learned prior与三臂同合同回执 | 同任务/根、同低层controller、同观察和预算，分别生成 arbitrary latent waypoint、已有learned subgoal/proposal、dynamics-compiled state；真实跟踪，量化可实现比例、轨迹/终点偏差及最终任务成功。冻结资产与调参只用校准；outcome配对比较 | 任意噪声不能冒名learned prior；D2P自己的waypoint tracking只能诊断执行；跨论文成功率不能回答同controller接口缺口。资产/权限协议可独立查，正式完整D2P比较依赖P1/P2；无资产且需新训练时须用户选择，不能静默删臂 |
| E2 Controllability-Changing Actions，§7 | A2/A3/A4/A6均不足；能力测量已完成，不再泛排“补能力探针” | 在P1合格任务上，同根Full D2P对CEM/MPPI、vanilla gradient MPC、local GN、faithful GRASP、SRA without composition、kinodynamic；另保留greedy/zero作局部机制对照。冻结真实成功predicate和主比较，配对不确定性/多重比较按既定门；先证C2同根合取，再证完整策略增量 | GN同解、restart-Adam、开环弱CEM、无物理未来的149184预测、更多测试不替代这张对照矩阵。若不优于无组合SRA，原文要求不主张新程序级能力。依赖P0/P1；Full组合效益依赖P2，前期基线可独立准备 |
| E3 One Frozen Model, Many Tasks，§7 | A2同Reacher三限制已真实执行但无稳定机制；PointMaze-SRA不等于D2P；PushT A5仅固定suffix | 每个冻结模型内部切换预定目标、horizon、动作约束；PointMaze看绕行，Reacher看连续控制，PushT看接触重配置。分别真实闭环到预设预算/成功/失败，输出success、长时完成、耗时、调用及backward，不能只报终距 | 不把不同环境强行同一权重；“same checkpoint”是每个模型内任务复用。旧teacher/repair、各一小pilot、按效果删任务不能代替。协议可准备，正式Full D2P依赖P1及相应P2/P3/P4 |

## 5. P0–P5：阶段闭环与不可跳过的决定性证据

| 阶段/要求来源 | 当前报告足够吗 | 最小决定性真实实验/停止点 | 测试不能替代；依赖 |
| --- | --- | --- | --- |
| P0 §8：谱progress对同预算random/CEM/GN/GRASP | **局部实现足够，完整比较不足**。A0/A1仅两根calibration；p1_unlocked=false；随机0评分、GRASP-style非官方；encoder/backward/墙钟不匹配 | 最终封存接口上，以校准冻结预算/超参，独立root/source-disjoint outcome执行谱progress、best-of-N random、CEM、GN、faithful GRASP完整动作；同目标/支持/反馈，保留全部动作/真实终态及资源。按既有P0门评估，不以新阈值解锁 | 536=536、7门、D2P=GN、源码封存fixture不能建立效益；新sampling单次运行也不补齐所有臂。原文PointMaze/Reacher范围保留，先最小任务验证再逐项补齐；新chart/model须重新绑定前提 |
| P1 §8 kill experiment | **未通过**。A4不足8根，A2/A3无完整合取；A6无suffix物理恢复 | M1任务前提→M2校准冻结→M3独立outcome同根非贪心/own-state真实能力/后缀真实成功联合确认；对单层搜索及普通latent subgoal，补E2强对照。项目既定CI须严格满足，触零/空outcome不放行 | 完整性全true、库内正例、校准描述CI、跨根拼机制不能替代。无合格模型条件队列先停，不自动追加根或更换成功定义。必须区分实现错误、代理偏差、任务不合格与模型能力边界 |
| P2 项目多程序组合 | **阻塞**；两段原型不充分 | P1正式准入后最小多程序搜索，在各边保存输入/postcondition/动作见证，实际反馈后重编译；同谱/solver无组合及平坦搜索消融，独立真实success/cost证明增量 | composition单元测试、序列拼接、旧SRA程序名不能替代；依赖P1，不启动 |
| P3 项目动态horizon/算力 | **阻塞**；账本不是调度 | P1及必要P2后，C3a同上限预算/同反馈的固定与几何调度真实比较；同时报开销、噪声敏感性和完整收益曲线 | 静态门限、预期计数、AVIC不同任务论文值、旧SRA通过不能替代；依赖P1/P2，不启动 |
| P4 项目RGB多环境闭环 | **部分分支实测，不足完整闭环**；A5/A6、Reacher数组与PointMaze-SRA证据各有限定 | 各环境自己的D2P准入后，固定起始/目标分布和总预算，真实RGB observe→recompile多轮直至终止；PushT分接近/接触/推送/闭合并以最终物体任务成功为主，保存失败全过程 | evaluator物理坐标不得进入planner；单次suffix、终态pose改善、渲染重放通过不能替代多轮成功。协议/回放准备可独立；正式Full D2P依赖P1及采用的P2/P3 |
| P5 项目跨模型/任务复制 | **阻塞/资产partial**；D6文献值和其它repair资产不能补本地training-free复制 | P1及核心算法冻结后，在预声明独立冻结模型/架构上重新做接口/P0与任务前提，再复现核心机制和任务收益；同模型多任务结果另报，既有预训练/额外学习成本披露 | 不把论文成绩冒充本地、一个模型多seed冒充跨架构、adapter/readout训练冒充training-free；依赖P1及待复制组件，不能凭资产存在启动P5 |

## 6. 强对照缺口必须映射到具体实验

| 对照/当前实物 | 仍欠决定性证据及用途 |
| --- | --- |
| random / CEM | A1 random0/0只是sanity；shooting真实runner无输出。CEM真实存在但A1仅预测样本匹配；E2须统一真实反馈、tail执行和incumbent/动作支持，P0补独立outcome |
| MPPI / gradient MPC / GN | 新sampling为clipped Gaussian heuristic不是faithful MPPI；action Adam存在但不自动是同反馈MPC；GN同解只验证D2P progress。各自需要真实同合同接线/执行，排除普通搜索解释，不以算法名字数充对照数量 |
| GRASP | 只有restart-Adam proxy不能填官方臂；官方身份/版本/关键机制核验partial。可先paper-reported外部背景，若声称当前同协议优于它仍欠faithful本地/严格同协议原始证据 |
| 无组合SRA | 不能用GN名字代替。必须同编译器、候选支持、反馈及资源只关掉composition/reconfiguration，用于E2/P2的组件因果增量；论文值不能代替 |
| kinodynamic | 未核实当前正式接线；E2同模型、扩展规则/停止预算/目标合同真实搜索，以排除经典动力学搜索解释；不能给额外物理oracle再称公平 |
| arbitrary / learned subgoal | E1三臂缺失不能由已有action-witness tracking或随机proposal补位；需同controller真实执行。外部已训练资产可作为披露训练的强基线，不授权训练D2P或新head |

报告值来源与可比性沿用 `D2P_BASELINE_PROTOCOL_FIELDS_20261026.md`：goal_offset≠prediction horizon，SD≠SEM≠pooled，memory observations≠stored actions，RGB≠state-assisted，base frozen≠全系统无额外训练。原文无法核验字段标unknown/unverified，不联网重试限流；外部成绩不能转成root-paired CI或本地等预算证据。

## 7. 不适用项与当前最短依赖链

- LLM/Python代码生成、语义技能库、新policy/critic/adapter/readout训练不是本提案必要交付；D2P程序是数值动作/状态/控制流程。训练资产可以作明确披露的外部基线，不能填主线冻结要求。
- 连续Lie bracket不是PushT接触不连续任务的必过门；原文要求实际短动作rollout/顺序差异。额外增加某个理论近似测试不能拖替代真实控制。
- “证明全局可控性/安全/最优”不是原提案成功条件；局部bounded-response测量也不能升级成这些保证。论文录用/截止日期不属于科学证据门。
- 初始RGB缺失、历史饱和atanh参数缺失、物理goal未保存属于证据缺失，不是“不适用”；不能事后补造。新报告应保存，旧报告维持限制。

最短科学链为：最终目标/动作/输入合同与具体实现缺陷消除 → P0强对照及资源前提、模型任务准入 → 正式P1独立校准/冻结/outcome → P2组合 → P3调度 → E1–E3及P4完整矩阵 → P5独立复制。E1/E2部分工作可复用同一封存实验，但每条科学要求须分别给证据，不能一份gate覆盖所有贡献。

现在可独立安全推进：已有数组/文档的合同核查、faithfulness/资产/许可补表、资源与统计协议、预运行封存清单、独立计数审计准备及文档登记；测试仅针对明确缺陷验收。fixedbank caller已见源码，但实际checkpoint校准及完整编译仍是后续交付；sampling无实际结果，不据mock预测收益。需要主线程有效host执行批准才可真实校准，本轮不申请/运行。

确需用户选择：是否允许显式goal-set而非单goal-RGB；冻结learned prior资产不可得时是否补外部资源或保留比较缺口（禁止自动训练）；若固定模型任务池不足，是否另授权新任务分布/模型/预算，而非挑受益根。normalizer出处、缺失原文及真实收据是证据/外部条件缺口，不应让用户凭偏好选常数、阈值或科学结论。

## 8. 本轮可复核验证

完整分段读原附件；读closure的最新累计段、阶段/强基线矩阵及专属审计。以 `.venv/bin/python -c` 仅stdlib `json/pathlib` 定向读取A0–A6的status/scope/gates/aggregate与计数字段；未重跑分析器或重算模型结果。检查sampling目录不存在；静态阅读 `pusht_d2p_metric_program_compiler.py`，没有导入或执行。大量旧报告中的历史测试数未作为本轮验证或科学结论。写后只校验本文件路径、章节/要求覆盖、引述结果路径存在及空白；不修改其他文件、不联网、不运行GPU/CPU神经/训练、不提交推送或删除。
