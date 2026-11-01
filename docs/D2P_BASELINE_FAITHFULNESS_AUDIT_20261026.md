# Reacher D2P P0/P1 强基线忠实性审计（2026-10-26）

## 结论与证据边界

现有实现包含真实 CEM、动作 Adam 和阻尼 Gauss–Newton，但不足以宣称已完成 CEM/MPPI/gradient MPC/GN/官方 GRASP/random-search 的完整强基线比较。尤其：P0 random 不是随机搜索；GRASP-style 不是官方 GRASP；P0 GN 是 D2P 同求解器等价性对照；P1 主 runner 的 full-horizon CEM 是开环，而 D2P 有中途真实 RGB 反馈。另有独立修复 runner 实现两阶段 CEM-MPC，不能说仓库完全没有 CEM-MPC，也不能用新实现追认旧比较。

本审计只静态阅读 `le-wm-main/experiments/spectral_reachability` 相关源码及已有 docs，未运行神经推理、GPU、训练、下载或 HDF5 读取。源码证明实现路径存在，不证明对应历史作业执行过或结果有效。以下文档数字仅作为既有报告的陈述，本轮不独立验证结果 JSON。未实现其它并行任务的 metric/reference-contract 审计。

代码位置统一前缀：`/home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/`。行号为本次阅读快照；并行工作可能继续改变源码。

## 方法逐项核验

### 完整覆盖矩阵：不是四篇论文的替代清单

`implemented` 指已找到相关可执行代码路径，不意味着本轮运行过；`faithful` 只针对所声明算法/协议，外部论文未经逐实现核对不得标为通过；`missing` 指当前 Reacher 对照接线或必要合同缺失，而非断言全仓库没有任何相关代码。下表九类均必须纳入研究覆盖，不能以引用四篇论文宣布基线集完成。

| 要求类别 | implemented | faithful | missing / 精确证据 | 本地机制与等预算对照，还是仅报告值引用 |
| --- | --- | --- | --- | --- |
| CEM | 是；开环与两阶段 MPC 都存在 | CEM 更新机制已核实；旧 P1 对照不满足同反馈合同 | P0 `run_reacher_d2p_p0_goal_progress.py:528–559`；MPC `run_reacher_d2p_refined_dispatch.py:170–209`；完整资源匹配仍缺 | 当前完整策略优势必须本地同预算/同反馈；外部论文值仅作 `paper_reported` 补充 |
| MPPI | 未找到 Reacher 接线 | 未建立 | 本目录检索仅见 PushT preflight 的 required-arms 声明 `preflight_pusht_d2p_p0.py:413`，不是 solver | 必须补本地同反馈 MPPI 才能声称胜过该类；暂可文献引用但标明本地未比较 |
| vanilla gradient MPC | 部分：action Adam 存在，P0 没有 MPC feedback loop | Adam 动作优化真实；不能称已完成 vanilla gradient MPC | P0 `479–495,1029–1040,749–754`；缺同反馈 receding-horizon 接线 | 当前同预算单层搜索对照必须本地补齐；外部值不能替代 |
| local GN | 是：阻尼 GN/DLS | 本地更新形式可核实；作为独立算法对照不成立，因为与 D2P 同函数同 seed | P0 `277–285,979–988,1050`；缺独立标识的同反馈强 GN-MPC 比较 | 同解核验与去组合 GN 消融必须本地；外部 GN 值只作背景 |
| faithful GRASP | 只有 restart-Adam proxy | **未建立官方 GRASP 忠实性** | P0 `498–525,997–999` 只实现随机重启 Adam；缺官方版本绑定与关键机制映射 | 可优先仅引用官方 `paper_reported`，不强制训练复现；但若要声称当前同协议胜过 GRASP，则须本地 faithful 接线或取得完全同协议可核验数据，不能让 proxy 冒名 |
| SRA without composition | 部分相关谱工具和 P0 DLS 路径存在；未核实独立 Reacher 无组合 SRA arm | 未建立 | P0 方法表 `1020–1026` 无该独立臂；GN 同解不自动等于冻结版本的 SRA；docs closure `391–392` 明列缺口 | **必须本地机制消融**：相同谱编译、候选支持、优化预算与反馈，仅关闭 composition/reconfiguration；文献值不能替代本地组件因果对照 |
| kinodynamic search | 未找到当前 Reacher 对照接线 | 未建立 | `docs/D2P_RESEARCH_CLOSURE_AUDIT_20261025.md:392,437` 明列缺口；当前 P0/P1 方法注册无此臂 | 可先引用对应系统的 `paper_reported` 背景；若要排除“普通动力学搜索即可实现”的机制替代解释，必须本地同模型/目标/预算搜索，明确扩展规则、动力学边与停止条件 |
| arbitrary latent waypoint | 有 waypoint tracking，但不是 arbitrary waypoint search | 对 arbitrary 搜索不成立 | `audit_reacher_d2p_waypoint_tracking.py:144–163` 从已生成动作 rollout 取得 waypoints；`177` 读取这些固定目标。这些带动作见证，不能冒充任意 latent 提议 | **必须本地机制对照**：共用底层 controller 与预算，独立提议/优化 latent waypoint，测执行偏差及闭环结果；已有 tracking 只作执行诊断 |
| learned subgoal prior / SAGE / Hi-LeWM | 未找到 Reacher 正式接线；不能把普通 proposal bank 当作这些系统 | 未建立各自忠实性；SAGE 与 Hi-LeWM 也不能互相替代 | docs closure `391,437` 明列 learned-subgoal 缺口；P0/P1 注册无独立臂。本轮未审外部实现 | 官方系统可优先仅 `paper_reported`；但“相同底层控制器下动作见证优于 learned subgoal”的机制结论，仍需至少一个可核实本地 learned-prior 对照或严格同协议原始数据。不授权在本轮训练，无法获得时保留缺口、收窄结论 |

此外，random shooting 和同库 greedy/zero 是必需本地控制，并不因上表九类覆盖而删除。对所有仅文献引用的类别，`paper_reported` 不等于 `faithful locally implemented`，也不计作已完成的本地等预算实验。

| 方法 | 已核实的实际实现 | 忠实性判断与缺口 |
| --- | --- | --- |
| P0 CEM | `run_reacher_d2p_p0_goal_progress.py:528–559`：Gaussian 参数分布，包含当前均值，按成本选 elite，更新均值/标准差，标准差下限 1e-4；返回最后一代最佳样本 | 是 CEM，不是 MPPI。没有跨代 best-ever 保留；均值保留不保证保留上代最佳样本。这是合法但可能偏弱的 CEM 变体，不是自动构成实现错误 |
| best-ever CEM | `reacher_d2p_cem_incumbent.py:12–49`：跨代保留最佳样本，同时返回 legacy 最后一代输出；不修改采样轨迹 | 修复输出保留限制的独立实现；不能假定 P0 主 runner 已调用它 |
| bounded suffix CEM | `reacher_d2p_masked_suffix_search.py:8–55`：raw-action box 内采样，包含 incumbent/center，初末 singleton 评分，退回原 incumbent | 实际是 GN incumbent 支持的混合后缀控制，不是纯 CEM；预算为 samples×steps+2 次 terminal 样本，不能与 legacy CEM 名称混并 |
| MPPI | 对本目录 Python 搜索 `mppi`，仅发现 `preflight_pusht_d2p_p0.py:413` 的 required-arms 列表；Reacher P0 方法注册见 `run_reacher_d2p_p0_goal_progress.py:1020–1026`，P1 注册见 `run_reacher_d2p_p1_reconfiguration.py:1014` | 在本次限定目录/runner 范围未找到 Reacher 接线实现；要求清单不是实现。未检查仓库外部 MPPI，故不声称全项目绝对不存在 |
| action gradient | P0 runner `479–495`：零初始化 action 参数，Adam，固定步数，返回最后更新参数；每步标量成本反传 | 是动作梯度优化，不是训练。但不是 receding-horizon gradient MPC，也没有 best-ever/回溯保护；最后一次更新不保证成本下降 |
| GN / D2P progress | P0 runner `237–298`：完整 terminal Jacobian、解 `(JᵀJ+λI)Δ=Jᵀr`、六档步长（含零）、重新线性化；`979–988` GN 使用同 `_d2p_program`、同初始化配置和 seed 再计算 | 是有意的代数/实现等价性对照，不是独立强 GN 算法证据；不能声称 D2P 胜过它。`1050` 直接比较两者参数 |
| GRASP-style | P0 runner `498–525`：0.25×Gaussian 多起点，每个起点 Adam，最后对全部终点批量评分选优；`997–999` 挂接该函数 | 仅 restart action Adam；没有该函数实现的虚拟状态优化、动力学软约束。不能称为官方 GRASP 复现。邻近 PointMaze runner `run_sra_p0_pointmaze_cuda.py:294–295,428` 也明确限定为非外部复现 |
| random | P0 runner `562–568`：只生成一条 Gaussian 参数序列，零模型评分；`1041` 也不计算该方法 predicted terminal cost | 是 random-policy sanity control，不是 N 候选 best-of-N random shooting；不能用它代表预算匹配随机搜索 |

## 反馈、目标与执行差异

### P0：同目标，但均为开环短程程序比较

P0 `_cost_fn`（`run_reacher_d2p_p0_goal_progress.py:209–223`）统一使用 terminal 与 rendered-goal 编码之间的均方误差。`141–155` 允许 frozen wrapper 的 predicted/goal role-specific coordinate；因此应按具体 wrapper/protocol 报告目标，不宜一概写成“原始 latent L2”。

P0 `1029–1040` 将各 solver 输出转换成真实动作，然后交 `_execute`；`740–754` 顺序执行整条动作，不基于中间观测重新求解。它可回答固定 horizon 搜索/控制质量，不能直接代表 gradient MPC 或 CEM-MPC。物理终态评价是 qpos distance（`755–761`），并不等于优化目标；这种 evaluator-only 后果测量本身合理，但不是同一损失，也不应从其差异直接归因 solver 实现错误。

### P1 主 runner：反馈访问不对称

`run_reacher_d2p_p1_reconfiguration.py:553–571,586,603–605,618–627`：执行第一段，读真实 successor RGB/history，再用 DLS 生成后缀。reconfiguration/zero/greedy 三臂共用 `_two_stage`（`1004–1006`），适合隔离首段的作用。

同文件 `1008–1014` 的 CEM 则在初始根搜索完整 horizon，并直接执行整条动作；没有相同中点反馈。它是真实 full-horizon CEM，但不是同反馈 CEM-MPC。若把对它的优势归因“重配置独有能力”，反馈和后缀求解差异构成混杂。

greedy 首段也不是廉价单步梯度：`967–973` 使用 128 proposal、4 elite 的 screened D2P 求解。source-matched greedy（`975–987`）另可共用 proposal bank；这两种对照必须分开命名。

### 已存在的两阶段 CEM-MPC 修复路径

`run_reacher_d2p_suffix_confirmation.py:187–198`：初始 horizon=3 搜索，执行首段，再由真实 advanced context 搜索 horizon=2 后缀；保留 first/suffix receipt 与 1920 predict-sample gate。

`run_reacher_d2p_refined_dispatch.py:170–187` 同样实现初始三段、真实反馈后两段 CEM；`199–209` 记录执行与实际样本账，`205–207` 分列 predict 样本、逻辑 VJP 和耗时。因此这是两阶段/chunk-level MPC，而不是每个 raw step 都重规划；比较时应精确声明重规划频率。

`audit_reacher_d2p_cem_execution_contract.py:173–178` 明确区分 `shared_dls`、`observed_cem`、`stored_cem_tail`，是拆分后缀执行混杂的已有入口，不必重新发明。已有 `docs/D2P_RESEARCH_CLOSURE_AUDIT_20261025.md` 的“强基线与资源合同”及后续段落也记录了此历史修复；本审计不复核其数值结果。

## 资源真实性

1. P0 legacy `forward_rollout_evaluations` 不是统一资源单位。restart-GD `523` 写 `restarts*steps+1`，但 `518` 最终批量评分有 restarts 个样本；CEM `556` 则按候选数计。不能直接比较这些字段。
2. 独立 `audit_reacher_d2p_p0_predict_accounting.py:1–8,119–136` 已明确区分 actual predict calls/samples、旧字段与 VJP，不将样本相等当全算力相等。该审计验证收据一致性，不验证 CUDA 作业真实性。
3. `docs/D2P_P0_PREDICT_SAMPLE_MATCH_20261026.md:3–20` 报告默认 D2P/GN 536 与 CEM 256 predict samples，并以 CEM 67×4×2 匹配 536；只涉及两根校准。`21–22,38–50` 明确 random/GRASP 缺口、源码封存与正式准入限制。此处为文档报告，不是本轮重新测量。
4. D2P 完整 Jacobian 的逻辑 VJP 行、scalar Adam 的 backward、CEM batch forward 的单位与运行方式不同。即使 predict samples 相等，也没有自动匹配编码、反向、候选覆盖、墙钟、峰值内存或真实反馈次数。
5. P0 `383–445` 的 screened proposals 包含全局筛选、elite 多次 DLS 及末端重评分；预算必须包含全部失败/未入选候选。P1 每个候选的想象后缀和实际 successor 重编译也都应计入，不能仅统计最终动作长度。

## 最小可实施补齐方案（建议，不是已完成实现）

保持所有历史 gate、receipt、源码版本和结论不变。新增独立版本的 baseline contract/runner，不借本审计改写旧 P0/P1 pass。

| 优先级 | 最小改动 | 验证与声明限制 |
| --- | --- | --- |
| 1 | 接入已有 best-ever CEM 与两阶段真实 RGB CEM-MPC，统一两臂执行 chunk 和反馈时点；保留原 full-open-loop CEM 为消融 | 同根/同目标/同动作盒，保存完整动作及初始和 successor receipts；不将两阶段 MPC 描述成逐 raw-step MPC |
| 1 | 新增 best-of-N random shooting，复用同一 cost_fn、候选域与评分器；原 random 改报告标签而非历史数据 | 零模型单样本仍保留作 sanity control；明确 N、样本数和 seed law |
| 1 | 将现有 action Adam 接入与 CEM 相同反馈 schedule；新增 incumbent 保留和有限/边界检查 | 命名为 gradient MPC；新策略单独版本，不能偷偷改变旧固定步数结果 |
| 2 | 新增 MPPI solver adapter：固定 temperature、噪声/动作变换、成本权重与稳定 softmax，按加权扰动更新 nominal action；同反馈/目标/预算执行 | CEM elite hard selection 不能改名 MPPI。先用非神经数值 fixture 检验权重、维度、边界与 seed，再经授权做独立 GPU 核验 |
| 2 | GN 保留同解等价性 gate；另加同预算 multistart GN-MPC 基线 | 若仍与 D2P 数值等价，诚实报告同算法，不把标签数当独立方法数 |
| 2 | 暂将 GRASP-style 在新报告中明确命名 restart-Adam；官方 GRASP 复现需先冻结论文/官方实现版本及算法映射 | 在虚拟状态、动力学软约束等关键机制与超参未逐项对齐前，不称正式 GRASP；本轮未查外部源码，无法给官方忠实性通过结论 |

所有新臂共用冻结 checkpoint、RGB/history、goal/metric、动作尺度/约束、终止与反馈协议。校准根只选择超参和资源上限，outcome 根独立封存；统一采集 predict calls、batch samples、预测 token 工作、逻辑反向工作、编码和规划墙钟，报告资源向量及预算曲线，不硬造单一“等 FLOPs”。共享缓存与编译成本明确按部署路径摊销，诊断重算单列。

正式比较须保留所有预定根，预声明主比较与多重比较策略；对同库 greedy 的增量收益和对强 MPC 的完整策略收益分别检验。基线实现通过、数值 fixture 通过或旧小样本均值有利，均不开放历史 P1 或后续阶段。

## 最终判定

- **有代码证据**：CEM/Adam/GN 算法路径真实存在；P0 random 非搜索、GN 为同解对照、GRASP-style 为 restart-Adam；P1 主 runner 与 CEM 存在反馈不对称；其它 runner 已有真实两阶段 CEM-MPC。
- **未建立**：完整 MPPI 接线、官方 GRASP 忠实复现、同反馈同资源强基线集、独立 held-out P1 优势。
- **不可推出**：上述缺口必然解释所有历史负结果，或补齐后 D2P 必然成功/失败。最小补齐先解决基线定义、反馈和完整资源合同，不需要本轮训练新模型。

## 补充授权：本地必跑与 paper_reported 的分界

用户补充允许对比实验并行，公开论文值优先直接引用以节省复现，但要求协议一致并标注 `paper_reported`。本节记录该授权的使用边界；不取消本次审计的只读代码、禁止 GPU/训练/下载/全量 HDF5 等限制，也不在本轮启动实验。

### 支撑当前 D2P 机制与等预算结论必须本地运行

- **best-ever CEM-MPC**：与 D2P 同一 checkpoint、根、目标、动作约束、真实反馈频率和冻结资源合同；旧 open-loop CEM 单列为消融。
- **MPPI-MPC、gradient/Adam MPC、multistart GN-MPC**：各用同一反馈 schedule 和成本接口；超参只在校准集选择。若论文要声称胜过这些方法，就需要本地同协议动作、资源和结果收据，不能用不同 checkpoint/任务的论文均值替代。
- **best-of-N random shooting**：同目标评分、动作支持和明确的 N/预算；原单条 random 只作 sanity control。
- **同库 greedy、zero-prefix 与去重配置消融**：相同后缀求解和反馈，隔离重配置本身的增量作用；这些是当前机制的局部因果对照，外部论文没有可替代的对应动作分支。
- **D2P/GN 同解回归**：继续保留，但只作等价性/实现完整性核验，不算新增独立强基线胜利。

这里“等预算”指预声明的资源合同，不代表各算法可同时严格匹配所有资源坐标。至少匹配选定的主资源约束与反馈访问，并完整披露 predict/token、反向、编码和墙钟差异；未匹配项不能隐藏。若某方法不进入本地比较，应收窄结论为“尚未比较”，而非默认胜过它。

### 可优先只文献引用的部分

- 官方 **GRASP**、learned-subgoal/Hi-LeWM/SAGE、kinodynamic 等外部系统的论文结果，可先作为 `paper_reported` 背景或外部参考，不必为完善相关工作而训练/复现全系统。restart-Adam 不因此获得“官方 GRASP”身份。
- 外部 CEM/MPPI/GN 等论文数值同样可以 `paper_reported` 方式引用，但通常只能提供各自论文设置下的上下文，不能替代上述当前任务的配对对照。
- 只有逐项确认 checkpoint/模型版本、训练数据与适配、任务与目标分布、动作接口、观测/反馈、horizon、指标/成功阈值、预算与计时范围、评估划分一致时，文献值才可进入协议一致的性能对照；每行保存论文版本、表/图号及协议核对来源。关键字段未知时标为 `unknown`，不视作已匹配。
- 即使协议一致，`paper_reported` 仍须与 `locally_measured` 分列：没有逐根原始值，就不能计算当前 root-paired CI、配对胜率或机制判据；没有同硬件/计时合同，不声称等墙钟。若正式结论必须依赖这些量，仍需本地运行或取得可验证的同协议原始收据。

本轮没有检索/核验外部论文具体数值，故这里没有新增任何“协议一致”的文献成绩。未来可并行执行独立基线，但须隔离输出与 seed 流，记录并发 GPU 争用；墙钟主比较应独占运行或使用预声明且一致的并发条件，不能把并行加速误计为单方法优势。
