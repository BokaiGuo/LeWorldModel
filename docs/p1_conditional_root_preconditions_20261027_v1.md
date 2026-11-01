# P1 conditional root 前置诊断：候选 tradeoff、β=1 与任务准入

## 结论先行

本批不是“80候选完全没有 tradeoff”，也不能称“只差调大β”。完整遍历四根×两Q×10候选后：

- 相对**各臂同库 immediate-greedy**，新Q根2有1个、新Q根3有2个严格 stage 更差而 own-reach 更好的候选；其余六臂为零。
- 三个候选的 stage 损失均大于 own-reach 收益，故固定β=1不选择它们；其余六臂的 greedy 同时严格最小化 stage 和 own-reach，任何非负β都不能从该库产生非贪心赢家。
- 四根来自已有 boundary-contact 校准集；没有“同目标/动作合同下直接梯度停滞、两段动作达到预定目标”的根准入收据。旧模型 preflight 的 pass 和 fresh compiler 成功均不能补此项。
- 当前可判定的是**有限候选库缺口 + 固定目标权衡 + 条件任务前置未认证**。不能推出全动作空间无候选、梯度确实停滞、真实两段可达或理论失败；也未发现这里必须靠修复一个β/192漏乘才能解释选择的证据。

本次仅新增本文。使用 scientific-critical-thinking 的条件样本/构念边界检查；只运行标准库的保存JSON/矩阵乘加算术至stdout，未导入compiler执行函数、神经/物理/网络模块，未新增审计代码、结果目录或测试，未改任何封存文件/β/阈值。没有重复main的完整机制审计器。

## 1. 身份与核验范围

路径相对 `/home/gbk/code/LeWorldModel/le-wm-main`，下文根号为原 root_index，不是新样本：

| 对象 | 现场核验 |
|---|---|
| `results/p3_goal_patch_consistent_compiler_20261027_v1_cuda_main_v1/report.json` | 文件SHA `64fd87f48f73674d98b8eb2ddce7415507bf1e6d457df8e78ad91da1c15c4fe1`；状态 `cuda_completed_h2_prerequisite` |
| `results/p3_goal_patch_consistent_compiler_20261027_v1_prepared/prepare.json` | 文件SHA `2b32137d1e7facf79335b25c6e6acf976ed10162073b938bff1262b385ba1425` |
| 同目录 `protocol.json` | 文件SHA `20909ca8eee96098dcf5fe8d2b25a0d70956df3fc7e816c8dc40685ea2118c43`，与prepare绑定相同 |
| protocol identity | `b9923b56096c7c81266a222dd7c86f1a95d62084c99abf0d2df413bd3cae94a4`；按源码 `json.dumps(sort_keys=True, allow_nan=False)`（保留默认空格）重算，与执行报告相同 |

特别区分protocol文件SHA、protocol内容identity和prepare文件SHA，三者不是一个哈希；用另一套紧凑JSON编码算出不同值不构成源漂移。

核对fresh编译器、`pusht_full_packet_compiler_20261027.py`、JSON canonical helper、旧P1 preflight和root collector这五份当前源码的SHA，均匹配fresh protocol的 `sources_sha256`。没有声称重验全部传递source seal、模型权重或旧物理轨迹。

遍历报告全部8条rows的 `compilation.candidates`：每臂恰好10个，80条completed/replay标记全部保留；逐候选 `objective == stage_cost + own_reach_value` 精确成立、相关标量有限；own-reach与保存的linear terminal+ridge penalty分解误差至多 `1.955e-16`。这些是保存内容的算术检查，不是重新神经重放或独立认证AD。

## 2. 严格比较定义：不要把三个参照混成一个

对**同一root、同一Q**的候选i，记：

`S_i = stage_cost`；`V_i = own_reach_value`；`T_i = nonlinear_terminal_cost`。

`g = argmin_i (S_i, index_i)`；当前选择 `w = argmin_i (S_i + V_i, index_i)`。

本文的greedy-relative tradeoff严格定义为 `S_i>S_g 且 V_i<V_g`，不设新epsilon，不据结局改变门。每臂遍历全部10候选，不跨Q比较标量，不按物理收益删候选。

源码中own-reach是 `min_{||u||≤CAP}(||Q(goal−nominal)−QJ u||² + .002||u||²)/192`，不是实际非线性终点T，更不是物理任务success。stage同样除192，所以β=1是两个同量纲项的冻结等权和，不存在此处stage除192而V忘除192的证据。V下降也可来自自身残差变化；它不独自证明响应子空间扩大或真实控制能力改善。

“比greedy更差”“比zero-prefix更差”“相对真实初始状态不进步”是不同条件。zero-prefix是模型推进一包后的coasting，不是初始状态。原提案§4.2描述短期不降低任务误差，而§8要求直接梯度容易停滞；二者也不能由same-bank greedy这一离散argmin替代。

## 3. 全80候选的覆盖结果

每行10候选，索引均0起计。`任意对tradeoff数`统计该臂45个无序候选对中出现相反S/V排序的对数；仅用于说明库内不是完全同序，不能把次优候选之间的tradeoff当成挑战greedy的见证。

| root/Q | greedy=β1 winner | own-V最小索引 | T最小索引 | 任意对tradeoff数 | 相对greedy tradeoff索引 | 最小其他候选objective差 |
|---|---:|---:|---:|---:|---|---:|
| 0/old_Q | 7 | 7 | 7 | 4 | 无 | +0.1491147768 |
| 0/goal_patch_Q | 7 | 7 | 7 | 1 | 无 | +0.0497345353 |
| 1/old_Q | 6 | 6 | 6 | 4 | 无 | +0.3866680765 |
| 1/goal_patch_Q | 9 | 9 | 5 | 7 | 无 | +0.0084717646 |
| 2/old_Q | 7 | 7 | 7 | 3 | 无 | +0.3635755798 |
| 2/goal_patch_Q | 1 | 5 | 3 | 5 | 5 | +0.0052186541 |
| 3/old_Q | 7 | 7 | 7 | 4 | 无 | +0.5112221479 |
| 3/goal_patch_Q | 1 | 0 | 3 | 8 | 0、4 | +0.0116374662 |

八臂共36个任意对tradeoff，但相对greedy仅3条（72个非greedy槽位中3条，不能把80或72当独立统计样本）。其余69个非greedy候选的S/V均严格差于对应greedy。六臂greedy同时严格最小化S/V，属于有限库的Pareto支配结果，不是β=1“压低”了一个已有更好V的替代方案。

### 三条真正相对greedy的tradeoff及为何β1不选

记 `ΔS=S_i−S_g>0`、`G=V_g−V_i>0`。相对greedy的总分差为 `ΔS−βG`。下表的 `ΔS/G` 仅是**固定保存直线的交点诊断**，不是调参建议、通过阈值或重新选择结果；交点处按原index规则处理，不一概声称严格胜出。

| root/newQ候选 | ΔS | G | β1总分差 | ΔS/G | T_i−T_g |
|---|---:|---:|---:|---:|---:|
| 2 / 5 | +0.0120272533430 | +0.0068085992407 | +0.0052186541023 | 1.7664798467 | −0.0052998508308 |
| 3 / 0（zero） | +0.0474909359517 | +0.0219158808839 | +0.0255750550677 | 2.1669645041 | −0.0388714137406 |
| 3 / 4 | +0.0310249582222 | +0.0193874919727 | +0.0116374662495 | 1.6002564058 | +0.0416885885044 |

根2 candidate5确实提供stage/own-V tradeoff，β1下尚差0.00521865；不是候选不存在。根3 candidate4虽然V更好，实际模型T反而更差，说明即便改变权重使其胜出，也不能称非线性终点改善。candidate0是零动作首段；被动等待的比较差异不能自动解读为主动重配置接触能力。任何权重变更都必须另版本、独立校准后冻结，不能用本表交点重选旧结果或制造P1通过。

### T不是V；greedy-relative不是initial-relative

新Q根1的candidate5虽V较greedy差 `+0.0011565459582`，T却低 `−0.0022381452930`；新Q根2的T最优candidate3其V也比greedy差 `+0.0008735586639`，T低 `−0.0236415865701`。这属于本批局部ridge surrogate与实际模型终点排序差异，不是当前代码按T选优有bug，也不是物理收益证据。

用prepared保存的 `root_context.latent_history[-1]`、原goal、各自Q做标准库矩阵乘加，直接计算初始 `C0=mean((Q(z0−goal))²)`：

| root | C0 old_Q | C0 goal_patch_Q | newQ greedy stage | newQ zero stage |
|---|---:|---:|---:|---:|
| 0 | 1.5332982376 | 0.2399342224 | 0.1518551769 | 0.2647570233 |
| 1 | 1.2767316937 | 0.1217786691 | 0.0843221615 | 0.1186038911 |
| 2 | 1.2913073915 | 0.1569733871 | 0.1464406868 | 0.1611681584 |
| 3 | 1.7083266161 | 0.3161167689 | 0.2681843606 | 0.3156752966 |

八臂greedy首段均使模型代价低于C0，所以没有“有限库内greedy一步完全无法进步”的现象；这**不证明**更长程直接梯度不平台，也不证明优化器达到最优。三个greedy-relative tradeoff中，根2 candidate5的stage为0.1584679401，确实高于C0；根3 candidate0/4的stage为0.3156752966/0.2992093189，都低于C0，不能称相对初始的短期退步。

相对zero-prefix，严格tradeoff另为newQ根0 candidate3、根2 candidate4；oldQ四臂及newQ根1/3均无。那两条都被各自greedy在S/V上支配，不能借换参照开启机制门。这进一步说明旧zero-relative模型见证不能当作当前greedy-relative或initial-relative见证。

## 4. 四根的来源不是原始P1条件任务准入

原始来源 `/home/gbk/.codex/attachments/861f6764-1b25-4227-bf15-f1a237ba7d8f/pasted-text-1.txt`：§4.2（343–374行）定义stage+βV与短期非进步/未来能力；§8（681–689行）明确“在模型已有相应动作动力学的前提下，选择当前直接梯度容易停滞、但两阶段动作可达的状态”，再验证真实闭环优于单层搜索和普通latent subgoal。原文没有指定本次β=1、四根或某个epsilon；这些是后来执行合同，不能反向当作原文已认证。

现场追溯的同一四个root SHA：

`9173218f…`、`23b8de78…`、`4afefa3b…`、`cf1b9bb9…`。

1. 最初 `results/pusht_d2p_p1_root_anchored_v5_calibration_20261026_cuda/report.json`：`root_seed=202753010`、count4、oversample4、cohort calibration，context=`actual_current_rgb_no_imagined_bootstrap.v1`。
2. `preflight_pusht_d2p_p1_reconfiguration.py:216`调用 `experiments/audit_pusht_reachability_horizon.py:84`的collector。它从16个boundary-contact提议按序取前4个unique且非终止根，不调用gradient平台测试或两段解的成功predicate。P0与cohort root去重不等于条件任务准入。
3. 原preflight在四根上均得到相对raw-zero的模型见证，selected proposal为9/10/13/6；但其聚合门只要求at_least_one。旧规则是先筛 `immediate>zero 且 value<zero`，再选最小value；不是当前无约束候选上的 `stage+V` argmin。旧sketch8、damping.01、每包重复二维参数；当前为active Q完整192维、damping.002、full10、per-prefix有界ridge。旧见证不能跨这些合同认证新Q/新动作库。
4. 后续已知库 metric preflight继承四根，声明known-bank compilation；P3 fixed-H3 → action-Jacobian v3 → H2/H3 v4 → goal-patch metric → 本次fresh conditional compiler继续继承同一四根/上下文。当前prepare只核“four distinct calibration roots”，contract显式 `new_independent_roots=0`；没有新的conditional inclusion predicate或直接梯度轨迹。

因此不是“普通随机四根”这么简单：它们确实是**boundary-contact且曾有旧zero-relative模型见证的四个已用校准根**。但这仍不足原始P1前置。正确状态是 `gradient_plateau=unverified`、`two_stage_task_reachability=unverified`、`fresh_conditional_cohort=false`，不是已经证明四根不合格、更不是全模型不具备该任务。

既有Reacher的 `docs/D2P_MECHANISM_CONTRACT_AND_MODEL_COHORT_20261026.md` 另有GN8→GN16平台+CEM逃逸的模型筛查；它自己也限定这不等于“两阶段机制成立”，且不足预定8根而输出空准入。不能把其1e-6、队列或Reacher结果移植给当前PushT Q充当已通过前提。

## 5. main决定独立新根协议前必须封住的确切槽位

以下是待预先确定的**可机器判定合同**，不是本次新增运行授权，也不假称已有数值。缺任一项就把相应任务证据标unknown/blocked；不能用增加普通校准根或事后β sweep代替。

| 槽位 | 新协议需冻结的判据/收据 | 当前缺口 |
|---|---|---|
| 同一任务对象 | 指定主Q、goal生成、原生RGB/动作/历史/权重、两包5×2动作、per-packet约束、反馈日程与目标success/cost阈值τ；旧Q只作预定对照 | 新Q是goal-only假说，尚不能由模型Q到物体任务排序自动推断效能；τ与独立任务标签不能补造 |
| 来源与不重复 | 固定有限扫描池/顺序、历史root及source episode排除清单、校准与确认分割、需求n与不足时停止；保留所有碰撞/失败，不自动扩池补根 | 当前四根已反复用于校准，不能再命名fresh outcome |
| 直接梯度平台 | 在相同完整H2动作域/目标下，指定强gradient/GN实现、初始化/重启、incumbent、连续预算B1<B2，保存完整轨迹；例如 `0≤L(B1)−L(B2)≤ε_plateau`。若称驻点另需预定投影梯度条件，不以一次J或小代价变化代替 | 本批只有root一次J与条件suffix ridge，没有同根直接优化延长预算轨迹；先前不同Q/域的gradient结果不可搬来 |
| 两段可达/逃逸 | 保存合法完整两段动作见证与精确模型重放，在同任务下 `L_H2≤τ`，而直接方法在其预定预算仍 `L_direct>τ`；若仅有 `L_direct−L_H2≥ε_escape`，只能称固定预算逃逸，不称已达到目标 | 本批T只是连续代价且未声明τ；V低不等于模型H2达到目标，更不等于物理可达 |
| 非贪心与未来收益 | 预声明greedy/zero/initial各自角色；同根 `ΔS>ε_stage`、`G>ε_reach`，若声称β1选择重配置还必须 `G−ΔS>ε_score`，并保持真实选择规则不变 | 3条tradeoff的 `G−ΔS`均负；六臂有限库被greedy支配。模型条件任务存在与固定选择器成功应分开报告 |
| 非循环筛根 | 平台/合法H2可达条件只读冻结模型预测，不读物理未来；若另按“β1会选非贪心”筛选，应显式命名selector-conditioned队列，不再把模型选择率当独立验证；固定候选支持和停止预算 | 不能为了恢复非贪心比例只挑当前三个候选所属根，或用物理oracle终点筛任务/调β |
| 真正P1验证 | 合格根另封物理calibration/outcome协议：selected、同库greedy、zero及同强度suffix，先记录真实首段，再测own-state能力、完整终点success/cost；强基线与普通latent subgoal同观察/支持/资源，失败全保留，执行既有准入统计门 | 本批无新物理数据；不能用不同根各满足一条拼合机制，不用响应rank增加或模型T下降代替真实任务收益 |

ε_plateau/ε_escape/ε_stage/ε_reach/ε_score、τ、扫描预算及n均须由main在独立协议中明确来源并预先冻结；本报告没有从这80候选反求“容易通过”的取值，也没有降低现有门。必要时只报告条件任务未找到并停止，不把门外样本当反例。

## 6. 可执行的判断分岔，而非理论裁决

- 对当前六臂：只能说“所保存的10候选不含优于greedy的own-V tradeoff”；应保留候选支持有限这一边界，不靠改β追求非贪心。
- 对新Q根2/3：候选存在但β1总分不胜，另有V/T排序失配；这是已定位的目标权衡/局部surrogate问题。改变β或改成T选优属于新算法协议，不是修复已证明的实现错账。
- 对原P1条件：目前最确切缺口是同任务**直接优化平台 + 合法两段目标见证 + 独立条件队列**，而不是再运行一批一般四根。若main保留β1路线，先冻结这些槽位及足够的选择分数裕量评价；模型未形成该族就不启动宣称P1的物理效能比较。
- 新Q候选生成已确实使用active Q，本次身份/算术未见必须修改封存源码的理由；但没有完成所有数值/模型/物理代码的全面排除。保留实现问题、代理偏差、有限搜索、任务前提不足和模型能力边界这几种解释，不下理论失败结论。

本文件是独立条件诊断，不覆盖main机制审计，也不把`winner==greedy`的再次计数当新实验。所有原report/prepare/protocol与源码字节保持不变。
