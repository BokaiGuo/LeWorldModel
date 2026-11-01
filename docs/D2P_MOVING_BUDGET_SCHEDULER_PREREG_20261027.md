# Moving fixed-H2 budget scheduler：后续 P3 预注册协议

日期：2026-10-27。状态：**协议提案已落文档，runner 尚未实现/封存；仅允许独立授权的工程测试，P1/P2/P3 科学 gate 均不由本文件开放。** 本任务不运行模型/物理，不修改已有源码、结果、seal 或共享进度账。

## 1. 原 C3/P3 与本次最小问题

原提案 §4.1、§5、§6 C3 要求冻结动力学的可达几何决定程序生成、progress/reconfigure、搜索深度/horizon 和计算分配；最终检验是同预算下更好的可执行程序及真实任务后果，不是少调用、SVD 诊断或一个最优性阈值。依据为原附件 `/home/gbk/.codex/attachments/861f6764-1b25-4227-bf15-f1a237ba7d8f/pasted-text-1.txt`；项目阶段与科学要求见本目录 `D2P_RESEARCH_REQUIREMENT_EVIDENCE_MATRIX_20261026.md`、`D2P_P3_PAIRED_ACCOUNTING_20261026.md`、`P3_ADAPTIVE_RUNTIME_20261027.md`。

既有 inner `p3_horizon_residual_budget_20261027_v4.py` 与 `p3_joint_horizon_goal_metric_scheduler_20261027_v1.py` 在 H2/H3 间分配预算，分别输出10/15条 raw commands，且使用 joint-logit ball。它们不是本轮两根 moving task 的可直接复用 runner：不能截断 H3、补零 H2、换成逐包域后仍称原协议不变。

本次最小问题：**固定原两根、原 H2、原 off 目标/Q、原逐包合法域，在完整两根任务队列之间分配同一有限候选池的评价预算，谱信号能否优于 uniform/打乱信号？** 这是新的静态、模型条件的 panel scheduler，不是在线反馈调度、非贪心重配置、程序组合或完整 C3。它不能替代原 P3 的时域/噪声/任务族与成本曲线要求。

沿用 experiment-design 的先固定小对照再扩展原则；这里明确不采用训练、调参 sweep、看结果选最好变体或以非零成功作为工程通过条件。

## 2. Outcome 防火墙与冻结顺序

这两根已有历史结果，不能假装是从未看过 outcome 的独立根。本预注册只约束**新调度/新 winner 的未来结果**，所有结论标记 known-root calibration。

1. MAIN 确认本协议版本，随后新 runner 的 source/tests/doc、参数、候选池规则、完整 root/arm 清单及输入 manifest 一起封存；先产生独占 prepare，再运行 CUDA。
2. 调度输入白名单仅为：原 blind task 的 context3、原 goal latent、identity192 Q、原 action normalizer/域、冻结模型、两臂全部已封存 baseline commands、当前新 nominal/J/SVD。原两根全部纳入，不新增/替补/重排有利根。
3. 不加载任何新调度 winner 的未来物理结果、success/reward、pose/contact、prefix/terminal 实际 latent 或 evaluator 标签用于配额、候选、停机或选择。当前 paired overlay fidelity 的 MSE/Q/rank 也不是选 off/on/Q 的开关；本协议固定旧 off task，不以其结果改表示。
4. 旧 baseline 资产通过窄 handoff 入口仅读取命令及 provenance，完整保留 GN/D2P 两臂，不按既往物理表现删臂/选锚点。历史物理只作已有背景，不输入本 scheduler。handoff 中存在的额外结果字段不进入算法参数或选择函数。
5. 新 winner/replay/commands 封存后才允许 MAIN 独立授权新的物理执行。工程 CUDA 模型成本是预声明选择依据，不是 heldout 物理 outcome。未来真实评价单向读取 winner，不允许回写调度参数。
6. 数值/接线失败保留并停机；修复需独占新版本和新 prepare。保留旧产物，不追认新源码执行过旧报告。任何看到新结果后的规则变化属于另一个探索协议，不回填本预注册。

## 3. 输入身份、坐标与兼容接口

所有下列相对路径以 `/home/gbk/code/LeWorldModel/le-wm-main/` 为根；不能按文件数组位置猜 root 身份。

| 冻结输入 | 外部文件 SHA256 / 用法 |
| --- | --- |
| `results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_00.json.gz` | `a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b` |
| 同目录 `task_01.json.gz` | `101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74` |
| `results/moving_history_planners_cuda_20261027_v1_cuda_main01/report.json` | `cd8432c06288577d25d9d9b92c7eb3673de592499ee64cae93238156f3f19aa0`；原 GN/D2P 实际完整 run |
| `results/moving_history_planner_handoff_main01.json` | `52c475ed499b077f261517bc592252b208926919909674cebf68fa4029cc8310`；schema `moving_history_planner_handoff.v1`，是 handoff，**不是上行 CUDA report 本身** |

原 moving loader 自有 `load_task` 返回 `gn_task,d2p_packet,q,task_id`；禁止走旧 zero-history gate、伪装旧 schema 或 monkeypatch module globals。handoff 的 `projection.source_report_sha256` 与 `data.plans[].whole_run_report_sha256` 必须回指原 CUDA report；在新 manifest 中冻结每条 command 的 JSON pointer、source_slot、arm、blind SHA、selection SHA 和 task_id，不能硬要求 handoff 文件 SHA 等于 CUDA report SHA。

完整原 sourcepack/root/context/goal 身份经原 saved-only loader 验证后附在 evaluator sidecar；root join 为 stable root/task/blind 身份，不向 blind planner 泄露 placement、state、witness、未来 RGB。两根 sourcepack 的精确路径/SHA、原 prepare SHA及模型/源码 closure 由实现 manifest 从冻结 task/原 source authority 显式绑定；缺一项就拒绝 prepare，不在此虚构新 hash。

每根真实 raw warmup10：前5 `[.1,0]`、后5 `[0,.1]`，float32 原位值；history 为实际0/5/10。候选固定两个五步 packet：logits `[2,10]`、raw `tanh(logits)` 逐包上限 `||logits_packet||₂ <= .75 sqrt(5)`，命令 `[10,2]`。原 float32 mean/std 归一化，不再 `/100`；三个 state latent 与两个 between-state action history 按原 rolling context 算术重建。whole 与 segmented replay 全程 MATH/noTF32，模型 frozen/CUDA-only，无 CPU 神经 fallback。

## 4. 固定数学、候选池与分配

根编号 `i=0,1` 只在稳定身份 join 后作为冻结 RNG slot，不能从质量排序生成。

在零 joint logits 的完整 H2 graph 上，对各根独立 fresh nominal `f_i(0)` 和 `J_i=∂f_i/∂logits ∈ R^(192×20)`；H2 两包之间不可 detach。此处动作 chart 的尺度继承原 logits/normalizer，不另拟合 R 或 Q。Q 在代码里是原 identity192 residual transform；不把后续平方代价中的 Q 偷换成新矩阵。

令 `r_i=Q(g_i-f_i(0))`，对 `K_i=QJ_i` 薄 SVD。保留 `sigma > sigma_max * 1e-6` 的列，定义

```text
chi_i = ||r_i - P_i P_i^T r_i||² / ||r_i||²
w_i   = (1-chi_i) * ||r_i||²
```

零 residual：chi=0、w=0；零 rank 且 residual非零：chi=1、w=0。有限浮点 chi 沿旧合同 clamp 到[0,1]；NaN/Inf、SVD异常、非法维度直接失败，不用 epsilon 隐藏。w 是一阶可解释残差能量的启发式优先级，**不是可达性/真实改善证书**，可能把预算分给更易根，故必须保留弱根与 signal-swap 对照。

固定 panel 总计20个 block，每 block3个独立完整 H2 candidate、6 predict。配额为：

```text
U_uniform:  b0=b1=10
D_geometry:
    if w0+w1 == 0: b0=b1=10
    else: b0=floor(20*(.25+.5*w0/(w0+w1))+.5); b1=20-b0
S_swapped: 与D同公式，但仅交换两根w；不交换task/context/goal/candidate池
N_no_signal_uniform: b0=b1=10；不计算/不访问J、SVD或w
```

D/S每根5–15blocks，即15–45candidates，两根合计60；U/N各30。三种信号组 U/D/S 都实际记录共同信号成本，N单列无信号公平成本参考。主对照 D−U；D−S 检验根相关信号的必要性；D−N 检验谱 overhead 不免单时的价值。N与U相同确定性搜索内容但不是假装免单或空跑padding：N是预声明的独立无信号成本对照。

各根预先固定至多45候选的**规则和哈希**，按配额执行前缀：index0 zero，index1原GN commands，index2原D2P commands，index>=3随机尾部。不调用新 GN/D2P 优化器，不读取原 witness。锚点不重投影、不修命令，非法就拒绝整个 prepare。

随机尾部固定 PCG64 seed=`20261027+101*slot`；每个索引依次对两个10维 packet 各取标准正态向量和独立 `u∼Uniform[0,1)`，生成 `v/||v|| * (.75sqrt(5)) * u^(1/10)`，再 tanh。零方向拒绝并保存 RNG/失败，不暗中重抽。raw float32 casting与锚点 bit合同在实现mock冻结，不能根据结果改proposal。所有方法相同 root/index 的 raw 必须逐位相同；不因uniform配额较少而重排池。

这是随机池加原编译动作锚点，不称新的 spectral program generator、官方MPPI或完整动态 D2P。旧baseline资产历史成本见下一节，不能冒称从零规划。

## 5. 预算向量、编码与共享成本

先固定四策略，再生成任何新结果；不根据前两个策略表现决定要不要跑对照。每策略搜索60条H2=120predict，两个winner各付 whole H2+segmented H1/H1独立验证=8predict；每策略独占128。重复/zero候选按声明完整执行并计账，不去重“退款”，也不额外补候选来烧满预算。

| 本次新 panel 操作 | 实际硬上限/完整运行预期 |
| --- | --- |
| shared nominal，两根各H2一次 | predict4、AE4 |
| shared H2 AD forward，两根各一次 | predict4、AE4；已包含在下面总数，不重复加 |
| U/D/S/N全部搜索与winner验证 | predict512、AE512 |
| **完整四策略 panel** | **predict520、AE520**；每次 batch1，predictor/pred_proj各520，processed action context tokens=1560 |
| shared完整Jacobian | 2个192×20；384数学VJP方向；固定逐行autograd.grad最多384次，autograd.functional.jacobian/JVP API为0；实际engine backward单独hook计数，未知不得填0 |
| shared signal SVD / 新ridge solve | 2 / 0；不新增候选优化梯度、输入导数或二阶导数 |
| 新 visual encoder / projector、物理、训练 | 此cached planning phase均为0，只有实际路径未调用才可断言0 |

**共享不等于免费。** 原moving两task分别 history3+goal1，共8个已实际 encoder/projector 样本，属于输入 acquisition 历史成本。新runner不重新编码，但 report 必须指向其上游报告/prepare/模型/normalizer SHA与8/8收据；无法完整复原上游所有内部工作时明确unknown，不能把总pipeline视觉成本写0。加载checkpoint一次、source/hash/CPU数组校验、模型tensor hash、共享J/SVD、CUDA同步、内存峰值、壁钟均独立记录。

raw联合执行账中共享8predict只出现一次，因此总520。另报每个方法“若独立使用此算法”的逻辑资源视图：U/D/S各 `8+128=136predict`、2J/384方向/2SVD；N为128predict、0J/SVD。该视图是共享成本全额归因，不是额外实际调用，不把三组独立壁钟通过估计相加。若只审批先跑U/D工程子阶段，需另封子协议，实际上限264，不可把它当完整四对照结果。

每信号方法、每根 exclusive上限94predict（45candidate×2+4replay），uniform每根64；共享每根另4。全局/策略/根cap同时检查，不允许同一方法两根都用94。候选、AD、验证共用同一不可重置guard；先attempt后dispatch，异常不退款，blocked记录但不冒称实际forward。

原GN/D2P锚点来自历史搜索：GN每根78predict/AE、768数学VJP，D2P每根66predict/AE、2112数学VJP；两根历史predict合计288，不能再次当成本次新520，也不能从warm-start端到端比较中删掉。完整历史SVD/solve/API账以原封存receipt引用；未知轴不写0。本实验只对U/D/S/N固定共同资产的调度作配对，不声明与GN/D2P/random/CEM/MPPI等独立算法总资源匹配。

新runner面板wall cap600s（含验证、共享谱开销和加载），allocated<=3GiB、reserved<=4GiB；output cap16MiB、final reserve1MiB、disk floor200MiB，创建前需至少200+16MiB可用，不复制模型或旧大报告。这些是拟议新runner硬限制，不修改任何旧协议cap；若现场不支持就工程拒绝，不能静默提额/换CPU/裁原精度数组。所有实际数组、timings及模块attempt/return保存；计数一致不是FLOPs或walltime优势。

## 6. 谱、梯度、采样与真实执行何时可早停

主实验选择**无新增效能早停**，使完整成功运行的逻辑资源可比较；没有新阈值、patience、cost平台检测或省钱后重分配。

| 路径 | 允许停止/必须完成的精确规则 |
| --- | --- |
| 谱预处理 | U/D/S共享两根完整J和薄SVD必须完成，即使residual为0也不跳过导数。零rank/零残差仅按上式定权；不能看到chi大就停止当前根。N预声明无谱，不能先看信号再决定跳过 |
| 原GN/conditional D2P | 本次仅消费已冻结commands，既有原算法不重跑、不截断原4次GN更新/固定bank，不以历史成功或旧cost平台追认早停；历史成本原样记录 |
| 新梯度 | 只允许上节384逐行梯度收据，不做新的MPC/Adam、线搜或动态derivative refresh。任何扩展都需单独预注册预算与停止合同，不能混入520 |
| 新采样/评分 | 每策略恰好其配额的候选与完整两包H2；不能在第一包看成本差就剪后缀，不能提前发现低Q就停止，不能用失败子集继续选择。budget耗尽是正常结束，不是科学成功 |
| winner replay | 固定whole+segmented两次轨迹都要完整，不能因search成本好/命令重复而省略；任何全context/token不一致阻断该策略release |
| 工程硬停 | CUDA/OOM、非法设备/数值、source/hash漂移、guard/磁盘/时间cap等立即中止，所有已付费attempt保留，后续槽标blocked；不退款、不补根、不自动retry，不把工程失败计成完整理论负结果 |
| 后续真实环境 | 只有MAIN另授权的新consumer可执行；遵守native terminated/truncated和现有RGB/stepguard，合法终止可早停，不补步、不伪造checkpoint。异常/错判/native-angle风险另列，真实终止不回流CUDA调度 |

这与旧P3_ADAPTIVE_RUNTIME在完整层后根据固定gain阈值停止不同；不复制其阈值并宣称moving已验证。若后续要比较“谱判断何时停梯度/改采样/扩时域”，须在P1及适用P2前提满足后单开prospective协议，预声明策略、阈值/计数、同上限固定组、打乱/噪声组及完整实际成本，不能把本轮跨根结果替代它。

若再接CEM、MPPI或gradient-MPC，其原封存iteration/sample数、温度/学习率、nominal dispatch与独立验证合同保持原样；不能因为谱判为难根就少跑baseline，或把best sample冒充MPPI nominal。它们不是本轮四scheduler槽，也不计入520；需要同task/同表示的独占新协议和完整向量账后才可作为新的公平算法比较。

## 7. 固定选择、执行顺序与评价

每策略独占执行，不从另一策略复用candidate成本或winner replay。信号可物理共享但完整披露；策略固定顺序U/D/S/N。采用静态单次新加载会话、执行前同步和完整阶段timings；该固定顺序有cache/order混杂，因此本轮不宣称速度优势。需walltime科学结论时另行预注册平衡顺序/重复，不能按本轮最快值选。

每策略先各根5blocks作为共同最小保障，随后按剩余整数配额循环slot0/1，配额用尽则跳过；不按候选cost改顺序或额度。每根候选索引单调，三个一block。选择规则在搜索完成后固定为 `min(cost,index)`，其中 `cost=mean((Q@(native_terminal-goal))²)`；保留ties、全部候选与完整失败分母，不进行阈值成功筛选。候选选择后不能用真实pose或terminalRGB重选。

固定模型内主终点：两个完整root的平均winner Q，主差D−U；同时报告两个根各自差值，D−S、D−N为预声明次对照。至少一根失败则主要完整配对标not_available，保留所有失败与lower-bound成本；不得将两组共同失败根删除后报平均。两个已知根不做显著性/普适性结论。

成本曲线仅报已付费完整block边界10与20的双根best-so-far（含各方法谱成本及实际时间），不补神经调用/物理、不把不完整candidate纳入。原baseline成本及Q作固定参考，不因其物理失败换掉目标。模型Q改善是模型内结果，绝非任务成功。

未来可选的物理阶段不是本任务授权：固定4策略×2根=8分支，按原recipe重新构造原moving初态/目标并执行各自10commands，保留全部合法终止/失败。必须新consumer、fresh endpoints和原RGBguard；不得借旧baseline终点填新winner。物理/encoding成本另封，不能隐含在520内；所有策略同观察/反馈（本轮无执行中重规划）。官方success与独立wrapped/object pose分别评价，后者不覆盖native终止。真实预测/物理排序失配必须完整披露。

## 8. 新runner最小接口与实际交付前置

本文件不声称以下API已有实现。拟议独占stem `moving_history_budget_scheduler_cuda_20261027_v1`；不得patch旧P3或moving baseline。

```text
material(manifest_path, external_manifest_sha)
    -> protocol, two_blind_tasks, four_baseline_commands, complete_binding_closure
signal(root, full_H2_predict_and_AD_receipts)
    -> nominal192, J192x20, signalSVD, chi, w
allocate(method, frozen_two_signals)
    -> exact integer blocks and per-root/total resource caps
candidate(root_identity, index, immutable_asset_refs, frozen_rng_contract)
    -> literal raw10 + provenance; no privileged state or outcome
prepare -> verify -> MAIN execute_cuda -> own consume
    -> all 8 selected plans, full census, raw10, task/root/blind/selection hashes
```

prepare仅saved数组/源码工作，命令须显式外部SHA；执行接口无CPU fallback。own consume独立重建RNG、配额、完整池前缀、cost/argmin、history/token、double replay与全账，拒绝partial release；不重新跑模型。未来implementation/tests/doc及真实prepare路径/hash未存在，**不得为此虚构可运行CLI命令或ready seal**。

实现前工程前置：MAIN接受跨根问题范围；绑定第3节全部素材及source closure；RAM mock覆盖allocation边界/零rank/零residual/seed、packet域、handoff→CUDA双层SHA、nonzero history、cap/first-error/部分数组和全分母；MAIN完整读代码后封存。真实CUDA/物理仍各需MAIN独占执行授权。若素材接口或资源前提不足，保持工程blocked而不换根、放宽门或伪造prepare。

科学准入前置：保留原P1直接搜索/能力改变/非贪心/真实后果/强基线合同与独立根确认，以及适用P2的组合增量；本文件不判这些已满足。当前不足时，即使mock、prepare、CUDA账目、模型Q、真实校准都成功，输出仍必须 `P1_admission=false, P3_admission=false, independent_confirmation=false, compute_advantage=false, walltime_advantage=false`。完整C3还缺任务内progress/reconfigure/horizon调度、独立任务族、打乱与噪声敏感性、真实质量—成本曲线；本受限工程比较不能将其销项。

## 9. 本次交付的可验证事实

本任务仅新增本Markdown，未实现runner、未生成prepare/results、未运行神经/物理/训练、未读取新heldout来选方法、未改旧sealed bytes。当前paired overlay evaluator由MAIN维护其修复与新prepare，完全不属于本文件write set。
以上预算是预声明完整运行计数，不是实际receipt。文档静态核验仅能确认固定规则和预算算术自洽，不构成工程运行或科学证据。
