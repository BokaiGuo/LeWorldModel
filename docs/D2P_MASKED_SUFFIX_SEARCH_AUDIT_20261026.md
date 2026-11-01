# D2P：固定前缀的后缀求解与残差合同审计

## 当前结论

后续全16根逐包CUDA归因及432条反馈执行对照已完成。关键全驱动004通过自身
真实RGB反馈转为native优于同样反馈greedy，但这不修复原同动作预测，肩/腕也
没有稳定整体收益；P1仍关闭。详见[逐包反馈执行修正](D2P_MASKED_PACKET_FEEDBACK_REPAIR_20261026.md)。

不能把“冷启动四步 GN 无法找到好的后缀”归为理论负结果。固定官方原始冻结
checkpoint、同八根、全三任务、全部17谱前缀、目标与旧选择器，只添加保留原解的
CEM 和 CEM 后四步 GN，已显著改善部分真实执行。这是求解路径不足的因果证据，
不是 P1 重配置成功、等计算优势或全部 RGB 排序恢复。

另将原方案的状态特异残差目标实现成**独立、显式、结果前封存的实验选择器**：
它不等于旧共同残差诊断，不回填历史 P1，不替换主线部署或放宽任何门槛。
来源隔离新八根的配对确认已完成，修复后的原GN4对照456分支及后续1224条
配对分支全部通过完整性检查。求解不足的物理影响在新根复现，残差公式修正
没有使P1通过；下述两批分开报告，不把已知根冒称独立确认。

## 完成的同根因果对照

证据：
`le-wm-main/results/reacher_d2p_masked_suffix_search8_20261026_cuda.json`，
及其 `.protocol.json`、`.predictions.json`、`.roots`、`.implementation`。

全八根 `202650000..007`，全驱动/仅肩/仅腕三个原始力矩坐标图，全部17个
前缀均保留。未启用关节力矩始终字面零，过去真实动作历史不改变。只干预后缀：

1. 原冷启动 GN4 原样复现。
2. 固定128候选×3轮、16 elite 的 CEM，以原 GN4 为 incumbent。
3. 从该 CEM 解再做固定四步 GN，沿用原线搜索与 damping=0.01。

同动作前缀/后缀与完整递推逐位一致。批量搜索得到的解最终以 singleton 重新
评分；若没有严格优于原 incumbent，保留原动作，避免 batch 数值差异丢失原解。
此保护是预先防止数值回归，不是已证实的历史 batch bug。

共1224条真实15步分支、3672包，349320个主机CUDA predict样本；16项完整性门
全部通过，原408条物理分支的动作、全部包RGB SHA、192维表征和 qpos 精确复现。
所有模型预测与决定在任何本轮未来物理执行之前统一封存。权重、buffer、eval状态、
源数据receipt和实现哈希检查均通过，无后训练、CPU神经实验或 outcome 选根。

保存了每个前缀完整192×20或192×10的后缀 Jacobian，以及两种完整192维残差。
816个有界线性二次问题由 NumPy/SciPy 独立复算，核验 CUDA 的目标、增益、奇异值
与数值证书。独立数值复算不是新的神经实验或物理可控性保证。

| 任务 | 全17前缀的原GN4均距 | CEM保留原解 | CEM后GN4 | 后者根级胜/负 |
|---|---:|---:|---:|---:|
| 全驱动 | 0.779198 | 0.226064 | 0.147112 | 8/0 |
| 仅肩 | 0.897793 | 0.675429 | 0.650112 | 8/0 |
| 仅腕 | 0.861826 | 0.735696 | 0.711909 | 7/1 |

指标为真实最终关节距离（rad），先在每根全部17前缀平均，再对八根平均。
CEM后GN4相对原GN4的描述性根配对均差95%区间分别
[-0.762304,-0.478799]、[-0.326602,-0.164157]、[-0.282905,+0.025688]。
三任务136条依赖分支中实际距离改善121/105/105条；模型成本全部136条下降，
但物理并非单调改善。这些是已知根、不等资源诊断，不是正式效能或多重检验结论。

## 为什么还不能宣布 P1 修复

共同残差选择器的原三个条件全部不变。CEM后GN4只在仅腕两根产生新模型候选；
全驱动和仅肩仍均0/8。真实执行检查：

| 仅腕根 | 模型选词 | 首段真实RGB成本相对零前缀 | 终点真实RGB成本相对同库greedy | 真实关节均距差 |
|---|---:|---:|---:|---:|
| 202650000 | 1 | -0.027475 | -0.052697 | -0.018739 |
| 202650001 | 1 | +0.022533 | -0.069904 | -0.020559 |

第一根首段实际是贪心，仍有符号反转；第二根满足短期非贪心且终点改善，但没有
物理后继能力改变证据、独立校准或强等资源对照，不能单根开放 P1。CEM单独路径
在第二根终态RGB还比greedy差0.000522，不能只报成功的精修路径。

旧选择器各任务全八根均距原GN4→CEM后GN4为
0.227556→0.228707、0.548005→0.550012、0.681375→0.682513。
因此“后缀集合整体更好”没有自动转化为该选择器整体收益，不接入默认部署。

## 来源隔离新八根：确认已完成

证据：`results/reacher_d2p_masked_suffix_confirmation_fresh8_20261026_cuda.json`。
新根202651000..007，源episode3914/3518/6932/945/5057/1330/812/8198，与全部
208个旧episode隔离。算法在新结果前固定；原GN4基线为同根配对控制，不用于
模型选择。显式v2任务合同的原始对照完整完成后，再封存本轮全部模型决定，然后
执行所有未来分支。没有根据新物理结果改变前缀、beta、采样预算、目标或阈值。

1224条真实分支、349320个主机CUDA predict样本、16门全通过，408个原GN4
物理分支逐位重放；另816个完整响应QP由SciPy独立核验。两批后缀审计合计
2448条分支、698640个predict样本；新原始对照59568样本单列，不混成单策略预算。

| 任务 | 全17前缀原GN4均距 | CEM后GN4均距 | 根级胜/负 | 描述性根配对均差95%区间 |
|---|---:|---:|---:|---|
| 全驱动 | 0.699495 | 0.073673 | 8/0 | [-0.805058,-0.432910] |
| 仅肩 | 0.842140 | 0.589849 | 8/0 | [-0.363697,-0.153638] |
| 仅腕 | 0.839171 | 0.762679 | 5/3 | [-0.205749,+0.078455] |

对应136条同前缀分支实际距离改善123/120/95条；模型成本均136条下降。腕部
区间跨零，不能把前两任务的规律外推为所有动作限制下均有效。这是固定算法的
来源隔离诊断，不等于正式校准/outcome或等资源方法优势。

| CEM后GN4选择合同 | 全驱动均距 | 仅肩均距 | 仅腕均距 |
|---|---:|---:|---:|
| 旧common-gain诊断 | 0.026580 | 0.625659 | 0.668168 |
| 原方案stage + own V | 0.027808 | 0.512656 | 0.699142 |

proposal-native更改2/1/1个原选择，没有任何真实非贪心被选根，0/8模型重配置
见证（三任务均如此）。仅肩改善来自一根202651000：同后缀greedy均距1.372434
降至0.468414；仅腕一根反而恶化。相同后缀greedy配对比较三任务分别0胜/1负/
7平、1胜/0负/7平、0胜/1负/7平；不能把一次更好的progress选择说成重配置收益。
这是明确实现原公式的合同对照，不足以作为默认部署替代。

旧common-gain仅在全驱动202651004产生新候选11；其真实首段RGB相对零前缀
确实增加0.038094，但最终真实RGB成本0.003654比同库greedy的0.001006更差，
真实关节距离0.027886也比0.013873更差。模型终点成本却0.004371优于0.005042。
因此该根的剩余反转在终点，不是首段符号；不能用后缀修复后产生的新候选宣称
原RGB排序全部恢复。P1保持关闭，P2/P3/P5不从该链开放；本批不构成理论否定。

最终三张全根PNG/PDF、独立summary/notes/LaTeX和全部9项产物SHA位于
`results/reacher_d2p_masked_suffix_confirmation_figures8_v2_20261026/`。首版保留；
v2仅纠正复用统计工具的描述文字：实际距离按关节拓扑计算，不能笼统称“两关节
都wrapped”，新来源诊断也不标成旧已知根。所有原CUDA数据和数值不变。

## 另一个需显式修正的合同：残差必须随前缀变化

原方案写的是：

`r_B = goal - predicted_endpoint(after_prefix_B, zero_suffix)`

`V_B = min_{u in [-1,1]} ||r_B - K_B u||² + 0.01||u||²`。

旧 reduced-chart 诊断却使用固定 `r_common = goal - endpoint(root, three_zero_packets)`
比较各后继的可解释增益。共同残差可隔离方向响应，但不是上述状态特异的绝对
目标代价。在本批全驱动/仅肩/仅腕中，仅最优线性指标索引就有7/1/8根不一致。
这证明二者不能互相代替，不证明哪种部署一定好，也不解释全部历史 P1 失败。

新增 `reacher_d2p_proposal_native_selection.py` 明确采用
`score_B = prefix_MSE_B + V_B/192`（beta固定1），平局按原前缀索引。
除以192是让平方和形式的 V 与 prefix MSE 使用相同视觉尺度，不是调参拟合结果。
真正的数值重配置见证另要求：被选前缀相对零前缀非贪心、终点成本优于同库
greedy、own V及总score的可行上界严格低于两参照的数值下界。选择器仍可以选择
progress程序；不以“制造非贪心见证”为优先级，不读物理标签、不自动开放P1。

新选择与旧共同残差诊断分列封存。独立分析器另写标量公式重算，完整 own-state
QP由SciPy核对；拒绝事后加入的选择、beta改动、丢根或不完整variant。

## 独立复核发现的两处潜在数值保护漏洞：已修复并完成CUDA回归

独立只读代码审计确认：`fit()`求解的是step-damped GN（正则
`0.01||u-u_current||²`），而几何QP求解的是action-regularized局部目标
`0.01||u||²`。两者职责不同，不能把这个区别本身当作代数错误；本轮own residual、
192维缩放与证书下界的代数检查通过。

另外两处是可用对抗代数fixture复现的真实合同漏洞：

1. 原flat-CEM会按batch分数替换GN原解，但最后singleton评分更差时没有回退。
   已抽出 `flat_cem_control()`，保持原零中心分布、种子、预算、elite和采样顺序，
   最后按singleton分数保留原GN动作/预测；不增加模型调用。
2. 后缀CEM未检查初始singleton输出的形状/有限性，初始NaN能使比较fail-open。
   已对初始与最后的完整 `(1,192)` 输出及其成本做严格检查；溢出亦拒绝，不走CPU。

不能将这些潜在漏洞冒称已发生的RGB反转根因。两批全部816个历史后缀receipt的
初始/终止成本均有限，与原GN成本一致，无已观察NaN事件。完成主机CUDA差分
回归 `results/reacher_d2p_singleton_guards16_20261026_cuda.json`：全部16根×3任务，
48组flat GN、flat CEM与零前缀后缀CEM的原始动作及完整192维预测逐位一致；
97392个实际predict样本、10门全通过。权重、buffers和eval状态均不变。
这项回归没有新物理未来，不能重复计算为效果或新增成功根。

因此本轮归因分层如下：软限位源状态拒绝是已复现的任务代码错误；冷启动GN不足
是已知根及新来源均支持的数值求解局限；共同残差与own-state目标是显式合同
差异；新发现的两处保护漏洞已修复，但正常输入差分回归未改变本批结果。合法
模型候选的RGB排序和真实重配置效益仍未完全解决，不称理论失败或P1成功。

## 已完成产物与复现

### 新确认运行触发的真实代码缺陷

原 v1 新根确认在未来结果封存和物理执行前中止。保留其 protocol、implementation
和 `.failure.json`，不能把中止算成规划负结果。官方 episode1330、start267447
的根202651005，其 HDF5 qpos 为 `[-2.3432855027319235, 2.8294514443309158]`；
腕超出名义上限 `2.792526803190927` 共0.03692464113998861 rad。这是原始来源
中的真实状态，不是随机产生的替代根。MuJoCo软限位solver ref为[0.02,1.0]，
起点不应被当作硬边界内状态；其原请求目标腕角 `2.179451444330916` 完全合法。

显式新增 `joint-feasible-v2-soft-source`，v1仍保留原行为。新合同只取消“真实源
起点必须在名义范围内”的假设，源状态不投影、不clip，目标仍必须满足同样的
完整±0.65偏移与未放宽的名义限位，若两端均不合法则仍拒绝。planner不读qpos。
新增 source penetration receipt，并以来源HDF5与实际模拟器状态逐位核对。

`results/reacher_d2p_soft_limit_source16_20261026_cuda.json` 的主机CUDA/EGL审计
已完成七项完整性门：旧八根与新八根全部保留，15个v1有效上下文的根/目标/
历史RGB及完整状态逐位一致，唯一v1拒绝精确复现，新版保留原始状态和合法目标。
该诊断无神经predict、无未来物理分支；不能与实际规划重跑混记为成功率。
同一八个新根已用显式v2完整重新执行全任务对照：456条真实分支、59568个CUDA
predict样本，14门全通过。新三任务均0/8模型witness，native即greedy；全驱动/
仅肩/仅腕native均距0.053230/0.629108/0.607023，同任务flat CEM+GN incumbent
为0.034606/0.523774/0.696909，非等资源，不据此判断独有重配置机制。
全非零前缀模型/真实符号一致98/128、83/128、87/128；不同episode之间来源隔离
并不使这些根内分支成为独立样本。后缀/选择器确认沿用同一合同，
不回填历史P1或将本次新代码缺陷冒称全部历史反转根因。
新基线的全根双图、summary及SHA位于
`results/reacher_d2p_masked_program_confirmation_figures8_20261026/`，已目视检查。

### 命令与产物

全根双图、独立summary、notes、LaTeX及SHA manifest：
`le-wm-main/results/reacher_d2p_masked_suffix_search_figures8_v2_20261026/`。
首版图保留，v2只增加零线之上的标题留白与严格候选计数，未改原CUDA数据。
图采用 `figure-generation` 的线图/散点图样式，全部根不删，统一物理纵轴从零起。

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`；所有神经命令必须在主机CUDA执行：

```bash
.venv/bin/python -u -m experiments.spectral_reachability.audit_reacher_d2p_masked_suffix_search \
  --reference results/reacher_d2p_masked_program_pilot_fresh8_v2_20261026_cuda.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --output results/reacher_d2p_masked_suffix_search8_20261026_cuda.json

.venv/bin/python -m experiments.spectral_reachability.analyze_reacher_d2p_masked_suffix_search \
  --report results/reacher_d2p_masked_suffix_search8_20261026_cuda.json \
  --output-dir results/reacher_d2p_masked_suffix_search_figures8_v2_20261026

.venv/bin/python -u -m experiments.spectral_reachability.run_reacher_d2p_masked_program_pilot \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --disjoint-reference results/reacher_d2p_masked_program_pilot_fresh8_v2_20261026_cuda.json \
  --seed 202651000 \
  --goal-contract joint-feasible-v2-soft-source \
  --output results/reacher_d2p_masked_program_confirmation_fresh8_v2_20261026_cuda.json

.venv/bin/python -u -m experiments.spectral_reachability.audit_reacher_d2p_masked_suffix_search \
  --reference results/reacher_d2p_masked_program_confirmation_fresh8_v2_20261026_cuda.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --proposal-native-selector --source-disjoint-confirmation \
  --output results/reacher_d2p_masked_suffix_confirmation_fresh8_20261026_cuda.json

.venv/bin/python -u -m experiments.spectral_reachability.audit_reacher_d2p_soft_limit_source \
  --failed-protocol results/reacher_d2p_masked_program_confirmation_fresh8_20261026_cuda.protocol.json \
  --original-reference results/reacher_d2p_masked_program_pilot_fresh8_v2_20261026_cuda.json \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --output results/reacher_d2p_soft_limit_source16_20261026_cuda.json

.venv/bin/python -m experiments.spectral_reachability.analyze_reacher_d2p_masked_program_pilot \
  --report results/reacher_d2p_masked_program_confirmation_fresh8_v2_20261026_cuda.json \
  --output-dir results/reacher_d2p_masked_program_confirmation_figures8_20261026

.venv/bin/python -m experiments.spectral_reachability.analyze_reacher_d2p_masked_suffix_search \
  --report results/reacher_d2p_masked_suffix_confirmation_fresh8_20261026_cuda.json \
  --output-dir results/reacher_d2p_masked_suffix_confirmation_figures8_v2_20261026

.venv/bin/python -u -m experiments.spectral_reachability.audit_reacher_d2p_singleton_guards_cuda \
  --reference results/reacher_d2p_masked_suffix_search8_20261026_cuda.json \
  --reference results/reacher_d2p_masked_suffix_confirmation_fresh8_20261026_cuda.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --output results/reacher_d2p_singleton_guards16_20261026_cuda.json

.venv/bin/python -m pytest -q \
  tests/test_reacher_d2p*.py tests/test_audit_reacher_d2p*.py \
  tests/test_analyze_reacher_d2p*.py tests/test_train_reacher_d2p*.py \
  tests/test_run_reacher_d2p_causal_dynamics_repair_cuda.py \
  tests/test_bounded_response_qp.py tests/test_d2p_stage_admission.py
```

已存在的不可变路径不可直接重跑覆盖，复现须使用新路径并保持对应引用一致。
最新广覆盖1072项测试通过，属于实现合同证据，不是实际效果结论。新增公式、
全响应独立复算、源状态软限位、证书下界与新来源确认均有拒绝篡改/部分报告的
回归；仍需将模型/任务可靠性、物理重配置机制与等资源优势分开验证。
