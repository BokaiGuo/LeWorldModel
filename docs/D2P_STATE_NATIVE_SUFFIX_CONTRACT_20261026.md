# D2P：统一想象评分与真实执行的数值后缀合同

## 要检验的代码假设

后缀多起点修复已证明部分严重失败来自数值盆地，但原前缀选择仍用较弱的fine8后缀
评分，执行端却已使用两种坐标加CEM。于是“重配置比greedy好”可能实际表示
“这条前缀上的局部优化器更容易收敛”，而非后继控制能力更强。本轮直接检验这项
代码/数值合同，不改变原始冻结模型、非贪心定义、margin、目标与动作时域。

全8根×17词的主机CUDA审计及128条真实分支现已完成，所有完整性门通过。
统一评分改变6/8个前缀决定，但没有改善旧runtime的整体真实表现：均距
0.124054→0.128201；6个新模型witness只有2个首段仍真实RGB非贪心。因此
这项合同差异值得修正，却不是本批P1失败的充分解释，不接入正式默认部署。
新来源8根的同协议独立确认已完成，全部完整性门通过；并未建立稳定收益。

P1和研究总闭环仍未完成。本次既不证明理论失败，也不声称所有代码原因已穷尽。

**后续代码发现覆盖旧归因**：目标生成器把有限位wrist错误视为周期关节，
导致旧批202644006与新批202646001的目标超出实际±160°范围。
这是已定位的任务构造缺陷；不能将这两根的严重失败解释成checkpoint可靠性边界。
保留原任务证据，限位合法的新任务版本已完成全16根CUDA配对审计：两个坏根
同样CEM设置的距离0.831375→0.075630、0.753711→0.086552，其余14根精确不变。
这是明确的新任务合同，不追认旧P1；最新广覆盖684项通过。
见[目标关节拓扑审计](D2P_GOAL_JOINT_LIMIT_REPAIR_20261026.md)。

## 已完成：来源隔离的新8根确认

证据：`le-wm-main/results/reacher_d2p_state_native_confirmation_fresh8_20261026_cuda.json`。
新根202646000..007与历史168个source episode隔离；8根全部136词、128条真实
30步分支、413302个实际predict样本已完成。FD、原17词fine8精确复现、分段、
来源隔离、权重、两级选择封存及全部执行门通过。

| 前缀与runtime | 全8新根平均真实关节距离 |
|---|---:|
| 旧dispatch + 历史多起点runtime | 0.1762570429 |
| 旧dispatch + 新runtime | 0.1753261127 |
| 新dispatch + 新runtime | 0.1768658554 |
| 同库greedy + 新runtime | 0.1768768342 |
| neutral + 新runtime | 0.1263629383 |
| 完整CEM + 新runtime | 0.1339800880 |

7/8个前缀决定改变，历史8个witness变成新4个；其中000/002/004首段实际RGB
非贪心，007反转（模型margin+0.00856638，真实margin-0.00081885）。对greedy
为3胜/1负/4平，均差-0.0000109787，描述性根配对95%CI
[-0.0074779229,+0.0090873640]。对历史runtime为4胜/4负，CI跨零；对neutral
为2胜/6负，对CEM为4胜/4负。不能将变化后的witness分母与旧6根校准拼接。

这是独立来源的数值合同pilot，不是正式P1校准/outcome、等资源优势或理论否定。
001的大误差后来定位到越界目标，另一个合法目标007的首段反转仍未解决；两者
不能混为同一种原因。完整summary及两张已目视PNG/PDF和哈希manifest见
`le-wm-main/results/reacher_d2p_state_native_confirmation_figures_20261026/`。

## 已完成结果

证据：`le-wm-main/results/reacher_d2p_state_native_selector_all136_20261026_cuda.json`
及同名protocol、selection、suffix.selection、roots、implementation。
实际总predict样本353414；包括全词候选、等价检查、重新评分和物理分支，
不是单策略部署预算。权重、原六候选成本/receipt/动作、原17词fine8轨迹、
分段递推、两级封存及历史完整动作/终态距离/RGB重放检查全部通过。

| 前缀与runtime | 全8根平均真实关节距离 |
|---|---:|
| 旧dispatch + 历史多起点runtime | 0.1240538912 |
| 旧dispatch + 精确command携带的新runtime | 0.1293162076 |
| 统一评分的新dispatch + 新runtime | 0.1282005544 |
| 同库greedy + 同样新runtime | 0.1318639571 |
| neutral + 同样新runtime | 0.1477972902 |
| 完整CEM + 同样新runtime | 0.1193686725 |

新dispatch对历史runtime为2胜/6负，均差+0.0041466632，描述性根配对95%CI
[-0.0004403796,+0.0096230507]。对旧prefix+同样新runtime为2胜/4负/2平，
均差-0.0011156533，CI[-0.0097326756,+0.0053363220]。对greedy为3胜/3负/2平，
均差-0.0036634028，CI[-0.0142683040,+0.0037636048]；对CEM为3胜/5负，均差
+0.0088318818，CI[-0.0078154245,+0.0254079100]。均不支持稳定优势。
这些是已知根、非等资源的描述性比较，不作多重比较校正后的正式效能主张。

两根由旧witness回退greedy，三根更换witness动作词，另一根从greedy变成新witness。
不改变immediate cost的计算，所以同一个前缀本身的预测/真实RGB排序不会被
后缀评分修复。新6个witness中根001/002/005/006反转，004/007一致；保留全部根，
不能把换词产生的新反转说成原历史首段的成功修复。

最难根`202644006`保留同一个第9词；历史/新runtime真实距离0.794063/0.800530，
同样新runtime的greedy/CEM为0.838331/0.752059。该根没有被解决。另一个独立
选择合同事实是：P1 selector优先寻找非贪心见证，并非全17词的终点cost argmin；
该根全库最小值来自progress词12，模型成本比已选词9低0.090474。后续真实
消融已完成，反而恶化，见下节；不能因此宣布P1通过。

全根两张PNG/PDF、重新核算summary、图注和SHA manifest位于
`le-wm-main/results/reacher_d2p_state_native_selector_figures_20261026/`，已目视检查
并独立验证全部artifact哈希。相关广覆盖回归596项通过，`git diff --check`无错误。

## 已完成：移除非贪心优先级的全库选择消融

`audit_reacher_d2p_native_global_word`从同一封存136词成本直接取全库最小值，
平局用词索引；不读真实终态、不更改margin或底层runtime。所有8根的决定统一
封存，观测后缀再统一封存，再执行3个新分支和1个原基线重放，共32条真实30步
分支。4个未改变的决定，其完整动作、qpos终态和RGB终态亦精确相同。

证据：`le-wm-main/results/reacher_d2p_native_global_word_all8_20261026_cuda.json`。
额外16260个实际predict样本，权重/来源/实现/选择封存/基线及重复决定精确重放
全部通过。它复用了既有全词编译成本，不能把额外16260样本说成完整部署预算。

全库选择改变4个决定，相对原native selector为3胜/1负/4平。均距却从
0.1282005544增至0.1369983645，均差+0.0087978101，描述性95%CI
[-0.0118976744,+0.0423769323]。最难根的全库progress词12在观测后模型成本
为0.733828，低于原词9的0.825094，但真实距离由0.800530恶化至0.919481，
真实终态RGB成本也由0.815874恶化至0.873709，不仅是qpos评价方式导致的反转。
因此“强制追求非贪心见证”会牺牲部分其他根的任务收益，但移除它并未解决最难根，
也没有稳定整体收益。保留该消融，不替换部署，不改写P1。

全根结果与模型/物理差异的两张图见
`le-wm-main/results/reacher_d2p_native_global_word_figures_v2_20261026/`。
首版图包保留；v2仅合并4个完全重合的unchanged点标签、改为根内配对连线，
不改任何数据或选择。独立summary、PNG/PDF、图注与SHA manifest齐全。
新增选择器8项和后处理13项fixture均通过；最新广覆盖617项通过。

选择器复杂度O(M)、存储O(M)；其余求解器仍是上述state-native接口。代码映射：
`select_global_word`只读模型score；`load_alignment`绑定原数值审计及两级选择SHA；
`summarize`按root identity重算选择、完整动作、基线及聚合值。与原非贪心selector
的差异是选择目标，不是新增训练、物理标签规划或修改理论命题。

## 已完成：扩大CEM搜索覆盖的全根审计

证据：`le-wm-main/results/reacher_d2p_cem_budget_all8_20261026_cuda.json`。
同一8个已知根、32条完整30步主机CUDA分支，804992个实际predict样本。
原完整CEM每代receipt及动作、原native完整动作和物理终态均精确复现，
来源、权重、封存及实现哈希全部门通过。固定128候选微批，避免将batch形状
变化混入搜索覆盖比较。原搜索为128样本×3轮/16elite，扩大为1024×12/64elite；
样本、轮数及elite比例同时改变，故这是联合搜索干预，不是单因素预算实验。

| 搜索位置 | 原平均真实关节距离 | 扩大后 | 胜/负 |
|---|---:|---:|---:|
| 完整六包开环计划 | 0.1520070105 | 0.1231277180 | 7/1 |
| 同native前20步，仅搜索后10步 | 0.1282005544 | 0.1425656068 | 3/5 |

完整搜索均差-0.0288792925，描述性根配对95%CI
[-0.0468302891,-0.0109819810]；同前缀后缀搜索均差+0.0143650524，CI
[-0.0057738500,+0.0477251212]。这是已知根、不等资源的诊断，不能作为独立
正式效益或重配置优势。完整六包搜索是开环，不是逐包真实重观察CEM-MPC。

最难根202644006的完整计划模型成本约0.847891→0.693687、真实距离
0.831375→0.799523，仍未解决；固定native前缀时模型成本约0.825094→0.742715，
真实距离却0.800530→0.925870。若只按后缀模型成本保留native/wide incumbent，
仅该坏根换成wide，全根均距反而为0.1438680409。因此覆盖不足解释部分损失，
但增加采样不能统一修复全部失败，也不能开放P1。保留原部署与所有失败分支。

两张全根PNG/PDF、独立重算summary、图注与已验证SHA manifest见
`le-wm-main/results/reacher_d2p_cem_budget_figures_20261026/`。
最新广覆盖回归632项通过；逻辑测试不替代上述主机CUDA结果。

## 不可变对照与单因素分解

已知根为`202644000..007`，来源于上一轮来源隔离的8根多起点确认。此次重复使用
这些根，所以是已知根代码干预，不再次称为fresh。每根重建全部17个动作词，
要求原fine8成本、优化轨迹与动作库精确复现，再给每词相同的六候选后缀求解器：
原fine8、tanh追加8/32次、command-box追加8/32次、独立CEM。所有词CEM使用
同一根seed+70000的共同随机数，不以物理标签选择起点或调节参数。

前缀选择仍调用同一个`select_refined_bank`，仅替换其`refined_goal_cost`。完整
六包递推与四包前缀加两包后缀的分段误差须不超过2e-6。所有根的前缀决定先封存，
再观察五组后继：旧dispatch、新dispatch、同库greedy、neutral、完整CEM。

真实RGB观测后，保留原cold fine8，并比较六候选及三个精确携带的command候选：
直接carry、carry后box8/32。保存command而非反算atanh，保留精确±1边界，
不能把已经有界的动作再次tanh。所有40组观测后缀决定再统一封存，才测128条
真实30步分支。旧dispatch另完整重放上一轮历史runtime，动作和物理终态必须精确。

因此可以分开比较：

| 对照 | 能定位的环节 | 不能直接推出 |
|---|---|---|
| 旧prefix+历史runtime vs 旧prefix+新runtime | 真实观测后的执行/计划连续性 | 新前缀选择有效 |
| 旧prefix+新runtime vs 新prefix+新runtime | 想象评分合同改变的选择后果 | 控制能力几何已经改变 |
| 新dispatch vs 同样runtime的greedy/neutral/CEM | 超过共同数值修复的增量价值 | 正式等资源或P1效能 |
| carry vs native，同一真实prefix | 携带完整动作与再优化的区别 | 低模型成本保证低物理距离 |

## 算法与代码映射

```mermaid
sequenceDiagram
    participant D as audit_reacher_d2p_state_native_selector
    participant W as reacher_d2p_action_words
    participant S as reacher_d2p_state_suffix
    participant R as reacher_d2p_refined_dispatch
    participant E as reacher_d2p_context
    D->>W: build_words(frozen action Jacobian)
    loop all17 words at all8 roots
        D->>S: compile_state_candidates(imagined cached state, original fine8)
        S-->>D: bounded commands, model costs, receipts
    end
    D->>R: select_refined_bank(common numerical scores)
    D->>D: seal every prefix choice and exact planned commands
    D->>E: execute each prefix, observe RGB/action context
    D->>S: compile_state_candidates(observed state, cold, planned_commands)
    D->>D: seal every observed suffix decision
    D->>E: execute complete branches; evaluate only now
```

```latex
\begin{algorithm}[t]
\caption{State-native numerical contract for action words}
\begin{algorithmic}[1]
\Require Frozen model $F$, RGB/action history $h$, goal $g$, bounds, fixed seed
\Ensure Model-only prefix index and exact bounded suffix commands
\State $\mathcal W\gets\textsc{BuildWords}(\partial_U F(h,0))$
\For{each word $w_i\in\mathcal W$}
    \State $h_i\gets F(h,w_i)$; $p_i\gets\textsc{Fine8}(h_i,g)$
    \State $(\mathcal C_i,c_i)\gets\textsc{CompileStateCandidates}(h_i,g,p_i)$
    \State $j_i\gets\textsc{ChooseStateCandidate}(c_i)$
    \State Retain $\mathcal C_{i,j_i}$ without inverse-tanh approximation
\EndFor
\State $i^*\gets\textsc{SelectRefinedBank}(\{J_{\rm immediate}(w_i),c_{i,j_i}\})$
\State Seal $i^*$ and all planned commands before physical observations
\State Execute $w_{i^*}$ and observe $h'$; $p_0\gets\textsc{Fine8}(h',g)$
\State $(\mathcal C',c')\gets\textsc{CompileStateCandidates}(h',g,p_0,\mathcal C_{i^*,j_{i^*}})$
\State Seal $j'\gets\textsc{ChooseStateCandidate}(c')$ before terminal evaluation
\State \Return $i^*,\mathcal C'_{j'}$
\end{algorithmic}
\end{algorithm}
```

实现：`reacher_d2p_state_suffix`仅消费latent/action缓存，不虚构RGB；
`state_costs`给CEM提供同一递推的批处理成本，最后仍用singleton重评分。
`compile_state_candidates`保留旧六候选优化器精确动作/成本/回执等价；
`choose_state_candidate`校验完整候选集合、有限非负成本，平局优先incumbent，
之后按候选名称排序。`reacher_d2p_state_native_words.compile_words`是独立命名的
完整编译器，不改历史`reacher_d2p_action_words`及其SHA。其原fine8回执必须逐项
等同旧实现，所有模型调用及VJP行显式核账。

设词数M=17，后缀H=2包、动作维d=20，latent输出维m。每种局部坐标至多
32次显式Jacobian，8次仅是内部snapshot，不重复求解；VJP按m行记录。每次存储
O(md+d²)，active-set线性系统最坏O(d³)，迭代次数以回执为准。每词CEM为
128×3×H=768个predict样本，singleton重评分另计。前缀每词4包，完整/分段
见证重评分另计6+2次。整体成本随M线性增长；这是诊断，不宣称等资源或效率优势。

## 结果验证与独立确认合同

`analyze_reacher_d2p_state_native_selector`用明确root/family身份连接全部封存记录，
重算旧/新selector与后缀选择，逐条验证完整动作、精确carry、分段误差、聚合数值
及fallback=greedy。所有根和所有词必须保留，不能用筛掉反转根制造通过。
两张PNG/PDF分别显示全部词成本与全部根的真实关节距离，并附summary、图注和
SHA manifest。逻辑fixture只检验审计合同，不能代替主机CUDA。

独立入口`run_reacher_d2p_state_native_confirmation`要求上一审计已完整通过；
固定8个新root并排除上一轮所有已使用source episode。每根还直接调用旧compiler，
要求新compiler保留的fine8全17词回执精确复现。五类prefix和128条物理分支
沿用同一协议，所有决定在终态之前封存。它仍是数值修复确认pilot，未自动填补
独立P1校准/outcome、真实控制几何、等预算强基线或C1--C3研究闭环。

独立确认的来源元数据预检排除168个历史episode。首个固定种子块
`202645000..007`被拒绝，因为其中`202645002`与`202645005`均落在episode3578，
是新批内部重复，并非已经观察结果后删除坏根。随后固定的下一块
`202646000..007`通过全8根episode唯一性及历史隔离预检。两次选择只读HDF5的
episode长度/offset，不运行模型、不读qpos、不产生新终态；记录见结果目录的
`reacher_d2p_state_native_seed_admission_20261026.json`。

## 复现

以下神经命令必须在获授权的主机CUDA运行，输出及sidecars须不存在：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -u -m experiments.spectral_reachability.audit_reacher_d2p_state_native_selector \
  --reference results/reacher_d2p_suffix_multistart_fresh8_20261026_cuda.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --output /absolute/new-state-native-audit.json
.venv/bin/python -m experiments.spectral_reachability.analyze_reacher_d2p_state_native_selector \
  --report /absolute/new-state-native-audit.json --output-dir /absolute/new-state-native-figures
.venv/bin/python -u -m experiments.spectral_reachability.run_reacher_d2p_state_native_confirmation \
  --goal-contract legacy-periodic-replay \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --alignment-reference /absolute/new-state-native-audit.json \
  --disjoint-reference results/reacher_d2p_suffix_multistart_fresh8_20261026_cuda.json \
  --seed 202646000 --roots 8 --output /absolute/new-state-native-confirmation.json
.venv/bin/python -u -m experiments.spectral_reachability.audit_reacher_d2p_cem_budget \
  --reference /absolute/new-state-native-audit.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --output /absolute/new-cem-coverage-audit.json
.venv/bin/python -m experiments.spectral_reachability.analyze_reacher_d2p_cem_budget \
  --report /absolute/new-cem-coverage-audit.json --output-dir /absolute/new-cem-coverage-figures
.venv/bin/python -m experiments.spectral_reachability.analyze_reacher_d2p_state_native_selector \
  --report /absolute/new-state-native-confirmation.json --output-dir /absolute/new-state-native-confirmation-figures
```
