# P2 剩余基线合同：Kinodynamic search 与 SAGE

## 状态与冻结边界

P1独占文档 `docs/p1_conditional_root_preconditions_20261027_v1.md` 已完成并保持冻结；
本次前后SHA均为 `60238c499e093efd6d90c543481955405b834ecfa2543a7bdc1e6203db2fd1ac`。
其结论不变：六臂greedy在有限库内Pareto支配；三条tradeoff不胜β1；四根未认证direct plateau与两段任务reach。
不把model-greedy当理论否定，不调β开门。

本次只新增本文，不改baseline inventory、旧报告/封存源码/共享进度页，不写wrapper或runner。
使用scientific-critical-thinking区分实现、保存执行收据、官方身份与可比效能；只读本地文本及JSON，
以标准库算术核账至stdout。无网络、训练、神经或物理运行、下载、解包和新代理。

下文未注明时，代码/结果路径相对 `/home/gbk/code/LeWorldModel/le-wm-main`，docs外层路径相对仓库根。

| 基线 | 本地可复用资产 | 论文reported/官方缓存 | 当前不能宣称 |
|---|---|---|---|
| Kinodynamic search | 实际bounded model-native beam/tree核心、CUDA接线、四根已完成模型搜索；另有固定H2 matched beam | 本地算法明确不复现某一具名官方论文；本轮未定位到可认证的对应官方实现/成绩 | “完全无代码”；官方复现；独立固定H2等总资源强基线效能已成立 |
| SAGE | 提案/比较附件与缺失registry；未定位到认证的SAGE执行模块/权重 | 有候选标题和Table1/Table18附件线索；无本轮可认证的一手正文/官方项目/权重来源；v1/v2/v3表均无SAGE数值记录 | “论文不存在/成绩为0”；附件转述是paper_reported；Hi-LeWM替代SAGE；可直接运行 |

## 1. Kinodynamic：已有真实搜索，不必从零重写

### 可直接复用的源码和语义

- `experiments/spectral_reachability/d2p_kinodynamic_program_search.py::search_programs`：
  输入完整latent/action context、明确raw primitive bank、replay和saved-goal metric callback。
  每条边只推进其新primitive，保留动作串及完整后继context；按`(cost, depth, bank-index sequence)`排序，
  逐层保留beam，跨全部已访问非空节点保留best。不是任意latent waypoint优化器。
- 同模块 `_validated_context`、`_bank`、`_cost` 可复用严格接口；已被P2 sampling等模块使用。
  raw primitive为完整五步packet的倍数，限制`[-1,1]`，不隐式clip/pad；depth计primitive而非神经样本。
- `experiments/spectral_reachability/run_pusht_program_search_cuda.py::metric_for/run`：
  固定goal[0]与保存Q的192维MSE，接`pusht_composition_cuda_replay_v2.CudaReplay`、冻结LeWM及PredictionBudget。
  此处没有把physics oracle传入callback；核心本身不能认证任意外部callback是否公平，须由父协议绑定。
- `experiments/spectral_reachability/p2_matched_composition_20261027.py`：已有固定H2、16首段→beam4→64后段的
  matched beam实现及flat对照。若main要固定H2，可复用此语义而非新写重复wrapper，且须披露它是已有固定bank搜索，
  不是凭重命名增加一个独立算法或新增结果。
- 现有非神经测试路径：`tests/test_d2p_kinodynamic_program_search.py`、
  `tests/test_run_pusht_program_search_cuda.py`、`tests/test_p2_matched_composition_20261027.py`。
  本轮未重跑测试，代码/fixture存在不作为效能证据。

### 实际保存结果与本轮独立核账

`results/pusht_program_search4_v2_20261026_cuda/report.json`：
文件SHA `4d0de9b3d39ac9df7b206dc301a49495ee4e5ea5523b2264ff9ba7669f04aa17`。
状态`integration_completed`，原8个gates全部true。逐根保存phase receipt与best动作长度的算术如下：

| 旧校准root | 选中primitive索引 | 深度/实际raw动作数 | planning / evaluation predict samples | 合计 |
|---|---|---|---|---:|
| 0 / 9173218f… | 10,2 | 2 / 10 | 80 / 2 | 82 |
| 1 / 23b8de78… | 2,14 | 2 / 10 | 80 / 2 | 82 |
| 2 / 4afefa3b… | 4 | 1 / 5 | 80 / 1 | 81 |
| 3 / cf1b9bb9… | 3,3 | 2 / 10 | 80 / 2 | 82 |

四根planning=320、整串evaluation replay=7，总327；每根cap82，failed attempts=0，
保存的`full_replay_exact=true`。本轮只核保存收据，没有重新模型重放。
每根16+4×16=80次扩展；所有根beam_pruned=72、partial_coverage=true。
`completed`表示预定beam过程完成，不表示全树覆盖；depth2层的60个末层剪枝也计在72内，
不应将72直接解释为72条未评估的末层边。

本轮还核对两份source-seal manifest文件SHA，并针对核心和CUDA入口验证
`当前源码SHA == archive副本SHA == manifest条目SHA`：

| 源码 | SHA |
|---|---|
| `d2p_kinodynamic_program_search.py` | `c92a945ca884d503b918c7f7357be7ee55fa36377d9de60e2dddf68b42208b7d` |
| `run_pusht_program_search_cuda.py` | `aaf9f9bb50ebc8902849112d309676dd7fe9b5ceb06c2e7a644188509bc2fba4` |

未重hash权重、全部输入依赖或所有传递源码，不扩大这两文件的认证范围。
报告记录的LeWM weights SHA为`48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`，
config SHA为`2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09`；本轮仅保留其来源绑定。

旧v1目录保留为接线失败现场，不覆盖或追认为完成；参考现有外层
`docs/D2P_PROGRAM_SEARCH_CUDA_INTEGRATION_20261026.md`的版本解释。

### 精确阻断与训练边界

1. **执行长度**：旧v2允许H1/H2跨depth选最优，根2返回5步。不能补零伪造固定10步，
   也不能将其与fresh conditional H2直接配对。固定H2语义已经存在于matched实现。
2. **动作支持**：旧bank是16个命令各重复五步，每个packet只有重复二维命令；
   fresh compiler是full10逐时刻动作、谱/DLS生成且per-packet logit球限制。
   声称同搜索域须统一支持/限制；仅都能输出10×2并不等价。
3. **指标与选择**：旧搜索取Q-terminal MSE，fresh conditional取stage+own-V；
   旧Q与新goal-patch Q不互换，不能跨目标直接相减冒充方法效益。
   父协议需要绑定确切Q/goal/normalizer/context、选择与停止规则。
4. **资源与反馈**：80 planning不是统一总成本。旧winner replay按深度1或2，
   matched实现每臂80+4，fresh每臂66且有11 Jacobian/12 SVD等。
   编译bank、继承编码、backward/缓存、物理反馈频次必须单列；不能凭扩展次数推导等总计算。
5. **结果层级**：旧v2报告是模型内校准，没有其独立物理效能结果。
   matched beam另有 `results/pusht_p2_matched_physical4_v1_20261027_cuda/report.json`，
   已选路径有partial/guard限制，不能移名为旧v2全部计划的真实执行，也不能替代独立P1条件根。
6. **官方身份**：kinodynamic是算法类别，当前`official_paper_faithfulness=false`；
   不必凭空指定一篇论文和SOTA值。可以明确报告“本地模型原生bounded beam search”。
   若要点名官方算法，先有真实论文/版本/代码与扩展策略认证，当前未建立。

这个本地搜索本身不训练新模块；冻结模型和bank的历史获取成本不是零。
本轮补文档不授权实现/运行新比较。最小可复用入口已定位，真正缺的是统一合同的接线收据和独立效能，
而非再写一个搜索核心。上述证据不能排除所有代码问题或打开P1/P2。

## 2. SAGE：有线索，不把附件当一手表格

### 已定位的材料与缺失表

1. 外层 `docs/D2P_REPORTED_BASELINES_SAGE_SUPERMIMIC_20261026.md`：原有缺失清单，
   标明当轮未取到原文，不能用网络失败推出论文不存在。本轮不联网重试，也不覆盖它。
2. `results/p2_reported_baseline_table_20261027_v1/report.json` 的`reproduction_readiness`中有SAGE：
   `published_value=unknown`、`project_values_ref=unknown`、`local_reproduction_verified=false`，
   `verified_paper_values=0`和`verified_project_values=0`表示**已核验条目计数为零**，不是性能为零。
   v1/v2的`reported_only`和v3的`rows`均未含SAGE成绩；v3四论文238条不能被当成覆盖SAGE。
3. `/home/gbk/.codex/attachments/380fd905-cc8f-4924-a507-d9292f925635/已粘贴的文本.txt:13`
   给出候选标题 **SAGE: Subgoal-Conditioned Action Generation for Latent World Model Planning**；
   :163声称Table1、:894–900指出state-assisted主表与Visual-only Table18的区分。
   这比只有简称多了检索线索，但仍是粘贴的比较说明，不是原论文PDF/HTML、DOI/arXiv版本或官方项目证明。
   该段只有arxiv域名favicon，没有可认证论文地址；本轮不转录其中数字，不确认作者、版本或表格真实性。
4. 原提案 `/home/gbk/.codex/attachments/861f6764-1b25-4227-bf15-f1a237ba7d8f/pasted-text-1.txt:424`
   也仅提供“多时域subgoal/条件action proposal”的方法线索，不补足官方资产或训练细节。

### 本地缓存检查范围与结论

本轮检查仓库docs、实验源码/tests、结果文件名、`external/`及`data/`，并对
`/home/gbk/.cache/huggingface/hub`、`/home/gbk/.stable_worldmodel`、
`/home/gbk/.codex/attachments`作最大深度4的SAGE命名查找；另检索attachments中≤2MiB的txt/md/json文本。
在这些范围没有找到认证的SAGE源码树、checkpoint/config/normalizer/许可/官方acquisition receipt。
现有external为JEPA/DINO相关资产，HF/世界模型缓存含LeWM/DINO等，不因共享LeWM骨干就变成SAGE权重。

`results/p2_reported_baseline_table_20261027_v3/sources/`缓存的是GRASP、Hi-LeWM、LeWM、PlanWAM原文，
没有SAGE专属正文；不能借Hi-LeWM高层模型或普通proposal bank替代。
这是有范围的“未定位到认证资产”，不是穷举全机所有匿名PDF/压缩包，也不是论文/权重绝对不存在的证明。

### 精确阻断顺序与训练边界

| 顺序 | 阻断 | 能解除它的具体材料；当前状态 |
|---|---|---|
| 身份/文献 | 只有候选标题和附件表述，没有唯一原文版本及一手表页 | 有来源的论文PDF/官方HTML、作者/版本/表号与raw cell；目前unknown，不录任何reported性能值 |
| 官方实现/许可 | 未认证官方代码URL、revision及本地缓存 | 官方项目与源码/许可/acquisition receipt；不能先猜GitHub组织或仓库名 |
| 模型资产 | 未认证subgoal/proposal权重、配置、normalizer及其对应LeWM checkpoint | 完整模块权重hash、结构与预处理/动作映射；只有本地LeWM权重不足以运行SAGE |
| 信息权限 | 附件声称主表state-assisted、附录visual-only，但本轮无原文认证 | 必须分臂核验planner实际输入，低维环境state不能自动当proprio；RGB-only对照不能直接移用state-assisted成绩 |
| 训练边界 | 附件称另训练subgoal和action proposal，具体训练量/数据/冻结程度未认证 | 原文和现有预训练发布资产应分别披露；若仅有训练脚本、没有可用预训练资产，则新训练需要用户另行决定，本任务不授权 |
| 比较合同 | goal offset、WM rollout、feedback、候选/optimizer预算、统计单位未具备一手绑定 | 逐项封存，尤其goal offset不等于prediction horizon、manifest SD不自动是training-seed SEM；原数值即便核实也先reported-only |

不因候选需要额外训练就将其从强比较清单静默删除；也不把额外训练隐藏成D2P的training-free。
目前可交main的是真实的来源/资产阻断，不是SAGE adapter可以立即运行的承诺。
在不联网且未提供认证正文/官方权重的本轮权限下，无法安全补齐SAGE数值或实际复现。

## 3. 给main的最窄交接

- **Kinodynamic**：复用已有核心/回放/预算账；固定H2优先复用matched beam语义。
  先解决动作支持、Q/选择目标、反馈和完整资源合同，再把已有执行结果登记为准确的本地基线层级。
  不把已有composition臂换名后算新增独立方法/样本，也不为补目录重跑旧实验。
- **SAGE**：先补唯一原文与官方发布资产的可认证来源；当前只保留unverified候选标题、缺项及训练授权边界。
  可供后续有权限时精确查找，但本轮不猜值、不请求网络、不训练。
- 两项均不改变冻结P1前置诊断，也不把工程存在/文档完善提升为机制或效能通过。

本轮核验：保存搜索收据的320+7=327、四根82/82/81/82、深度2/2/1/2与raw动作数一致；
两文件源码/档案/manifest SHA一致；SAGE三版数值表无已认证条目。所有核验仅只读stdout，未运行新测试或模型。
