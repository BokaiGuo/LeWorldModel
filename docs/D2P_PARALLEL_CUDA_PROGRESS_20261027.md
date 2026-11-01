# 并行后续阶段：实际 CUDA 与物理进展

完整研究仍未闭环。P1 未通过；以下工程/校准证据不开放依赖阶段，不重写历史失败。
四条并行任务分别为 P2 matched composition、P3 dynamics-adaptive runtime、
P4/P5 prerequisites，以及 PushT sampling baselines。主线程负责代码审查、
实际主机 CUDA 调度和真实动作执行。

## P2：模型搜索和八个真实执行分支均完成

规划报告：`le-wm-main/results/pusht_p2_matched_composition4_v1_20261027_cuda/p2_matched_report.json`。
四根、两组均严格 H2（10 个 planned primitive steps）；composition 为16首段+64后段，
flat 为40个完整双段，两组每根均80 planning +4 independent replay samples。
实际共672个CUDA predict samples，8项完整性门全部通过。
这是固定动作库 beam composition 对 flat ordered-pair search，不是完整谱 D2P
或强 no-composition SRA，不能把相同预测样本写成相同总算力。

物理报告：`le-wm-main/results/pusht_p2_matched_physical4_v1_20261027_cuda/report.json`。
动作选择和见证在物理执行前封存；主线程新增实际执行器并运行，8分支全部保留，
7项完整性门通过，8个实际终态均CUDA编码。全部计划80步，实际执行72步，
guard阻止剩余8步；没有重新选择、padding 或恢复后重试。

| 校准根 | composition 物体 pose cost | flat 物体 pose cost | composition / flat 状态与步数 |
| --- | ---: | ---: | --- |
| 0 | 162.95602 | 174.67809 | completed 10 / completed 10 |
| 1 | 134.31427 | 159.25319 | completed 10 / completed 10 |
| 2 | 89.44509 | 141.73034 | completed 10 / observation_unavailable 10 |
| 3 | 28.89755 | 28.89755 | observation_unavailable 5 / observation_unavailable 7 |

两组在全部四根的 official success 均为 false。不能把根3不同中止时刻的相同物体
pose 当固定时域效能；根2虽完成10步，其flat终图agent不可见也必须保留标记。
composition 在前三根的物体pose终距较小，仅是已知四根的描述性观察；
根0 official state distance 反而更大，不能称所有任务指标改善。
没有独立根、非贪心机制见证、配对显著性或已成立的组合增量。

主线程的真实动作验证先检查完整root predictor context、原始候选库、
声明/receipt hash、固定双段、实际goal/RGB绑定与selected metric；
14项非神经回归覆盖错根、改动作、缺arm、坏hash、非有限分数等拒绝路径。
执行源码运行前archive seals、运行后验证均通过。后续发现新问题必须另版本，
不能修改本轮已封存driver或tests假装历史不变。

独立只读代码复核未发现本批动作精度、root/goal/stack-RGB错接或失败候选过滤；
另外逐分支核对 callback trace 长度=completed=attempted，合计72步。
保留一个未来工程风险：终态PNG/编码成功前尚未写branch JSON；若该步骤异常，
应归为工程中断并另版本修复，不能当作任务负结果。本轮八个编码/落盘均已成功，
没有以该风险替换或否定已保存物理结果。

## P3：已经执行，不再只是 runner

报告：`le-wm-main/results/P3_ADAPTIVE_RUNTIME_20261027_cuda_v1/P3_ADAPTIVE_RUNTIME_report_20261027.json`。
`runtime_integration_completed`，10 项完整性门全部通过。四个已知校准根全部保留，
每根固定和自适应两组，逐根交替先后。参数/来源字节未变，全部动作见证独立重放精确一致。
实际共 1,171 个 CUDA predictor samples：固定 586，自适应 585；峰值 allocated
显存 106,376,192 bytes。没有物理执行、没有新根泛化。

| 校准根（前8位） | 固定 predict samples | 自适应 predict samples | 固定最优路径 | 自适应最优路径 |
| --- | ---: | ---: | --- | --- |
| 9173218f | 146 | 146 | 10,2 | 10,2 |
| 23b8de78 | 146 | 146 | 2,14 | 2,14 |
| 4afefa3b | 147 | 146 | 7,14,11 | 9,4 |
| cf1b9bb9 | 147 | 147 | 9,3,3 | 9,3,3 |

自适应在根0/2扩大 frontier 后达到 expansion cap；这是保留最佳已评估计划的
预定预算停止，不是未捕获的预测异常。根3出现动力学 horizon-stop。
根2的 Q-MSE 从固定 0.9593272301683169 到自适应 0.887057713618011，
只说明同一有限模型/度量下搜索选择不同，不证明物体任务改善；其他三根选择相同。
节省一个 verification sample 是返回路径长度差异，不是已证明更少规划计算。
不报告 compute/walltime advantage，不把未测物理 success=false 当任务失败率。

主线程审查时发现新 imagined tokens 与 action encoder 实际处理历史 token 的单位
有歧义；已在首次运行前分开。实际 processed context tokens=3,513，
恰为 3×1,171；不是 3,513 个新预测步。计数对应关系和两种 cap 均实测通过。
NumPy SVD 是非神经数学，所有神经前向均主机 CUDA，没有 CPU 神经回退。

事后逐层独立核对又发现一个未被上述10门捕获的资源字段错误：根0/2 adaptive
有一个 depth=3、attempted_expansions=0 的空层，旧runner仍将其计入
`max_planning_horizon_tokens=3`；实际搜索和独立验证均最多深2。
因此原始 `equal_resources` 标记不能直接使用，虽然1,171实际predict计数和见证
重放不受此错误影响。保留原报告/源码，要求新只读重算器依据非空attempt与实际
验证路径修正horizon并重新核账，不用额外神经预测来掩盖计数错误。
上述10门通过仅是原runner检查结果，不代表独立审计已认证所有资源字段。

主线程纠正账本现已完成：
`le-wm-main/results/P3_ADAPTIVE_RUNTIME_ACCOUNTING_AUDIT_20261027_main_v2/report.json`。
由原始事件、kernel receipts、verification witness重建，拒绝缺失/重复/失败收据，
仅改两条空层导致的horizon；原始文件/源码字节核验未变。9项新测试与57项旧
paired accounting测试共66项通过。根0/2自适应实际max horizon=2，固定为3；
根1/3均为3。完整资源向量相等仅根1/3，四根均在原预算cap内。
1,171预测样本、3,513处理context tokens、动作及选择不变，无新神经执行。
该纠正只反映原始成功事件核账，不追认P3效能，也不覆盖旧报告。

## 强采样对照：先修真实代码缺口，再运行

第一版 `run_pusht_sampling_comparison_20261027.py` 已预封存、未启动 CUDA。
主线程审查发现物理阶段把 `[sha(frame) for frame in history]` 与上游
`history_rgb_sha256` 比较；真实上游字段是 `sha(stack(history))` 单字符串，
该比较必然拒绝所有根，与模型理论无关。这是新 runner 的代码缺口，
不能倒推它解释所有历史 P1 失败。

保留原始源码与 `pusht_sampling_comparison_20261027_v1` prepared 目录，
要求在全新 v2 runner/目录修复，加入真实 stack-hash 回归和已有 live RGB guard，
所有中止保留；不能修改旧封存版本后继续冒用其 source seal。

v2 随后已完成主机 CUDA：
`le-wm-main/results/pusht_sampling_comparison_20261027_v2_prepared/plans_report.json`
状态 `plans_sealed`，4门通过；8份动作计划、2,056条候选记录，实际 planning
4,112 samples + selected replay 16 samples，failed predict calls=0。
这是 CEM 与 random shooting 各自257候选的内部对照，远多于P2每组80 planning
samples；不能跨两套报告冒充相同预算。

v2 物理执行八分支均 completed、每分支10步且分支JSON已保存，但最终收尾
`base.digest(path)` 把字符串传给只支持 `Path.read_bytes()` 的 helper，触发
`AttributeError: 'str' object has no attribute 'read_bytes'`，总报告没有生成。
目录 `pusht_sampling_comparison_20261027_v2_physical` 保留，不能称完整性已通过。
这是新执行器的输出/汇总代码错误，不是模型或理论失败；修复版本只读取已封存
v2 plans，不重新优化，不覆盖v2现场。新版本须复核实际分支重放和完整最终报告。

v3 修复已实际执行：
`le-wm-main/results/pusht_sampling_comparison_20261027_v3_physical/report.json`，
`guarded_calibration_recorded`，7项完整性门全true、failures空。
8分支均完整10步；与旧v2的全部逐步trace和整个branch对象逐项精确相等，
`old_v2_all_eight_exact=true`。新版本不重新优化、模型加载0、神经预测/编码0，
只是执行已由主机CUDA选定的同一动作；这不是CPU神经回退。
旧v2仍保留为汇总失败现场，不补写或覆盖其缺失报告。

| 已知校准根 | CEM物体pose cost | random shooting物体pose cost |
| --- | ---: | ---: |
| 0 | 77.12995 | 237.22720 |
| 1 | 231.12457 | 121.57320 |
| 2 | 94.23540 | 156.35842 |
| 3 | 107.92203 | 193.99588 |

两组official success均0/4；本批CEM在模型成本上四根均较低，但真实物体pose
在根1反而更差。此已知根观察不证明统一任务排序修复，也不据此改超参或选根。
v3的8次精确重放不是8个新独立根，不能加倍统计样本量。

## P4/P5：复算而非开放

`le-wm-main/results/p45_prerequisite_snapshot_20261027/manifest.json` 已完成，
新脚本复算 64 份保存 witness、128 条链边和 384 次历史预测收据。
这是保存内容核账，不是本轮模型 replay。Full D2P P4/P5 仍未满足前置条件；
旧 PointMaze-SRA 受限 P4 不能移用到 Full D2P。公开论文数字只在协议可比时用于比较，
PlanWAM 驾驶任务数字仍是背景，不进入本地配对提升统计。

## 本轮集成验证

主线程实际联合运行14个相关测试文件，335 passed（7.68秒）。包括新P2/P3、
物理动作验证、P4/P5快照、三版sampling执行器、旧witness/预算/guard和原始ViT
参考的逻辑回归。测试用非神经fixture，不等同于CUDA或真实物理效能证据。
本轮额外实际CUDA预测总数为5,971（P3 1,171 + P2 672 + sampling 4,128），
P2另有8个实际终态CUDA编码。新的v3物理重放不增加神经调用或独立科学样本量。
未修改已有封存源码、未训练或CPU神经降级、未删除数据、未提交或推送Git。

## P1 新的最小排除：不能只把线性 nuisance 目录越加越大

独立worker复核Q的对称/幂等性、goal0、残差/Jacobian投影方向和192维单位，
未找到可由改一个转置/索引便解释当前失配的证据。其建议的最小下一假设由
主线程现场读取四份已封存 clean catalog 再复算；四个catalog文件的SHA均与
原始CUDA构造报告一致，未运行模型或拟合readout。

每根分别用全部234个向量，以及去掉原始五个目标后的229个clean编辑向量，
构造 D=(goals[1:]−goals[0])ᵀ。使用现有compiler的原rank规则，不调阈值，
八个D均rank192/192；clean-only最小奇异值为0.00034431、0.00031544、
0.00034032、0.00041373，原rank tolerance约5.4e-12。
四次直接调用原 `compile_metric(all_vectors)` 均按原规则拒绝：
`task metric would discard the entire latent space`。

对这些有限保存向量，精确消掉所有该差分的线性算子只能是零；所以“扩大当前
有限nuisance span便得到全局不变且非零线性Q”不是可直接采用的修复。
这不是物理actor invariance不可能的证明：目录是近似RGB编辑、模型/向量有限精度，
不能外推到真实物理、局部或非线性度量，更不能据此否定D2P理论。
该复算缩小了下一次度量修复范围，不提供P1准入或新物理收益。

另一个实现假设已用真实模拟器排除：installed `_set_state` 内含
`space.step(dt)`，所以不能从请求goal_state默认实际渲染物体不变。主线程新增
`audit_pusht_goal_variant_geometry_20261027.py`，hook官方reset中的第一张render，
在渲染当刻保存实际七维状态并对照原封存20张goal图片SHA。
`le-wm-main/results/pusht_goal_variant_actual_geometry20_20261027/report.json`
实际完成：20张hash全部精确一致，物体位置/模2π角度变化0/20（1e-9容差）。
因此“原五目标nuisance被reset碰撞改变物体污染”在本批未获支持；新目录仍需逐图
验证，不能外推。该审计只有主机模拟器渲染，没有神经前向或新控制未来。
6项非神经回归通过，源码已封存，旧结果未修改。

## P3 增量身份绑定审计已完成

`le-wm-main/results/P3_ACCOUNTING_BINDING_AUDIT_20261027/report.json`
重新验证 protocol 内容SHA、52份当前源码、4根8组、72 kernels、1152规划事件，
重建父节点/frontier/primitive顺序与选中raw动作。总predict仍1171，无新增神经执行。
两条空层horizon修正与主线程main_v2逐字段一致；不依赖旧agent修正报告。
主线程独立运行该65项测试，加上新的P2分解8项，共73 passed。
这不是独立重算transition/SVD或真实效能验证，不能开P3门。

## 实际renderer目录与主线程共同ID核验

已实际主机CUDA编码956张通过几何资格检查的官方renderer目标图，全部1024
grid slots（含68个不合格位置）均保留；不再只依靠sprite编辑目录。
`le-wm-main/results/pusht_simulator_goal_catalog_20261027_cuda_v1/pusht_simulator_goal_catalog_20261027_report.json`
7门通过。按每根共同229个稳定ID比较，agent引起的真实latent距离变化仅略减，
没有恢复物体任务排序。此目录使用特权task/geometry，只是诊断，不是single-RGB部署。
详细边界见 `docs/D2P_SIMULATOR_GOAL_METRIC_DIAGNOSTIC_20261027.md`。
主线程另联合运行6个相关测试文件，125 passed；不是额外物理收益证据。

## P2 主线程独立分解：预测排序与真实编码排序不能混称物理成功

新增只读复算器 `p2_objective_decomposition_20261027.py` 已实际运行，产物：
`le-wm-main/results/p2_objective_decomposition_20261027_main_v1/report.json`。
逐稳定root/arm连接原CUDA计划与真实执行，检查selected模型Q-MSE可独立复算，
计划raw action与完整执行前缀逐值一致、trace终态一致、终图hash、原输入hash。
以原goal[0]和原Q重算8个已保存真实encoder终态；没有新的神经/模拟器调用。

前两根两组均完整10步，composition的预测Q与真实编码Q均低于flat，物体pose
终距也较低；但根0官方含agent状态距离反而较大。故至少这一个完整H2比较中，
仅使真实编码Q更好仍不能推出官方任务更好，不能只把失败归到预测偏差。
根2/3保留中止/不同endpoint，不作固定时域效能比较；根3实际编码Q相同，
但预测排序不同，也不抹去这一预测失配。此诊断未证明唯一根因或理论负结果。

## 新一轮并行工作（尚不记为完成）

独立文件范围推进：P2的40完整双段CEM/random（与P2各80规划+4验证预测样本
匹配，但不能称总算力或动作搜索域完全匹配）；P3八条路径的真实物理执行；
P4两轮真实observe→recompile工程校准；P5第二模型/环境的本地资产与接口验证。
主线程承担集成与实际主机CUDA排程，不允许worker并发改动已封存代码。
新增runner需经测试、源码稳定交付和真实产物核验后才升级本段状态。

### P3 新物理验证已经完成

`le-wm-main/results/P3_MATCHED_PHYSICAL_20261027_v1_cuda/report.json`：
`calibration_recorded`，8项完整性门全true、failures空。8条保存计划各两次真实
重放，16分支全部保留、逐步精确复现；实际154 selected-path env.step，16次
终态CUDA编码，无新predict/规划。物理轨迹和branch先落盘，后编码，防止
encoder/汇总故障吞掉已执行分支。P3执行器新增25项测试；与P2 sampling22项
主线程独立联合47 passed。

前两根两组路径相同，完整10步、任务终距相同。根2固定15步pose cost255.67764，
自适应10步50.61408，官方state distance分别484.92353与342.74641；这是真实
不同端点的描述性观察，不是等horizon或完整资源匹配效能结论。根3两组都在第6步
因RGB guard停止（计划15步），pose28.89755；全部中止保留。全部16分支官方成功
均false；重复重放不增加独立根数。P1/P3仍未准入，不声称compute advantage。

### P2 新等预测预算CEM/random已实际CUDA完成

`le-wm-main/results/p2_matched_sampling_20261027_v1_cuda_v1/report.json`：
`calibration_completed`，5门全true，四根两组全部各80 planning+4 replay，共672个
实际predict samples。源码两份seal均验证。连续raw box的CEM/random与原离散16命令
bank不同proposal域；只匹配预测预算，不冒充总资源或同候选域比较。全部新动作
见证已封存；真实物理执行由另新consumer推进，不能预先把模型分数当任务收益。

物理consumer随后已实际主机执行完成：
`le-wm-main/results/p2_matched_sampling_physical_20261027_v1_cuda_v1/report.json`，
`calibration_recorded`，9门全true、failures空；8分支全部完整10步，共80步及8次
终态CUDA编码，无新增predict。动作与封存选择前缀逐值一致；原输入/源码未变，
全部物理branch先于编码持久化。主线程专属21项非神经测试通过。

主线程用原goal0/Q及已保存终态encoder向量离线复算（无新神经调用）：

| 根 | CEM预测Q | random预测Q | CEM真实编码Q | random真实编码Q | CEM物体pose | random物体pose |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.204371 | 1.210849 | 1.645652 | 1.405299 | 160.94837 | 117.87130 |
| 1 | 0.807126 | 0.826299 | 1.230516 | 1.429423 | 233.95303 | 232.42298 |
| 2 | 1.048019 | 1.072379 | 1.396363 | 1.276730 | 85.51556 | 76.48164 |
| 3 | 0.711050 | 0.849108 | 1.646842 | 1.725593 | 177.67673 | 175.99267 |

全部指标越低越好。根0/2预测Q与真实编码Q的选择排序反转，提示prediction error；
根1/3真实编码Q仍偏好CEM，物体pose却更差，提示任务度量排序不一致。二者可能
同时存在，不能统一归为“只是预测错”或“理论失败”。这是4个已用根、8个已选分支
的描述性核验，不是CEM普遍劣于random、动作谱优势或唯一根因证明；official
success仍0/8。真实编码复算按稳定root/arm连接，复用经8项测试的独立metric函数。

### P4 两轮真实回灌已实际CUDA/物理完成

`le-wm-main/results/p4_receding_runtime_20261027_v1_cuda/report.json`：
`engineering_calibration_recorded`，7门全true，8分支均完成两轮10步，共80物理步。
每分支160 planning+4 verification，共1312实际predict samples；每分支2次真实
三帧context编码，合计16次/48帧；无训练/反向。主线程新增18项测试独立通过。

| 已知根 | fixed物体pose | receding物体pose | fixed官方state distance | receding官方state distance |
| --- | ---: | ---: | ---: | ---: |
| 0 | 162.95602 | 186.94166 | 486.39914 | 450.85019 |
| 1 | 128.60541 | 94.28821 | 367.96785 | 169.57042 |
| 2 | 50.61408 | 66.74792 | 342.74641 | 338.42460 |
| 3 | 131.51512 | 131.51512 | 463.28983 | 463.28983 |

两轮真实观测回灌工程链已打通，但物体pose一改善、两变差、一相同，官方任务指标
不同；全部official success仍false。两臂搜索预算时间分配不同，不是纯反馈消融，
仅4个已知根和短程两轮，不证明长程P4或完整D2P成立。保留此前blocked_upstream
科学状态，不把本实验重命名为已验证机制。

### P5 本地第二环境接口已实际CUDA重放

`le-wm-main/results/p5_local_asset_binding_20261027_v1_assets/manifest.json`
完成41项本地资产绑定。随后主线程实际运行
`le-wm-main/results/p5_local_asset_binding_20261027_v1_tworoom_calibration1_cuda/report.json`：
`replay_completed`，7门全true；固定旧校准根202610360，zero和一个source-supported
候选各独立两次，共4实际predict calls/samples，3真实history帧和1goal帧CUDA编码。
15次真实warmup控制用于复现context，没有执行候选future。保存预测成本匹配、
重复预测一致，参数/buffers与输入绑定未变。训练action列完整packet pool
870809×5×2的SHA重核一致，但未重算12GB HDF5全文件SHA，不能扩大来源认证范围。

这是另环境同LeWM架构的接口核验，不是独立架构P5或任务收益。固定路径下
DINO-WM dynamics缺少，本地DINOv2 encoder缓存不能代替完整世界模型；JEPA
archive存在也不等于其身份/runtime已验证。P5仍不准入，无outcome开放。

主线程将本轮6个新增/增量测试文件联合运行，165 passed（14.64秒）。新科学
准入仍为零：本轮实际新增1988个predict samples（P2 672 + P4 1312 + TwoRoom4），
P3另16个终态编码；这些计数不能混成新增独立根或方法有效性的证据。
后续P2真实consumer另新增8次终态CUDA编码，不增加上述1988次predict或独立根。
纳入该consumer后的最终7文件联合回归为186 passed（16.00秒），git diff --check通过。

P4事后只读复核另确认：第二轮RGB hash链为旧第2/3帧加真实执行终帧，between
packets为旧第2包加刚执行包，重新归一化与保存action_history逐值一致；未发现
所查历史动作错位。保存见证核验不等同于重新编码核验。P3独立review核对全部
journal/170张图像SHA、8对重放逐步RGB序列及终态编码一致。旧replay gate本身
没有覆盖全部RGB比较，此次补证不代表旧gate已修复，未来版本仍应补此检查。

## 完整包内动作空间：新增真实实验与主线程评分归因

详见`docs/D2P_FULL_PACKET_CAPACITY_AUDIT_20261027.md`。新全时序编译实际完成
572 CUDA predict样本，80候选独立重放精确一致；从重复二维命令到包内十维动作，
四根模型响应秩2→10，但全部八组选中动作仍等于greedy，没有P1非贪心见证。
后续80条真实物理尝试共791步、157次CUDA编码，3条guard中止缺固定十步终态，
原`integrity_fail`及false编码门完整保留。不是全绿实验或理论否定。

主线程已完成新评分层级离线审计：原线性末段换成非线性模型末段Q，八组选中
候选全部不变；模型→真实编码排序与真实编码→物理pose排序均存在反转。
没有物理结果重选，没有训练或新根，没有理论负结论。四新文件联合118 passed，
新增评分审计另4 passed；两张实际结果图已目视检查。

用户本轮要求加并行后，已新派四路：P2完整20维CEM/random等预测预算、P3已存
真实质量—成本审计、P4相同轮次预算的反馈对照、P5独立模型资产身份核验。
写入域独立，GPU由主线程调度。派工不等于实验完成；实际核实结果如下。

### P2 full20：规划及物理已完成

`results/p2_full20_sampling_20261027_v2_cuda_v1/report.json`（相对`le-wm-main`）
已实际主机CUDA完成，5门全部true，每臂80planning+4replay，共672预测样本。
两个5×2包的20个raw分量独立，不再把每包限制为同二维命令重复。
主线程新consumer的输入预检通过后，已执行
`results/p2_full20_physical_20261027_v2_cuda_v1/report.json`：9门全部true，
8分支全部完整十步，共80真实步骤、8次终态CUDA编码，无额外predict。
新consumer明确禁止以guard提前端点替代固定十步终态；本批没有中止。

主线程按root SHA/arm连接，用已验证metric函数离线复算真实终态Q：

| 根 | CEM预测Q | random预测Q | CEM真实Q | random真实Q | CEM pose | random pose | CEM官方distance | random官方distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.347268 | 1.365556 | 1.587044 | 1.793473 | 158.54237 | 105.21132 | 296.96307 | 446.66419 |
| 1 | 0.990835 | 0.747579 | 1.151317 | 1.244013 | 70.98106 | 145.68339 | 278.60033 | 145.09160 |
| 2 | 1.110170 | 1.222682 | 1.403625 | 1.791055 | 88.94899 | 116.19402 | 295.82998 | 289.74243 |
| 3 | 1.510320 | 0.871803 | 1.606553 | 1.999400 | 21.15836 | 73.46920 | 237.86461 | 426.59065 |

全部越低越好，official success仍0/8。完整动作空间CEM的pose在三根优于random，
不同于旧constant2批的0/4，但不是跨批正式效应证明。预测Q与真实Q在根1/3反转，
真实Q与pose在根0不一致；真实Q与官方distance在根1/2不一致，保留不利指标。
这证明对照现在确实覆盖完整动作空间，不证明D2P胜过CEM或P1/P2成立。
新sampling33、physical7、P3审计33项主线程联合73 passed。
后续worker只读复核确认sampling两份seal共72源码条目、physical三份seal共84条目
均匹配当前及归档字节；八条完整动作见证、终态normalized历史、RGB、journal和
逐分支文件内容/哈希一致。该复核没有新增神经或物理工作。

### P3 真实质量—资源分类已完成

`results/p3_quality_cost_audit_20261027_v2/report.json`与`report.md`已生成，
独立核验386输入/源码哈希与5产物哈希。四根中仅根1为完整路径上的已记录资源
匹配，根3只为partial guard endpoint匹配，根0/2不匹配。
16条重复重放全部保留，4条guard中止不删；1152planning+19verification按每条
已选计划记账一次，不因物理重复两遍翻倍。没有重复样本置信区间、FLOPs等价或
adaptive superiority主张。来源编译/编码成本仍明确排除，P3不准入。

### P4 逐轮同预算反馈对照已实际完成

`results/p4_feedback_matched_20261027_v2_cuda_main_v1/report.json`：8项门全部true，
全部8分支完成两轮十步，共80真实步骤；每分支160planning+8verification，合计
1344实际CUDA预测样本；两臂都做两次真实三帧编码，合计16调用/48帧，imagined臂
第二轮只用已封存模型中间context选择，不用第二轮真实编码选择。这样匹配了逐轮
候选数/horizon/预测预算/编码帧数，消除旧fixed80pairs vs receding40pairs的分配混淆。
TF32显式关闭，每次env.step前attempt持久化；没有训练或结果重新选择。

主线程独立复核全部3份source seal、逐轮真实计数、实际/想象context绑定及相同
候选schedule：imagined第二轮逐值等于第一轮模型mid；actual第二轮逐值等于
真实RGB编码context。两者latent最大差依根为1.40105、1.32579、1.38790、0.69403，
不是意外把同一个context送进两臂。前三根第二轮选择一致，第四根选择不同。
另用已保存npy独立核验八分支：第二轮RGB前两帧精确等于首轮后两帧，末帧精确
等于真实执行终帧；between旧包移位加实际执行包正确，NumPy float32重归一化
与保存action_history逐值一致。未发现所查回灌动作/帧错位，无新增神经调用。

| 根 | imagined pose | actual pose | imagined官方distance | actual官方distance |
| --- | ---: | ---: | ---: | ---: |
| 0 | 186.94166 | 186.94166 | 450.85019 | 450.85019 |
| 1 | 94.28821 | 94.28821 | 169.57042 | 169.57042 |
| 2 | 66.74792 | 66.74792 | 338.42460 | 338.42460 |
| 3 | 73.09841 | 131.51512 | 350.79261 | 463.28983 |

前三根实际路径/任务指标相同，根3真实反馈更差；official success仍0/8。
因此旧批观察到的局部收益不能直接归于反馈。仅两轮、四个已用根、固定bank，
没有证实反馈普遍有害或长程P4理论失败，也不主张完整总FLOPs或wall-time相等。
P1/P4准入保持false，新增56项非神经测试通过。

### P5 独立架构候选已确认存在，原生加载仍在推进

`results/p5_architecture_readiness_20261027_v2/report.json`已保存实际静态检查。
三个JEPA-WM权重SHA与v1相同，元数据有完整六层AdaLN action-conditioned predictor，
并非只有encoder。该架构不同于LeWM；原生PushT配置含proprio，不得伪称RGB-only。
此时只做有界ZIP/pickle-opcode检查，不执行反序列化；不能由字符串元数据推导
严格state_dict形状或可运行性。DINO-WM dynamics仍缺失。
23项安全/逻辑测试通过；没有下载、解包、训练或神经CPU实验。
已继续派工受限weights-only/meta形状与源码对应核验，尚不记为完成或P5准入。

随后v3已交付，主线程也已实际完成三任务CUDA严格原生加载：
`results/p5_jepa_strict_metadata_20261027_v3/{pusht,pointmaze,wall}_native_strict_cuda_main_v1.json`。
当前入口绑定`report_frozen.json`，不能使用旧`report_native_interfaces.json`：后者
记录的是收据补充前源码字节，当前入口会拒绝。旧报告与源码均保留，没有覆盖。

三份均`native_dynamics_strict_load_only`，每份predictor92键、proprio encoder2键，
先比对真实原生state_dict，再strict=True加载，missing/unexpected/shape/dtype差异
全部为空；所有状态张量实际位于cuda:0，所有参数冻结、所有子模块eval，源码与
checkpoint前后SHA一致。没有模型forward、encoder构造、数据集读取、训练或
神经CPU fallback。PushT/PointMaze proprio为4维，Wall为2维；首次把4维套到
Wall的形状错配收据保留，后续按原生接口核对，不能冒称Wall权重损坏或放宽门。

v3本身11项非神经测试通过（与v2联合34项），补齐的是“权重存在→严格原生CUDA
装载”，尚未验证数值有限性、动作敏感性、encoder等价、预处理与完整D2P。
因此`model_forward=false`、`encoder_runtime_verified=false`、`D2P_ready=false`、
`P5_admission=false`都保持，不把三次装载计入预测样本/真实任务收益。

实际命令（三次分别替换TASK为pusht、pointmaze、wall；输出均为全新文件）：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.p5_jepa_strict_metadata_20261027_v3 --validate-cuda --metadata-report results/p5_jepa_strict_metadata_20261027_v3/report_frozen.json --task TASK --output results/p5_jepa_strict_metadata_20261027_v3/TASK_native_strict_cuda_main_v1.json
```

本轮十文件主线程最终联合回归274 passed（14.40秒），git diff --check通过。
进一步纳入前轮七文件的17文件联合回归为460 passed（26.76秒）。
本轮新增full20基线和匹配反馈对照共2016实际predict样本、160真实步骤；前述
全时序编译的572预测/791步骤/157编码另行列明，不混为新独立根或科研准入。

## 后续新增：未来几何候选与真实能力探针

详见 `docs/D2P_GEOMETRY_REAL_CAPABILITY_AUDIT_20261027.md`。本节记录真实完成
产物，不用“已启动”替代结果。新v2数值验证失败保持原状：float32 tanh参考与
CUDA相差1.19209e-7；v3仅用float64数学参考，不改1e-7门槛/科学目标，实际
3092CUDA预测、936Jacobian、179712标量VJP方向、96独立重放，六项门全通过。
八个赢家仍来自原共同十候选，新非贪心关系仅改变greedy参考，P1保持false。

新96分支物理批保存952步骤/190编码，两分支第6步RGBguard停止，故固定第10步
编码门false且原integrity_fail保留；其他完整性门全true。关键角色均完整，
selected的真实Q更好但物体pose更差，不能以含agent/速度的官方state距离掩盖。
非神经独立分析12测试通过，两张PNG/PDF已生成并查看。

接着实际完成固定能力探针：仅按封存模型资格得到根1，三角色×41后缀，123分支、
1230真实步骤、246CUDA编码、6完整响应组、六门全true，新增预测0。selected在
两档小探针的物体响应均为0而RGB响应非零；greedy/zero物体响应非零。
各自RGB残差拟合较好不等于几何更好：固定公共残差后selected相对greedy更差。
主线程独立检查真实动作/state/phi/差分/Q/拟合KKT及源码日志RGB哈希通过，
九项含损坏收据的回归通过。该结果限定已用根/五步/小探针，不构成理论否定。

P5 v4真实数据CUDA前向已完成：episode0 frames0/5/10→15，六门全true，模型
冻结不变，新增动作输出差17.7821、重复差0、同HF接口差0、预处理差0；真实
target patchMSE .0349896仅描述。encoder1+同HF直接1/proprio1/predictor3，
normalized zero是原生动作均值不是rawzero；同HF自比较不是官方DINO等价。
输出 `le-wm-main/results/p5_jepa_real_forward_20261027_v4/pusht_real_forward_cuda_main_v1.json`，
P5/完整D2P准入仍false。

随后v5固定episode1..8真实CUDA动作保真对照也实际exit0完成：
`le-wm-main/results/p5_jepa_action_fidelity_20261027_v5/pusht_action_fidelity_cuda_main_v1.json`。
八个episode全部保留、每个四种动作仅替换第三个新包，两历史包/proprio不变。
真实动作相对normalized-zero、negative-normalized的真实target patchMSE均8/8更低，
对照减真实的平均差分别+.19568178/+.31666260；相对时间倒序只有4/8更低，
平均差−.00250086。实际predictor32、proprio8、native encoder8及其同HF直接
接口合计16 forward，全部计数匹配、权重/源码/数据哈希不变。此处是有正向
动作条件保真、但细时序辨识不稳定的有界证据，不是物理counterfactual或P5成功；
官方DINO等价仍未证，不能用HF shim自比替代。

P3新独立前沿重算已落地：
`le-wm-main/results/p3_budget_frontier_20261027_v1/report.json`。
按重复执行配对记录计为2 full-path、2 guard-partial、4 unmatched；等价于
旧报告按root计的1/1/2，不新增根。所有16次执行、4次abort保留，1152规划+
19verification=1171预测样本仅收一次共享规划；154真实steps、16端点编码。
完整配对均tie，P3 false；未覆盖上游编译、加载、reset、SVD/FLOPs等明确unknown。
该线独立回归223项通过，主线程联合回归另行核验。

用户要求新增智能体后，已新增P2/P3独立实现线，复用P4/物理复核线；P5线继续。
主线程仍承担能力探针执行、独立复算、图与整合，不将任务全部外包后等待。
本轮主线程七文件联合回归182 passed（37.16秒），git diff --check通过。
两名独立只读复核智能体均完成后已释放；P2/P3实现、P4审计和P5架构线继续。

## 新增实际完成：反馈定位、P2采样基线与固定H3预算（2026-10-27）

以下均已由主线程执行并核对最终报告，不将prepare或进程启动计为完成。

### P1：分离想象前缀漂移与真实状态后的局部响应误差

`le-wm-main/results/pusht_geometry_feedback_response_20261027_v1_prepare_v1/report.json`
已完成249次CUDA预测、3次encoder调用/9帧、无新增物理步骤和backward。
保留同一已用根、selected/greedy/zero三角色、各41个固定后缀和相同旧历史；
只将prefix后的想象上下文换为真实第5步RGB。selected的零后缀Q预测MSE由
0.32916310降至0.00916599，全部41后缀均值由0.33597490降至0.00795652。
但selected的小探针响应余弦在epsilon=.05仍为-0.040478，epsilon=.1为0.791478。
这定位出明显的滚动上下文漂移，同时说明重新编码不能自动修复全部局部响应。
它不是新物理效益实验，也不能由单根校准推断泛化或理论失败。独立复算与两张
已查看PNG/PDF位于`results/pusht_geometry_feedback_figures_20261027_main_v1/`；
该算术/绘图模块5项测试通过。

### P2：严格准入仍关闭，新增MPPI-inspired实际比较资产

`results/p2_composition_admission_20261027_v1_audit01/report.json`确认完整性有效，
但独立P1、无组合SRA、完整资源与组合增量等科学槽位尚缺，P2=false。
新`p2_full20_mppi_20261027_v1`沿用clip-Gaussian启发式，不冒称官方MPPI；
每根40个完整H2候选=80规划predict，另4次replay，四根共336实际CUDA预测，
五项门全true。输出`results/p2_full20_mppi_20261027_v1_prepared01/report.json`。
对应真实执行`results/p2_full20_mppi_physical_20261027_v1_cuda_main_v1/report.json`
已完成40物理步骤、4次CUDA端点编码、四根全部完整、九门全true、failures为空。
物体pose误差按root0..3为137.78012、128.85874、113.18528、114.56258，
官方/object success均0/4；不能据此声称D2P胜出。与既有CEM/random的比较需
按稳定根身份和相同动作/目标合同配对，而不能按JSON数组顺序拼接。

### P3：同深度/同predict预算调度已实际CUDA完成

`results/p3_fixed_horizon_budget_schedule_20261027_v2_cuda_main_v1/report.json`
完成四根×两臂完整H3=15步计划、1176实际predict；每根每臂144规划+3验证。
uniform固定[4,4]，dynamic在root0/2为[4,4]，root1/3为[3,5]；
dynamic减uniform模型代价分别0、+0.1109636281、0、+0.0010649151。
chi来自含自主漂移的有限H1位移响应，不是action Jacobian；全部结果保留。
这消除了旧深度不匹配的比较缺口，但还不是物理质量—成本优势或P3准入。
完整H3物理consumer随后已由主线程实际执行：
`results/p3_fixed_horizon_physical_consumer_20261027_v2_main_v1/report.json`，
guarded_calibration_recorded，102实际步骤、8次CUDA编码、source/model核验true、
failures为空。四根有3组完整15/15步配对、1组双方第6步guard中止；中止不补跑，
保留为guard-partial，不冒称完整120步。root0/2完整配对pose差为0，root1的
dynamic减uniform为+24.9389224（更差）；root3部分配对差0不作H3效益结论。
这不是动态调度优势；当前chi仍是有限位移信号，尚不能代表理论中的action
Jacobian不可达残差。该语义缺口需新版本修复，不能据此直接否定C3。

### P4/P5：接口证据与待执行的独立验证

P4新准入审计`results/p4_feedback_admission_20261027_v1_main_v1/report.json`
为interface_audit_passed_scientific_blocked，复核80步骤、1344预测、16次编码/
48帧/4032上下文token；缺独立根、长程机制和完整资源等，P4保持false。
P5仅新增下载并校验4.6MB官方DINOv2源码，commit为
7764ea0f912e53c92e82eb78a2a1631e92725fc8；没有新下载权重/大数据或CPU神经前向。
官方与HF原生实现的独立等价v6随后已由主线程实际CUDA完成：
`results/p5_jepa_official_equivalence_20261027_v6/pusht_official_vs_hf_cuda_main_v1.json`。
官方8/HF8 forward，strict loads及源码/模型冻结核验通过，预处理最大差0；
八episode的特征门全部false，状态paired_equivalence_failed，不能以exit0说等价。
patch最大绝对差2.69262–3.92194、RMSE .10976–.12542；位置嵌入差.01440058。
已确认原生位置插值存在实现差异，但尚不能把全部差异唯一归因于它。
因此旧HF编码链不能追认为官方等价；新官方encoder部署consumer另起版本，
不改v6阈值、offset或封存结果来消除失败。

### P1实际修复尝试：反馈后重新编译动作，而非只重算误差

主线程新增并封存`pusht_feedback_suffix_compile_20261027_v1.py`，十项非神经
测试通过，并实际主机CUDA完成：
`results/pusht_feedback_suffix_compile_20261027_v1_prepare_v1/report.json`。
三角色×想象/真实反馈上下文全部六槽保留；原Q、目标、damping、动作logit球和
谱半径不变，每上下文生成十个full10后缀，以真实非线性模型Q成本固定argmin。
动作生成不读物理终点向量或诊断分数；每赢家独立replay精确一致。
78实际预测、6 Jacobian、1152标量VJP方向、12 SVD，模型不变、无新增编码/物理。
selected的想象臂选择suffix1，真实反馈臂改选suffix6；其他两角色两臂均suffix6。
这证明反馈确实改变该编译器的可执行输出，尚不证明物理获益或独立P1通过。
六分支5+5步物理consumer随后已主线程实际完成：
`results/pusht_feedback_suffix_physical_20261027_v1_prepare_v1/report.json`。
60真实步骤、12次CUDA编码、七门全true、failures为空；所有第5步原前缀RGB
逐像素精确复现。selected反馈臂pose为175.458326，想象臂113.291889；
greedy两臂为138.227580/139.891343，zero两臂113.291889/85.776871。
这次并未带来稳定任务收益，六分支官方/object success仍false。selected新动作
已造成物体变化，不能继续套用旧小探针“物体全静止”的解释；大动作终态投影
预测MSE仍.238092，旧零后缀误差下降不足以保证新编译动作准确。
独立轨迹与接触恢复复核另行推进；不能将本批失败升级为理论否定。

P4另有静态真实形状几何审计已完成：
`results/p4_saved_contact_geometry_20261027_v1_main_v1/report.json`。
按原安装环境T形多边形、pymunk COG变换、推杆半径15与原collision_slop=.1，
核对80条P4步记录、123分支/1230能力探针步骤及持久化输入哈希。
selected的41个五步后缀总205个快照均非近接触，表面间隙15.0291–23.1264，
最大物体顶点步位移0；greedy/zero各145次快照近接触且有物体运动。
这解释了小探针观察到的物体静止，但静态快照不能证明子步无接触或力因果，
也不是新增物理实验/图像分割或P1/P4准入。

### P5修复部署链后已实际完成新协议，不追认旧等价

v7使用未修改官方DINOv2代码、严格映射现有缓存权重，再接原生JEPA predictor/
proprio；不构造HF encoder，不改变官方offset或旧结果。主线程完整审查后，
实际CUDA完成固定episode1..8：
`results/p5_jepa_official_action_fidelity_20261027_v7/pusht_official_action_fidelity_cuda_main_v1.json`。
official_conditional_accuracy_recorded，官方encoder8/proprio8/predictor32，八行
全部保留，source/data/model冻结与计数门通过。actual对normalized-zero和
negative-normalized均8/8目标patchMSE更低，control减actual均值为
+.19922992/+.31800756；对reverse只有5/8更低，均值−.00100688。
这是改用正确原生接口后的新条件保真结果，不是新独立holdout、物理counterfactual、
独立官方权重来源认证或P5完整D2P；原v6等价失败和旧v5条件结果均保留。
v6/v7主线程联合29项非神经回归通过，没有CPU神经fallback。

P2独立稳定root配对分析也已完成，
`results/p2_mppi_matched_comparison_20261027_v1_audit01/report.json`，来源/预算/
动作/目标/实际完整终点核验通过，四根×三基线全保留。MPPI-inspired减CEM的
pose均值+38.68898、1胜3负；减random为+13.45719、2胜2负，仅已用根描述，
没有独立置信区间或全管线算力相同的结论。未开P2。无组合SRA v1拒绝报告
保留，并另开联合H2谱参数化v2新协议；不把原逐prefix重编译器改名为无组合。

### 后续实际CUDA：无组合SRA与真实action-Jacobian调度

联合H2无组合SRA v2已由主线程实际完成：
`results/p2_no_composition_sra_20261027_v2_prepared01/report.json`。
calibration_completed、八门全true、四根全保留、352实际predict、4个192×20
联合Jacobian、768标量VJP方向、8 SVD、1056处理上下文token、模型不变。
它不按prefix分支重编译suffix；固定40个完整H2谱/DLS候选，原Q非线性argmin。
每根80候选预测+4重放+4名义/Jac编译预测=88，不能与84预测CEM/random声称
等全预算；20维联合logit球也不等于原分包球或相同proposal分布。
其模型代价按root0..3为1.391151、1.236501、1.209249、1.161770，仅描述。
物理consumer已由主线程实际运行完成：
`results/p2_no_composition_sra_physical_20261027_v2_cuda_main_v1/report.json`。
四根均完整10步，共40真实步骤、4次CUDA终点编码、九项完整性门全true。
root0..3的实际pose cost为256.187666、60.885532、66.663624、103.790022；
不以模型代价替代这些终点。独立稳定root匹配分析另行生成，不能据此宣称组合
增量、独立泛化或全预算匹配；新增编译成本仍全部保留。
稳定root匹配分析随后实际完成：
`results/p2_sra_matched_comparison_20261027_v1_audit01/report.json`。
四根×四臂16分支、857份文件绑定核验，inputs不变。SRA减CEM/random/
MPPI-inspired的物体pose平均差为+36.974016/+11.742227/−1.714968；
相应官方state distance差为+84.893335/+35.185615/+43.902940，四臂各0/4
official success。不能以单一pose指标或MPPI对照小均值优势掩盖其他指标。
这仍不是组合消融，SRA352预测对其余各336且另有Jacobian/SVD，未主张总预算相等。

P3 action-Jacobian语义修正版v3也已实际完成：
`results/p3_action_jacobian_budget_schedule_20261027_v3_cuda_main_v1/report.json`。
1176实际predict、8完整H3 root-arms，source_failure/failure为空。
用同冻结模型/原上下文合法继承的四份192×10 root-H1导数，
`r=Q(goal−zero_nominal), K=QJ`的col(K)残差替代含漂移位移span；原分配公式
不变。继承获取成本仍单列unknown，不计成0或重复按arm加总。
dynamic的四根分配为[4,4]/[3,5]/[4,4]/[4,4]，与uniform模型代价差为
0/+0.1109636281/0/0。修复信号语义并没有让这条固定调度法产生模型优势；
不能靠继续事后调阈值宣称成功，也不能据此否定其他动态时域/资源策略。
v3新物理consumer已由主线程实际完成，并非复用v2终点：
`results/p3_action_jacobian_physical_consumer_20261027_v3_main_v1/report.json`。
共102真实步骤、8次CUDA终点编码，source_verified/model_unchanged均true。
root0/1/2各两臂完整15步，dynamic减uniform实际pose差为0/+24.9389224/0；
root3两臂均在第6步因canonical agent pixels为0触发观察缺失护栏，9步未执行。
该部分终点保留但不当作完整H3效益。完整配对中未见调度优势；不能否定整个
自适应资源思想。独立配对审计另行运行。主线程32项回归通过。
独立配对/护栏/标签算术审计随后已完成：
`results/p3_action_jacobian_outcome_audit_20261027_v1/report.json`，
saved_outcomes_verified、source seal verified；主线程37项新测试通过。
三对完整H3的pose均差+8.3129741（两平一负），双方official success均0/3。
root1的official position error反而减少64.8410、真实编码Q减少.0382164，
但official state distance增加169.3358；所有指标单列，不用代理改善覆盖物体损害。
root3短轨迹完全留在记录中，但不进入该审计的完整配对差值。

### P5原生动作响应：主线程真实CUDA已完成

`p5_jepa_native_action_response_20261027_v8.py`及其独立外层协议文档封存，
主线程完整审查后19项非神经测试通过。固定已用episode1、原生normalized
十维动作，十列double-backward JVP与两个固定epsilon的全轴中央差分；
预期51 predictor调用，保存完整98304×10响应矩阵供非神经独立复算。
该版本不执行求解动作、不产生物理收益证据、不打开P5。主机CUDA已实际完成：
`results/p5_jepa_native_action_response_20261027_v8/episode1_native_action_response_cuda_main_v1.json`。
native_action_response_recorded，51 predictor/1 official encoder/1 proprio encoder，
10 JVP与20次autograd.grad请求/完成，全部计数、帧数、阶段及冻结门true。
20组FD/JVP最大相对L2为.0009243744，最小cosine为.9999995782；这些是
全量保留的诊断数值，不是事后新增阈值门。实测CUDA峰值allocated约1.18GiB。
原始目标MSE .0664345921，固定ridge线性残差MSE .0610866416；该版本
未评估求解动作，不能把线性预测下降记成真实或非线性收益。
独立智能体负责保存矩阵的Gram/rhs/解/谱复算。主线程另写v9固定delta原生
非线性重放consumer，5项非神经测试通过并另行封存；仅三次predict，不调参/
筛候选/执行真实动作。其目标来自已观测frame15，明确只算拟合/编译诊断。
v9随后主线程实际CUDA已完成：
`results/p5_jepa_native_delta_replay_20261027_v9/episode1_native_delta_cuda_main_v1.json`。
native_delta_replay_recorded，encoder1/proprio1/predictor3、exact replay与全部冻结
计数门true。非线性candidate MSE .0659379351，较nominal下降.0004966569，
但仍比线性预测高.0048512935；因此不能把线性下降量作为实际下降量。
这是原生可微响应→固定动作参数→非线性精确重放的局部实现闭环，不是P5
真实控制、独立泛化或机制闭环。两轮notes保留完整解释，不从已观测target
拟合结果推导prospective planning收益。
独立NumPy-only已存矩阵审计也已完成：
`results/p5_jepa_native_response_audit_20261027_v1/episode1_saved_arrays_audit_v1.json`。
saved_array_arithmetic_consistent，24项来源/调用/哈希/残差/Gram/rhs/解/谱检查
全true。它未重新运行模型；FD原始向量未保存，只能核对摘要代数，不能将
其命名为“独立重算FD通过”。主线程该审计和新v9联合21项非神经测试通过。

主线程本轮九文件联合186项非神经回归通过（32.31秒），git diff --check通过。
后续SRA v2和P3 v3分别独立19/32项回归通过，不与重复测试累加成独立证据。

当前六个独占并行执行单元（另有主线程）：Meitner从SRA预算核验转入真实P2
组合实验的可运行前置和最小实现；Poincare完成SRA匹配分析后转入固定fullH2
梯度MPC基线；Arendt完成v3物理配对审计后转入已有论文reported值的独立表格
与对照可比性；Ohm完成接触/运动拆分后转入固定三轮反馈runtime；Euclid完成
原生JEPA响应及独立矩阵审计后转入官方物理接口/reset/动作坐标桥接；Harvey
完成P1目标合同审计后检查既有identity/Q对照，避免重复并定位下一可修合同。
这些下一批实现尚待审查/实际执行，不计成科学完成。
主线程已直接写P1与P5新编译/重放consumer并执行全部上述主机CUDA与物理
验证，负责跨线审查和本进度页，不是只派工。任何prepare尚未实际运行的任务
都不能记成实验完成；独占文件和新版本规则避免交叉修改封存产物。

### 新一批并行产物：梯度基线与三轮反馈已实际执行

固定四根 full-H2 gradient MPC 主机 CUDA 已完成：
`results/p2_full20_gradient_mpc_20261027_v1_prepared01/report.json`。
共336次predict、156次标量目标反传及投影更新，MATH SDPA、冻结模型不变。
这是40候选的固定开环梯度优化，不是官方GRASP复现，也不是receding-horizon
MPC优势证明。独立物理consumer由主线程实现、封存并执行：
`results/p2_gradient_mpc_physical_20261027_v1_cuda_main_v1/report.json`。
四根共40步、4次CUDA终点编码，六项完整性门均true，所有分支完整H2。
根顺序0–3的pose终距为61.03108024、157.97043688、64.88951673、84.57587902，
official success均false。与CEM/random/MPPI-inspired/SRA的稳定身份配对算术由
独立智能体继续核验；反传、投影和不同attention backend不能记成零成本。

固定单校准根三轮反馈主机CUDA及真实执行已完成：
`results/p4_multiround_feedback_runtime_20261027_v1_prepare_v1/report.json`。
status=calibration_recorded、failures为空、模型不变；imagined与actual两臂各20步，
在5/10/15步重编译。80次predict、6个Jacobian、1152标量VJP、12次SVD，
6次视觉编码覆盖18帧。另计30次setup执行，不隐藏在40次selected执行之外。
三轮winner imagined=1/1/1、actual=6/8/3，证明两种反馈历史确实改变选择；
不证明真实收益。独立智能体正在从原始事件日志复算终态任务指标与护栏。

真正配对的联合H2无组合与条件后缀组合v2通过主线程21项测试及源封存verify，
主线程已完成新CUDA执行：
`results/p2_composition_paired_20261027_v2_prepared01/`。
实际report为calibration_completed，十项冻结/计数/完整性门全true。
协议为相同前缀槽位、40候选全保留、条件后缀重新线性化；四根实际1008次predict、
164个Jacobian、31488数学标量输出VJP方向（不等于测量的backward API次数）、
168次规划SVD；组合与无组合不具等总资源合同，P1/P2准入保持false。
对应八分支物理consumer已由另一智能体独占实现，主线程完整审查并通过源验证后，
已实际执行完毕：`results/p2_composition_paired_physical_20261027_v2_main01/report.json`。
calibration_recorded、九项门全true、failures为空，共80次selected steps、8次CUDA
终点编码，八臂全部完整H2。组合/无组合pose根0–3分别为185.4705/256.1877、
97.3999/60.8855、82.4475/66.6636、114.8101/103.7900；八臂official success全false。
这批已用校准根是一胜三负的混合结果，不是独立P2通过或理论否定。独立结果
consumer与负结果代码合同归因分由两个智能体负责，不据结局重选动作。

原生JEPA物理桥审计查到可操作的环境合同阻塞：旧归档5维pose与agent速度不足
以认证完整block动力学状态，H264帧也不是可精确复原的原始RGB；当前安装的
pymunk/gym与原生版本不符，scikit-image缺失。不得据此宣布模型理论失败，
也不得把猜测的隐藏速度当作精确reset。独立智能体正在准备小体积隔离运行时与
前瞻新根方案，不修改现有依赖。当前磁盘仅约1.5GB余量，不进行大规模下载。

六智能体继续分别承担P5原生环境、P4真实结果、P2梯度匹配对比、已发表基线
数值表、P1训练/部署契约以及P2组合物理consumer；主线程负责CUDA与整合。
文献reported数值必须保留原任务/指标/模型/预算，不与本地校准数字混排排名。
梯度五臂稳定身份比较已生成独立报告：
`results/p2_gradient_mpc_matched_comparison_20261027_v1_audit01/report.md`。
gradient减CEM/random/MPPI-inspired/SRA的pose均差为+7.2090/−18.0228/−31.4800/−29.7650，
全部方法official success均0/4，不宣称同backend或等总资源。主线程联合回归中
156项通过，P4零步护栏fixture一项失败交回修正；不把测试问题当理论负结果。
随后主线程重新启动当前已修fixture的P4测试，独立44项全部通过（5.09秒）。
step0空白初帧明确应由root身份门拒绝，不放宽生产检查。

### 后续实际推进：P4终态复算与P5原生依赖接通

主线程读取了独立P4 journal算术报告：
`results/p4_multiround_outcomes_20261027_v1_main_v1/report.json`，
saved_arithmetic_audit_passed，两臂均complete20。actual减imagined的pose为+68.1709449，
物体XY为+76.0117108、wrapped angle为−.3920383；official state distance为−60.7454099，
但official position error为+67.2337761，双方success均false。不同指标方向相反必须
并列：state distance含速度，不把其下降写成物体位置或最终任务成功。已安排独立
逐轮机制拆分，不用再添加反馈轮数掩盖当前任务损害。

P5小体积隔离原生环境已由主线程真实安装，不再只是安装计划：
`results/p5_jepa_native_physics_runtime_20261027_v1/bootstrap_artifacts/installer_receipt.json`。
两个stage均exit0，新增约39.55MB（150MB限额内），保留安装日志/版本/完整文件hash，
gym0.23.1、pymunk6.8.0、scikit-image0.25.2等装入独占`.jepa-physics-site`，
未替换现有venv依赖、未下载torch、无神经CPU回退。
`results/p5_jepa_native_physics_runtime_20261027_v1/probe_main_v1.json`为
native_import_ready_physics_not_executed，problems为空、sources不变。
仅原生导入通过，不等于物理或ABI行为通过；下一步独立新根factual collector执行，
使用两个独立同seed/config环境精确重放，而非补造旧episode1隐藏状态。
主线程runtime与P1来源审计联合30项测试通过；P1当前CHW预处理与live六帧逐元素
相同，NumPy横纵轴分支/JPEG差异仍需checkpoint训练来源与真实调用路径证据，
不能把这些源码差异直接宣布成P1已证根因。

### P5新鲜原生物理→CUDA链已实际接通

主线程完整审查后执行固定seed2026102701的两个独立原生环境：
`results/p5_jepa_factual_pilot_20261027_v1/pilot_main_v1/report.json`。
fresh_native_zero_pilot_recorded，32帧/30control calls/302物理子步，16个对应时刻
RGB/state/proprio/full-public-state/transition均逐文件精确相同；全源不变。
继而执行预先固定15步非零u的v2，两臂全量留存且同样16对精确相同：
`results/p5_jepa_factual_nonzero_20261027_v2/factual_main_v1/report.json`。
fresh_native_nonzero_factual_recorded；这是同seed校准，不称独立新cohort。
没有恢复旧归档的未知隐藏状态，没有用物理outcome选动作。

主线程新增v10 consumer，并实际主机CUDA完成原生official encoder/proprio/dynamics
factual排序：`results/p5_jepa_fresh_fidelity_main_20261027_v10/cuda_main_v1/report.json`。
输入是lossless帧0/5/10→真实帧15，两个历史packet不变，只干预第三包；目标仅作
事后factual标签，从未用于动作优化或选择。保存完整4帧patch features和四臂successor。
actual/normalized-zero/negative-normalized/reverse-steps的MSE分别为
.0355068836869/.0388793724527/.0462483527326/.0347221403768。
actual优于前两控制但不优于reverse；单seed不能说明稳定action fidelity，更不是P5。

独立审查确实找到v10的失败记录代码漏洞：晚一臂失败会丢已完成数组/代价，
post hook漏记失败forward attempt，收尾SHA异常可能遮蔽原错误。已保留v10字节，
主线程另写v11修复，并通过12项测试（含纯常量数组接口故障注入，不是CPU神经）。
`results/p5_jepa_fresh_fidelity_main_20261027_v11/cuda_main_v1/report.json`真实CUDA重跑
为fresh_native_fidelity_recorded，attempted/completed均encoder1/proprio1/predict4，
源/模型不变、收尾错误为空，四项MSE与v10逐值相同。因此这类记录漏洞已修复，
但不解释本批reverse更接近factual目标；不以修复记录替代科学机制修复。

独立P2配对算术与代码合同审计也已完成：
`results/p2_composition_paired_outcomes_20261027_v2_report01/report.json`及
`results/p2_composition_paired_code_audit_20261027_v1_audit01/report.json`。
160个suffix KKT独立检查最大归一化残差约7.03e-14。root0/2是模型到真实编码
排序反转；root1/3是实际Q改善而pose损害，不能全部归结为预测误差，也不宣称
所有代码问题已排除。主线程factual/paired-outcomes/P4-mechanism联合60项通过。

四论文一手值v3已完成：
`results/p2_reported_baseline_table_20261027_v3/export/reported_only.csv`。
238条指标单元格（236数值、2原文缺失），不是238独立试验。保留论文版本/表页/
任务/预算及missing；GRASP结果是模型内而非真实物理成功率；Hi-LeWM保留完整sweep，
LeWM Table10来自v3/p26；PlanWAM驾驶指标独立呈现。主线程文献表与NumPy路径审计
联合37项通过，不与本地校准混排。NumPy ToImage轴交换是确证的第三方分支问题，
但当前P1调用并不经过该分支，不虚构生产修复或把它当P1根因。

P3新鲜联合H2/H3 Jacobian residual分配协议v4已由主线程完整审查、31项测试和
source verify通过，主机CUDA已实际完成：
`results/p3_horizon_residual_budget_20261027_v4_cuda_main01/`。
每根unique paired142 predict cap含共享nominal/ADforward与两臂双重重放；8Jacobian/
8SVD/1536数学VJP方向另列。实际cuda_completed、failure为空、模型不变，552次
predict与552次action encoder、8Jacobian/8SVD/1536数学VJP方向，所有八臂完整。
dynamic四根的H2/H3 blocks为7/3、6/4、6/4、6/4，对照均5/5；八个赢家均H2。
dynamic对uniform模型Q差为0/−.0172064774/−.1096081033/0；这只是模型内结果，
物理consumer另行审查执行。协议允许H2/H3，若实际选出不同horizon必须分层，
不据相同predict cap声称全FLOPs/历史输入采集同预算；P1/P3准入false。

### 六路并行继续与 P3 v4 真实执行完成

六个既有智能体分别推进：P5 三新seed原生事实面板、对应v12 CUDA consumer、
P5前瞻C1执行桥、GRASP官方算法/适配合同、历史checkpoint来源绑定、
P3 v4独立真实结果审计。主线程统一承担实际物理/CUDA与独立测试；
神经实验无CPU回退，不让子任务抢显存或修改旧封存产物。

P3 v4 physical consumer 新增 raw warmup→原始float32归一化token与saved
predictor_context.action_history逐元素一致检查；封存后主线程51项测试通过，
bound verify通过。主线程已实际执行：
`results/p3_horizon_residual_physical_20261027_v4_main01/report.json`，
status=guarded_calibration_recorded，80/80 selected-path动作返回、8/8 HOST CUDA
终态编码、0 failed encode、0 predict attempt，sources/model不变、failures为空。
四对均相同初始context且full_H2；dynamic减uniform的pose分别为
0、+71.8571017148、+11.6351722439、0。真实编码原Q差分别为
0、+0.2097468145、−0.0236775029、0。
root1预测Q改善但真实编码Q/pose均损害；root2真实编码Q改善而pose损害。
这不支持当前动态预算任务收益，亦不能以此判定所有实现已排除或理论被否定。
独立journal/配对算术审计继续；P1/P3 admission仍false。

P5 v10/v11独立saved-array审计已有284项检查通过，主线程完整读取审计源码、
另行13项测试通过。五个数组文件逐字节相同，进一步限制了修复失败记录漏洞
的归因范围：该修复没有改变这批预测值，不解释reverse控制更近factual目标。
它仍只是单seed事实目标分析，不是反事实真实标签或前瞻控制效益。

P5三新seed事实面板v3已由主线程实际执行完成，不再是queued：
`results/p5_jepa_factual_panel_20261027_v3/factual_main_v1/report.json`。
2026102702/03/04全部保留并完成，六个独立原生环境、90 control calls、906物理
子步、96帧，逐根16对原始RGB/state/proprio/full-public-state/transition精确一致。
来源不变、runtime/source error均null，输出约20MB；无神经/训练/候选选择。
主线程独立15项测试通过。单测曾因无outcome读取说明中的`v11/outcome`文字
误触字符串规则，已在封存前改为精确旧consumer名称与AST调用检查；非实验负结果。
多根CUDA consumer与独立物体/agent运动审计分别由不同智能体推进，采集完成
不等于P5或控制效益成立。

P5对应v12已在主机CUDA实际完成：
`results/p5_jepa_fresh_fidelity_main_20261027_v12/cuda_main_v1/report.json`，
fresh_native_panel_fidelity_recorded。三根全部complete、encoder3/proprio3/predict12
且attempted=completed，模型/来源不变、error=null、finalization errors为空；主线程
独立23项测试通过。15个完整原始feature/successor数组全部留存。
actual减normalized-zero三根分别−.0042538169/−.0041363509/−.0029931837，
actual减negative为−.0164603480/−.0197914866/−.0140636014；
actual减reverse为−.0019494483/+.0034781995/+.0001969742。
因此事实目标上有动作信号，但动作时序排序仍混合（对reverse仅1/3改善）；
不宣称稳定action fidelity、真实反事实收益或跨架构P5通过。独立数组复算另行进行。

### 前瞻原生 C1：独立目标与实际 CUDA 编译已完成

独立goal生成已实际执行：
`results/p5_native_goal_scene_20261027_v1_scene_main_v1/report.json`。
固定object[256,256,pi/4]、agent nuisance[64,448]，一次reset/一个.01s子步、
零控制动作。原生T顶点经live变换与官方goal body逐顶点精确相同；coverage未测，
不补造coverage=1。无事实frame15/预测值选图，goal_spec最后发布。

ready bridge与compiler已由主线程联合106项回归验证，再实际主机CUDA运行：
`results/p5_prospective_c1_cuda_compiler_20261027_v1_cuda_main_v1/compiler_receipt.json`。
status=selected_packet_replayed，encoder1/4frames、proprio1、predict13、JVP10列、
autograd.grad20，全部attempted=completed，原错/收尾错误为空、源模型不变。
nominal与ridge对独立goal的MSE=1.6608344928/1.6302777934，选ridge并精确重放。
normalized→native实际roundtrip最大误差1.4909239e-8，未clip、未改权重。
主线程独立saved-array审计（21 tests，外部只读review后封存）21项复算全通过：
`results/p5_prospective_ridge_array_audit_main_20261027_v1/cuda_v1_audit.json`。
Gram/rhs/ridge/两候选分数、saved normalized输入、raw动作映射与replay bytes均一致，
solve residual=1.8683544e-17。并非FD精度或物理收益证明。

在物理执行前，独立审查确认executor v1的失败路径漏洞：普通存储预算耗尽可
阻止close intent，导致env.close未调用并继续下一臂。尚未运行该物理consumer；
已要求独立v2使用终结预留并在未close时阻断后续dispatch，不改旧封存字节。
因此旧成功编译不被改写，物理暂缓不是科学负结果。新executor合同需重新绑定
ready bridge与compiler prepare后重跑，不能静默替换原executor hash。

P5三根独立saved-data审计已完成1168项物理检查、1205项含fidelity检查，主线程
完整读源码/结果并另19项测试通过。关键限定：三个新seed的object在全部16时刻
均精确静止，coverage恒定、contact计数为零，只有agent移动。其factual排序
不能证明物体操纵能力或C2。v12数字只作为动作输入链诊断，不升级任务结论。

### P3/P4度量假说与真实GRASP执行

P3 v4独立算术50项测试由主线程通过；另只读packet接线审计验证205个candidate
context与1059绑定文件，未建立解释root1/root2损害的接线错误。warmup中间15步
缺少逐步journal仍是证据边界，不宣称所有代码已排除。

新goal-only object-patch消融度量已由主线程实际HOST CUDA完成：
`results/p3p4_goal_patch_metric_20261027_v1_cuda_main01/report.json`。
80encoder+80既有projector、8metric SVD，零predict/Jac/VJP，源模型不变、failure=null。
四根task rank14/nuisance rank9，同候选池old/new Q的选择全部改变，新H为2/3/3/2。
这只是修复假说和score-only消融；不是完整一致新compiler、根因定论或物理收益。
独立数值审计与selected double replay/新物理consumer由不同智能体继续实现。

GRASP官方adapter机制子集和同H1的Action-Adam runner由主线程完整审查及60项
测试后已实际完成主机CUDA：
`results/p2_grasp_h1_cuda_runner_20261027_v1_cuda_main_v1/report.json`。
status=h1_calibration_recorded，failure=null，模型不变，四根八臂均完整。
21624 H1 transitions/7200 backward另24 real-H3诊断，总21648 predictor与
21648 action encoder；不混称H3等预算、全FLOPs匹配或完整论文复现。
四根GRASP减Action-Adam的H1终态MSE为−.20905542/+.13735163/−.11971569/
−.17533779，均值−.09168932，仅模型内3/4改善。八个raw计划仅4个在物理
动作支持域内；其他4个必须整臂拒绝，不能clip或从比较总数中消失。
新的guarded physical consumer另行审查，尚不能称真实任务优势。

### 新协议绑定重跑及前瞻三臂真实执行完成

executor v2经主线程与独立审查各29项非神经测试通过，使用final-budget预留
执行close意图/返回，任何未完成清理阻断后续分支。未改旧封存源码或产物。
新ready bridge绑定v2后重新实际主机CUDA编译：
`results/p5_prospective_c1_cuda_compiler_20261027_v1_cuda_main_v2/compiler_receipt.json`。
receipt FILE SHA=4c3529de98d55d736df5bd7140dbfd9948e05d4cadb0d4699f6d69a352866b0a；
独立saved-array审计21项全部通过，选出的ridge动作和数值与v1相同。
新三分支freeze先于任何环境构造完成；ready manifest FILE SHA=
1099ee3b001ddb9b88fad4d9c31e1fac39fef07b1120501ccee44a907e6c95b3。

主线程已实际完成：
`results/p5_native_c1_executor_20261027_v2/native_main_v1/report.json`。
status=native_c1_three_branch_execution_recorded，global_error=null。
compiled/nominal/factual_reference三臂均16帧完整，每臂11个前缀byte gate
全部通过、close完成且返回持久化，来源不变；总45control/453物理子步。
三臂终态物体位置和角度完全相同：XY误差156.8583147194、角度误差
1.7828973626rad、coverage=0。agent终态不同不代表物体收益。
因此该固定根上前瞻编译与真实执行链已贯通，但没有物体任务改善；独立轨迹/
journal审计与JVP有限差分检验继续，不将本结果升级为C2/P5通过或理论否定。

### 三臂独立审计、JVP实测及新度量真实对照

主线程接管独立saved-outcome auditor后完成：
`results/p5_native_c1_outcome_audit_20261027_v2_main_v1/report.json`。
542项检查全true，三臂suffix物体位置/角度逐帧精确静止、覆盖率增益0；
compiled/nominal/reference推杆suffix路径12.4682/1.1351/15.8447px。
前缀初始数帧有接触计数，但决策frame10至15三臂全部0；不能把“整条轨迹
有过接触”说成编译动作实际改变了接触能力。

JVP有限差分已由主线程实际HOST CUDA完成40predict、零encode/AD/physics：
`results/p5_prospective_c1_native_fd_20261027_v2_cuda_main_v1/fd_receipt.json`。
固定eps=.01/.005，全10坐标central FD与JVP整体相对误差
0.0003638283/0.0005198091、cosine0.9999999353/0.9999998654。
模型/来源不变、error=null、counts match；未按outcome设通过阈值。
主线程另写无compiler/model/env导入的saved-array auditor，9 tests通过，
实际91项全部true（40对输入/输出重建、两FD矩阵、journal顺序/有效扰动等）：
`.../independent_array_audit_main_v1.json`。这限制“大规模导数接线错”解释，
不证明目标度量、物理模型或全研究正确。

新goal-only metric独立审计422项全true，主线程另完整读源码并在113项联合
非神经测试中验证。八个selected计划随后实际HOST CUDA double replay完成：
`results/p3p4_goal_patch_replay_20261027_v1_cuda_main01/report.json`，
36predict+36action encoder、whole/segmented/saved full context与actions完全一致。

新的独立physical consumer已实际完成：
`results/p3p4_goal_patch_physical_20261027_v1_main01/report.json`。
八臂全部complete，90/90selected动作、8/8HOST CUDA终态encoder与projector，
零predict，failure为空、源模型不变。两对same-H2新减旧pose为
−57.8736070612与−3.0984134738。另两对oldH2/newH3保持mixed层，不进入
同H差或同预算均值。八臂official/object-only success仍全部false。
新Q真实编码代价四根均低于旧Q选出动作的新Q代价，但新度量仍只是已知根上的
score-only假说，不是完整一致新编译器、独立泛化或P1/P3/P4通过。

GRASP physical也已实际完成：
`results/p2_grasp_h1_guarded_physical_20261027_v1_main_v1/report.json`。
4eligible全部15步、60/60selected动作；4out-of-box整臂拒绝，全部八槽保留。
仅root1形成完整配对，GRASP减Action-Adam pose=−39.3659809249；其他三根
unmatched、差值null。不把唯一完整配对概括成四根总体优势，P0/P2仍false。

六路原有智能体已复用并重新派发独立后续任务。三路随后遭服务端
403 insufficient balance，主线程继续接管未完成的独立审计；不能把“已派发6路”
当成此刻6路均在运行。所有已完成实验保留，不因智能体错误重启实验。
磁盘剩余不足1GiB，未删除资产或下载大包；新physical运行前检查了至少
64MiB输出估算+200MiB终结余量。该估算不是源码内硬存储配额。

GRASP独立saved-outcome审计由主线程完整阅读后另23 tests与实际重验通过：
`results/p2_grasp_h1_saved_outcome_audit_20261027_v1_main_verify01/report.json`。
selected60/warmup60分账，root collection32reset、warmup4reset；完整历史帧与
原始pre-warmup state未随物理结果保存，来源hash不冒充独立重建。

goal-patch outcome旧v1实际测试触发32,000,000-byte evidence读取上限；此是工具
兼容问题，不是实验效果负结果。保留v1sealed源码/test不改；新v2仅对精确本地
protocol path+SHA允许64MiB，其他文件仍原上限。主线程全读v1/v2后，联合测试
30通过/1个旧v1实际测试已知失败；不得写成该联合测试全绿。新版实际纯saved
审计已完成：`results/p3p4_goal_patch_outcomes_20261027_v2_main_verify01/report.json`。
八臂/90动作/8encoder+projector及两Q/物理标签独立复算一致，sameH2两根
mean pose差−30.4860102675、新Q真实代价差−.0947741471，而旧Q真实代价差
+.0365973456。目标度量之间的结论不同仍需保留，不以新度量替代任务成功。

主线程另核对v4实现：Jacobian参与residual预算调度，但search的候选来自固定
pool，而非task-weighted谱程序优化。因此下一致Q编译器必须接真正动作响应谱/
ridge或有界优化，不把随机pool重评分当作原方案C1/C2完整生成规则。

## 后续 P 并行任务重新分配（2026-10-27）

按用户新增并发要求，关闭了三个已经因服务端余额错误终止的会话，释放槽位；
新建 Beauvoir、Huygens、Jason 三个智能体，并复用 Arendt、Meitner、Poincare。
以下是任务分配，不是六路持续运行或任务完成的证明：

- Arendt：真正 Q 一致的 fixed-H2 full10 conditional spectral/ridge 编译器；
  主线程负责审查与主机 CUDA，智能体不运行神经/物理实验。
- Meitner：P2 组合增量 saved-data 审计器、稳定根配对与预算/准入检查。
- Poincare：先独立审查主线程 P5 接触几何重建，再推进独立 P3 joint-H2/H3
  调度实现；不与 Arendt 的 fixed-H2 编译器混用算法或预算。
- Beauvoir：P4 多轮反馈预测—真实状态—物理收益归因审计与测试。
- Huygens：强基线证据清单校验，区分论文报告值与本地同协议结果。
- Jason：只读独立审查新 conditional 编译器的 Q、动作尺度、历史与预算。

写集合独立，旧封存不改；禁止以 mock 或实现测试替代科学验收。
主线程已运行新 conditional 编译器与接触几何审计的联合非神经测试：
29 passed in 8.55s。此时磁盘实测剩余 689 MiB，没有删除资产或下载数据。

主线程新接触几何 saved-data 审计产物：
`le-wm-main/results/p5_native_c1_outcome_audit_20261027_v2_main_v1/contact_geometry_main_v1.json`。
60 项检查全部为 true；决策时圆形推杆到实际 T 多边形表面的净间隙
8.4507323356（原生坐标单位），三个分支后缀最小净间隙分别为
7.0952825597、7.4046015747、3.1529822664。PD 位置/速度重建误差均为 0。
这是既有轨迹的算术重建，不是新增物理执行或 swept-path 保证；
后缀无接触不能否认整段前缀曾有接触，也不能据此宣称所有代码问题已排除。

## Q 一致完整条件编译：主机 CUDA 实际完成及独立复核

新运行 `results/p3_goal_patch_consistent_compiler_20261027_v1_cuda_main_v1/report.json`
（相对 `le-wm-main`），SHA256
`64fd87f48f73674d98b8eb2ddce7415507bf1e6d457df8e78ad91da1c15c4fe1`。
八臂全部完成；主机 RTX 5060 Ti CUDA，528 predict、528 action encoder、
88 fresh Jacobian、96 SVD，模型未变，failure=null、failures=[]。
这是每个前缀独立重算后续响应的完整 conditional full10/fixed-H2 编译，
不是旧 pool 的 score-only 重评分，也不是 joint-H2/H3 动态调度。

主线程另写两个纯保存数组审计器并实际运行：

- `p1_conditional_mechanism_arrays_main_20261027_v1`：八臂 selected 均等于
  自身 stage-greedy 选择，没有选出严格非贪心前缀；四对新旧Q动作均不同。
  不跨Q相减标量分数当物理效用。192 saved-array SVD，零新神经/物理。
- `p1_conditional_history_arrays_main_20261027_v1`：从封存源码 AST 常量
  独立重建 float32 normalized packets，全部80候选、24重复选优/重放context、
  8导出context共112项内容和元数据检查全true。零新神经/物理；不证明训练
  normalization provenance。两个审计器联合10项测试通过。

独立审查同时确证 v1 有异常收尾丢首错、partial 数组保存窗口和 verifier 未验
action-history内容三类实现缺口。前两类未在本次成功运行中触发；第三类由上述
实际112项补验收窄，但没有借此修改旧源码或声称所有错误排除。另派新版v2
修复。新动作尚需独立 fresh physical consumer，不能沿用旧终点。

后续并行交付：强基线 evidence inventory 已完成，P4 attribution auditor 已完成；
主线程重跑两者联合61 tests通过。P4只有一批20step/3round执行，不是两批，
step20真实latent缺失；actual−imagined pose变差而official state distance改善，
不可混指标宣称任务成功。两批原要求和P1/P4准入继续保持未满足。

### P1 原始前置不应被普通校准运行替代

独立完整诊断见 `p1_conditional_root_preconditions_20261027_v1.md`。全80候选中，
六臂的greedy同时严格最小化stage和own-reach；其余新Q根2/3共三条候选存在
stage更差/own-reach更好的tradeoff，但冻结beta1总分均不胜greedy。故不能说
全库绝无tradeoff，也不能只调beta就称机制成立。own-reach与非线性终态还会
排序不一致，zero/greedy/initial三种参照不互换。

四个根实际是boundary-contact、曾有旧zero-relative模型见证的已用校准根，
没有同一新Q/full10合同下的强直接优化平台与两阶段任务可达准入收据。
当前应标记前置未认证，而非理论反例或已证明根不合格。后续独立协议需要
预先冻结任务/动作/目标、有限扫描池与排除清单、平台与escape判据、非循环
筛选和完整真实对照，不能按已有物理结果补选容易成功的根或调整阈值。

P2新独立预算审计主线程已实际复验：39tests通过，产物
`le-wm-main/results/p2_composition_increment_audit_20261027_v1_main_verify01.json`。
四根完整H2配对，但standalone预测88vs168不等；真实pose仅1/4改善，
故继续保持科学准入关闭。准备扩展真实候选数的predict-matched frontier，
仍将AD/backward/FLOPs差异分账，不用padding声称等总计算。

## 新完整编译程序已真实执行；v2修复与P4补编码已实际完成

新consumer `p1_goal_consistent_conditional_physical_20261027_v1`已由主线程实际
执行，报告 `le-wm-main/results/p1_goal_consistent_conditional_physical_20261027_v1_main_v1/report.json`，
SHA `445ffe9c231bb4f132049389b9e689c01c3e6f8572eebfe2845c6445f143ebed`。
八臂全complete，80/80 selected、120/120 warmup，32collector+8warmup reset分账；
8/8 HOST CUDA encoder及existing-projector，零predict，source/model hash均通过。
四对同H2新Q−旧Q pose为−20.94913754/−59.16534902/−35.29529599/−67.74441500；
XY误差四根均改善，但angle三根变差，official state distance三根变差。
八臂official/object-only success仍全false，且selected均为各自stage-greedy。
这是完整Q一致生成的真实校准收益线索，不是独立泛化、非贪心机制或P1准入。
不能复用旧score-only物理终点来替代本批；本批独立saved-outcome复核正在推进。

独立新v2已修R1/R2/R3并由主线程真实HOST CUDA重跑：
`le-wm-main/results/p3_goal_patch_consistent_compiler_20261027_v2_cuda_main_v1/report.json`，
SHA `676f245a471aee56486bcd617a7a5c6b10a0d2ee4da340395acbf795a3b7f397`。
528predict/528action-encoder、88J、96SVD全部完成，durable journal complete，
无错误、模型未变。Main两个真实export校验和逐根逐臂完整compilation对象比较
均通过：80候选/88J/所有动作、代价、重放context与v1逐值精确相同。
因此这些故障留证修复不能被宣称修好了本批行为/机制效果；所有旧sealed字节保留。
物理证据仍标v1来源，不把动作相等偷换成v2已经真实执行。

P4遗漏的step20真实latent已从原封存两张真实RGB在主机CUDA补编码：
`le-wm-main/results/p4_terminal20_cuda_encoding_20261027_v1_cuda_main01/receipt.json`，
SHA `b181d413dec0ee4bb922e99c155542947c2daa43371f84e17ba24e8f17860cb3`。
两次encoder+两次existing-projector，零predict/AD/physics，模型未变/errors[]；
imagined/actual原Q代价0.9277847588/1.2569677374。这是同一旧物理批次两臂的新
派生编码，不是新增两批或新独立根；仍需新auditor消费填满最后一轮预测误差。
旧v1审计“当时末端latent未保存”的历史结论不被追改，P4科学准入不变。

主线程本轮最终联合回归（新旧conditional编译器、物理consumer、两个主线程数组
审计器、P2组合/基线清单、P4归因/末端编码、P5几何）实际252 tests通过，
19.50秒；全部为非神经/mock/保存数据验证，不替代上述独立真实CUDA和physics。
共享研究文档 `git diff --check` 无错误；最后磁盘实测约531 MiB，不删除资产。

## 六路并行恢复与新 P3 主机 CUDA 实际完成

已向六个现有智能体分别派发独占文件任务：P1 task-admission、P2 predict-matched
组合 frontier、P3 四臂 physical consumer、P4 terminal20 attribution、P1 独立 outcome
审计及 P5 native stage-admission。完成的 P1/P4 审计智能体继续接 P3 saved-array
独立审计与 P4 第二独立批次前置核验；派发不等于完成，GPU 由主线程统一执行。

主线程已实际运行新 joint-H2/H3 metric×scheduler 四臂：
`le-wm-main/results/p3_joint_horizon_goal_metric_scheduler_20261027_v1_cuda_main01/report.json`，
SHA `23eae72d18fa02e9997859311bef282fc4f5e3478790d846524a3fa6f6b70950`。
38 项 NumPy/mock 测试及 prepare verify 通过后，在主机 RTX5060Ti 执行；
1066 predict +1066 action-encoder 全完成，8 fresh J、1536 数学 VJP、32 execution SVD，
errors[]、model unchanged。主线程 selected_actions 逐值复算16 plans、own-Q响应、
候选/代价/选择、normalized history 与 whole/segmented replay 通过；新增产物约54 MiB。
Old-Q dynamic 在第2/3根模型代价更低、第1/4根相等，均同H2；new-Q 第2根
dynamic H2 对 uniform H3 且模型代价更差，其余三根相等。不同Q不相减、mixed-H
单独分层；这不是物理收益、独立根泛化、等总FLOPs或P3科学准入。

## P3 新动作已实际执行与两种负结果的分离

主线程已完成NEW consumer真实16分支：
`le-wm-main/results/p3_joint_horizon_goal_metric_physical_20261027_v1_main01/report.json`，
SHA `6c3659e199a8151fee2fa2c020a0cfe63db00b691b947cdbeb762c3c18d6004a`。
全部full/fresh，165selected+240warmup=405 public steps，collector32reset+
warmup16reset；16主机CUDA encoder+16existing-projector、0predict/AD，
model unchanged/source verified/failures[]。官方与object-only均0/16成功。
新增main独立审计 `p3_joint_physical_arrays_main_20261027_v1_main03.json`
逐七维状态重算pose/官方指标/reward、PNG/journal/actions/reset/预算及预测/真实两Q
代价全部通过，8项非神经测试；不是重新运行模型或物理，四根仍为已用校准根。

旧Q dynamic−uniform真实pose为0/+71.85710171/+11.63517224/0（均同H2）。
第2根预测Q差−0.01720623但真实Q差+0.20974681，存在模型排序反转；第3根
预测Q差−0.10960833、真实Q差−0.02367750，pose却变差，说明该例不能只用
排序反转解释，任务代理也存在失配。新Q三根schedule动作相同，另一根mixed-H，
不能把指标改善写成已证明C3。具体机理仍待进一步独立检查，非理论否定。

执行前主线程抓到13项fake接口失败：32×32图像不满足生产224×224 RGB合同。
只改fake、保留guard；最终44项consumer测试通过后才prepare/实际执行。
另外补齐原先遗漏的collector/warmup/selected实际调用分账，不把240selected上限
误称总物理预算。sealed planner及旧endpoint均未改写或借用。

P4 terminal20补编码的v2归因，主线程独立saved复算通过：imagined/actual末轮
raw latent误差L2分别10.12582373/1.06562063；但两臂真实Q末轮均变差，
不推出真实反馈有稳定任务收益。第二批前置审计只找到同一根完整source链，
其余旧根缺capability/response源包；未伪造新协议或把两臂算作两批。
P5新增机器gate实际45checks/101输入通过，但接触任务见证、同根C2合取、独立根/
架构复制和强基线等预算仍缺，P5不准入、不归为理论失败。

P2已从bridge推进到NEW可执行CUDA runner，主线程独立79tests与verify通过，
已启动主机CUDA `p2_composition_predict_frontier_cuda_20261027_v1_main01`；
启动不等于完成。计划真实80joint/40conditional、两臂168predict/root，
全四根unique1328predict/164J，AD/FLOPs不等预算，下游新物理consumer并行实现。

## P2 新168/168预测前沿也已实际完成

主线程真实HOST CUDA report：
`le-wm-main/results/p2_composition_predict_frontier_cuda_20261027_v1_main01/report.json`，
SHA `4023a0e3896a90407188d260a45dd334e0802ba17ecd59ef62ea931603eba173`。
四根complete/model unchanged/frozen eval；1328predict+1328action-encoder，
164J/31488数学VJP、168SVD/164solve、480候选评分全部完成。
主线程NEW all-slot exporter复算80joint/40conditional候选、fresh bank及suffix解、
动作/全context/见证/argmin/whole-segmented重放通过，导出8plans。
两臂各168predict/root，但AD/FLOPs仍不等；不能改称等总计算。

composition−joint80预测Q差依次−0.06865267/+0.11880530/−0.13266488/+0.13334418，
组合仅2/4预测更好。扩大joint库在第2/4根新选择index47/67，使原first40模型
优势不再全根保持。nested40仅诊断，不免费声称新40点独立重放；物理比较还需
NEW consumer，不能复用旧paired-v2终点。四根均已用，非独立泛化/P2准入。

主线程审查发现v1故障路径可能以journal/hook/finalization错误覆盖首错、
primitive数组在solve失败前未即时留存；已分派NEW v2，不改sealed v1。
该轮正常执行无报错，不能把异常健壮性修复追认成改善其已完成数值结果。
当前最新10文件联合非神经/mock/saved-data回归363 tests通过，26.52秒。

## P1 原始 kill-experiment 前置缺口已机器化，未把 procedural 根误当数据 episode

新纯数组 core `p1_conditional_task_admission_20261027_v1` 已由主线程实际
audit-current 与61项联合测试复验。当前四根来自procedural seed202753010，
dataset source_episode不适用；实际缺的是逐proposal/reset来源映射、连续同Q
full-H2 direct优化轨迹、独立calibration推导并冻结的plateau/escape/目标阈值、
合法完整两段目标见证及fresh confirmation收据。不能把当前80个合法预测候选
自动称为“直接优化停滞但两阶段可达”的任务，亦不称四根被证明不合格。

新core支持有界procedural seed/reset-span扫描、历史来源范围排除、calibration
推导阈值后freeze confirmation，不强制用户手填数字，也不允许根据已有物理
结果选根/改β。真正源生成与CUDA direct-screening仍需后续真实执行；目前只
完成接口、当前材料缺口审计和非神经验证，不把这些记作P1机制通过。

## P2 新预测前沿的八分支真实执行完成与六路继续推进

最新物理报告：
`le-wm-main/results/p2_composition_predict_frontier_physical_20261027_v1_main01/report.json`，
SHA `9144c3f25f2880af62d94dd88257918fa508b5ddbea51d98e9ccbc3d3b7532e2`。
8分支均完成完整H2、来源核验/冻结参数未变、failures为空；实际80selected+
120warmup=200 public env.step，40 resets。8次encoder与8次projector均主机CUDA，
物理阶段无predict/AD。主线程只读独立复算8终态pose/官方指标、80步reward与
4组配对差；official/object-only成功均0/8。

下表均为composition−joint80；负pose是改善，负Q是相应代理度量改善。

| 根前缀 | 预测source-Q差 | 实际source-Q差 | 物体pose差 |
| --- | ---: | ---: | ---: |
| 9173218f | -0.06865267 | +0.06322242 | -70.71712967 |
| 23b8de78 | +0.11880530 | +0.01111011 | +42.86336650 |
| 4afefa3b | -0.13266488 | +0.01943989 | +15.78390158 |
| cf1b9bb9 | +0.13334418 | -0.37239728 | +47.44742678 |

第1/3根模型与实际Q排序相反，但第1根物理pose反而改善；第4根实际Q改善而
pose变差。不能将这些现象压成单一理论失败，也尚不能宣称代码错误已经全部
排除。四根仍为已用校准根、AD/FLOPs未匹配，P1—P5准入全部false。

6个现有智能体已重派互不重叠任务：P1 bounded procedural source生成；P2独立
物理归因；P2 v2故障链与primitive持久化；P3全分支prefix编码；P4参数化多根
反馈runtime；SAGE及GRASP/Hi-LeWM/LeWM/PlanWAM原始reported数值核验。
主线程独占CUDA/真实物理调度与集成复核；mock测试不替代真实证据。
本次复查磁盘仅368MiB可用，继续保留200MiB安全余量，不删除历史资产，
新实际作业须逐项检查输出预算；不以CPU神经回退绕开资源限制。

## P3 全分支17前缀已完成真实CUDA，误差定位进入第二段

原16轨迹全部step5及唯一H3轨迹step10共17帧，未挑根/去重；主线程完成原冻结
encoder/projector各17次HOST CUDA，0predict/AD/新物理步，参数与source核验通过。
prepare SHA `ec969feae661973b845eb38677dd4d0a90d22cd990bda65d54569966c481504e`；
receipt SHA `ca6cdef177b2b8fe5e9739f2a07d2254b1183730ff9c340f52d96707f322a802`。
目录 `le-wm-main/results/p3_joint_prefix_cuda_encoding_20261027_v1_cuda_main01`。
独立consumer逐帧NPY/attempt/return/journal/完整context chronology验证完成；主线程
另从17个native NPY独立复算两Q代价通过。consumer报告SHA
`147c3cca6a66789fbd5052fc4f4017b4e0a6b63c59038cfac1fbc19c01da77de`，
目录 `le-wm-main/results/p3_joint_prefix_cuda_encoding_20261027_v1_numerical_main01`。
35项本地回归通过；mock不是实际CUDA证据，上述receipt才记录新增真实执行。

旧Q dynamic−uniform在step5：第2根预测+0.28867323/真实+0.18011910，二者均偏好
uniform；第3根预测−0.17900066/真实−0.10651307，二者均偏好dynamic。另两根
动作相同、差值为0。因此第2根终点的排序反转并未在step5边界出现；第3根
终点真实Q改善但物理pose损害也仍不能归为预测排序反转。只观察packet边界，
不能精确定位内部substep错误发生时间、接触因果或证明不存在代码错误。
三目录实际占用748K/332K/40K，没有复制权重或PNG。科学准入不变。

## 固定动作的真实前缀条件预测已实际完成：反馈减少误差但不消除排序失败

主线程新增并测试 `p3_prefix_teacher_forcing_20261027_v1`。保持原16分支动作不变，
在全部step5（及唯一H3的step10）使用真实已观测latent重建完整因果history，
action history保持原值；每个边界只预测下一个5步packet，不用未来帧或privileged
state替换模型输入。43项相关联合测试通过后封存，实际HOST CUDA完成17predict+
17action_encoder，无训练/AD/新encoder/新物理步，参数/来源未变，errors为空。
报告 `le-wm-main/results/p3_prefix_teacher_forcing_20261027_v1_cuda_main01/report.json`，
SHA `c5bd1f15a4a8aa7dd60050f8ce5a681aa20c1966e1715cdc3e1df013e9fba499`。
主线程另从17份raw output context独立复算native MSE及两Q预测代价通过。

17/17项native MSE下降（含重复动作，非17个独立收益）。第2根oldQ dynamic
MSE从1.32709831降到0.93923039，uniform从0.06928415降到0.01495747；
校正后预测dynamic−uniform Q仍为−0.27262176，实际却+0.20974681。因此这个
根的排序问题不能仅靠消除第一段想象误差解决；也不能据此直接否定方法理论。
第3根校正后预测Q差−0.05549441、实际−0.02367750，排序一致但原真实pose仍
损害，任务度量失配仍待处理。该实验没有重新选动作/闭环执行，不能把MSE改善
写成任务收益或P1/P3通过。

基线来源归档也已完成主线程20tests复验：GRASP、Hi-LeWM、LeWM、PlanWAM共
238cells（236数值、2缺失），原页表与可比边界见
`docs/d2p_baseline_reported_values_20261027_v1.md`。GRASP learned-model success
不能冒充真实physics成功，PlanWAM驾驶分数不能横比PushT SR；SAGE新身份解析
定位发布入口，但权重/许可/原表统计合同仍需核验，不自动算本地baseline完成。

## 尚未完成的原始要求

### 新增：P1 full-H2 校准已实际执行；后续阶段六线并行

`le-wm-main/results/p1_direct_cuda_calibration_main_20261027_v2_cuda_main01/report.json`
现已终态 `direct_calibration_completed`，errors为空。两根分别执行full-H2 GN与
gradient，主机RTX 5060 Ti完成120predict/120action_encoder、12Jacobian、2304数学
VJP方向、6SVD/6solve、12proposal/selection；encoder/projector各8次。精确预算
核验通过，模型前后tensor hash相同，无训练、无新增物理步、无CPU神经回退。
prepare SHA `24eaf8b02873b3d93ebeeadbead9ca56c2caa079cea8c30b308357ddd8a3d2e6`。
source为两根实际procedural calibration来源，历史全球排除覆盖仍unknown，history
为重复实测帧的causal padding，不是观测转移。B1=1/B2=3不是收敛平台证明；尚无
独立两阶段witness、预定阈值、独立确认或真实非贪心收益，P1准入仍false。

本轮已重派六个现有智能体并行：P1两阶段witness实现；P4真实prefix-specific CUDA
AD响应实现；P5按原定义生成artifact驱动闭环前置检查；新P1 saved CUDA独立审计；
reset/goal真实重放诊断producer；RGB目标方向contrast。各自NEW文件，不修改封存
版本，主线程独占真实CUDA/物理调度与结果集成。P4 root0 saved sourcepack已经
prepare，但不是已执行response或长程收益。磁盘最新仅316MiB，保留200MiB，
禁止自动删除/下载；后续每次实际执行均须检查有界输出预算。

主线程随后实际执行固定8-attempt reset/goal诊断，终态
`diagnostic_completed`、`all_reproduction_checks=true`、first_error=null；
16reset、8显式render、32setter、16内部goal render均完整。selected两根的
state/goal/RGB/root逐字复现，全部历史attempt也保留核验；报告位于
`le-wm-main/results/p1_source_reset_goal_audit_20261027_v1_main01/report.json`。
无public step/神经调用；这只排查本批source复现，不是实际任务收益或P1通过。

独立raw-array审计7tests随后通过，发现四组件B1→B2均还在下降：root0 GN
1.290452→1.155790、gradient1.353354→1.273638；root1 GN2.068284→1.968030、
gradient1.920786→1.830458。因此不能用本批三次更新声称direct plateau。
独立报告 `le-wm-main/innerdocs/p1_direct_saved_audit_20261027_v1.md`。

主线程新增固定完整catalog编码器，28项array/RGB测试通过后封存，实际HOST CUDA
执行104encoder+104projector，0predict/AD/physics/training，模型hash未变、errors=[]。
实际报告 `le-wm-main/results/p1_orientation_cuda_main_20261027_v1_cuda_main01/report.json`，
SHA `ad5649a5168123611b3bbec743b4886b7ef98854f19df82a7db5d8bcc5456234`。
独立saved-journal测试另逐项复算全部latent距离及104帧/resource binding通过。
104输入仅27独特pixel图，77重复对输出完全相等；非零角度与clean latent MSE
约1.29—2.33，但距离不随绝对角误差单调变化。原图去agent的MSE约0.224/0.948/
0.246/0.475；same-angle sampler最大MSE约0.03846。只说明模型不是对全部raster
旋转输出同一向量，不是物理角误差标定、可靠Q或任务收益。插值/质心/去agent混杂
继续保留，未生成新Q或改变选择阈值，P1仍false。真实observed-history来源producer
已另派实现，禁止将旧padding追认为观测转移。

### 两阶段见证与真实prefix条件重编译：新增实际CUDA结果

两根新witness终态`cuda_completed_not_admitted`，errors=[]、模型前后hash相同。
路径 `le-wm-main/results/p1_two_stage_witness_20261027_v1_cuda_main01/report.json`，
SHA `d85ad0999f316db9860bc5358f787488cadfd76c5bc8e2a0ff5a59439e5903c5`。
每根完整16prefix×4suffix，逐prefix真实own-context重新求192×10导数；共新增
252predict/action_encoder各、34J、6528数学VJP方向、34SVD/solve、128次评分。
优胜路径另付完整H2 nominal/whole/segmented独立精确重放，未用初始J伪充后继J。
root0/1预测Q MSE为1.19559215/1.72024295：root0不如direct GN的1.15578971，
root1低于direct GN1.96803027与gradient1.83045810。搜索总预算不等、没有真实
执行或direct plateau，本结果只是可执行模型见证，不证明非贪心收益或P1通过。

P4 NEW prefix-response在启动审查中发现normalizer metadata错误地`**`传入
mean/std-only函数；实际执行前修复并新增完整metadata回归，未记作理论负面。
修复冻结后，主线程实际HOST CUDA完成root0 imagined/actual双臂，终态
`two_prefix_specific_responses_saved`、errors=[]，模型未变。receipt位于
`le-wm-main/results/p4_feedback_root_cuda_response_20261027_v1_cuda_main01/receipt.json`，
SHA `a7fc0c38e17bf379d663ad639d1e2d9ae170b279f172eeda8c71d3b8f07936a8`。
各臂13predict/action_encoder、1J API、192grad API与192Python engine入口，共26
预测与384反向入口；每臂十候选全部实际非线性评估并独立精确重放。imagined
选candidate7 cost1.26076960，actual选candidate9 cost1.38428266。不同context下
分数不是实际控制优劣比较；当前只是known-root saved-prefix重编译，不是live
长程反馈收益。执行约6.71秒，peak allocated/reserved约141/145MB，cap未触发。

P5依赖清单已主线程实际重建核验，129份文件快照一致，`--require-admission`
预期返回3，科学依赖未满足；报告
`le-wm-main/results/p5_closure_readiness_20261027_v1.json`。不会把跨论文reported
数值、native strict load或单根C1诊断冒充独立架构机制/任务复制。

### 零点导数经真实CUDA有限差分检查；三帧来源改为实际观测

FD实际32次full-H2扰动/64predict与64action_encoder全部完成，无新增AD/编码/物理，
模型未变、errors=[]。报告
`le-wm-main/results/p1_direct_fd_cuda_20261027_v1_cuda_main01/report.json`，SHA
`d77f7aadf3fd3b6bd7da8c32ce499e5dc486a6989b0c178622aba4ff4478eaa1`。
两根×四固定方向×两个预定epsilon共16比较，relative L2约0.000221—0.000823，
cosine最低约0.999999664。只覆盖零点/这四方向，不等于证明全部非零点AD正确；
没有事后设科学pass阈值，但本次未呈现明显零点导数/两包链错配。

observed-history producer随后实际完成4reset、20public steps、6explicit renders、
8setter、4goal renders及208个实际hook计数内部physics steps，全部attempted=completed。
两根source均完整、errors为空，报告
`le-wm-main/results/p1_observed_history_source_20261027_v1_main01/sourcepack.json`，
SHA `0b89d77494acc6cf03bb0110e87231c1c240a253e844326acc668b7e4f93d546`。
主线程独立NPZ/checkpoint/journal复算通过：原定zero动作在本批确实保持静止，
真实0/5/10 RGB/state/actionhistory与旧padding数值逐字相同，root SHA等于parent。
因此填补了来源语义缺口，但不制造新独立根，也不支持把本批失败归因于padding
输入数值差异。未在看到此结果后改warmup动作以强求运动。

在采集器测试中修复partial archive错误：中途失败仅保存真正执行的raw动作，
不把剩余预定动作写成已观察历史。原生eval_state另发现多圈角差未先mod的风险，
保留native-stop并独立wrapped报告，不静默改外部环境；本批无native terminal拒绝。
后续P1真实四臂物理执行保留既有fresh RGB逐步guard，MAIN明确将显式render预算
从24提升为88后再封存，禁止为节省render弱化guard。磁盘最新仅256MiB，后续
大输出仍须在200MiB安全余量之上满足完整预算，不降级CPU或删除旧资产。

### 四臂真实执行与24帧主机CUDA诊断已完成

四臂物理结果已保存至
`le-wm-main/results/p1_witness_physical_20261027_v1_main01/report.json`，SHA
`48df27e54eca33159c44a834ce0be1a13723553dc9d64dde996b7b4ca9eaf747`。
两根各zero/GN/gradient/two-stage，全部完整10步，共16reset、80step、88render；
独立saved审计核对24 PNG、原动作、state、guard及日志，11项测试通过。
官方success为0/8，本批negative native角差与native/wrapped终止分歧均为0。
root0 two-stage相对zero物体位置误差增加6.22149，角误差减少0.04710rad；
root1 two-stage物体pose不变。不能将agent运动或小幅角改善冒充任务收益。

MAIN随后实际HOST CUDA完成所有24帧原生encoder/projector，各24次，无predict、
AD、新物理、训练或Q拟合；errors=[]，冻结模型hash未变。结果目录
`le-wm-main/results/p1_witness_rgb_cuda_main_20261027_v1_cuda_main01`。
原始192维latent已durable保存于`image.events.jsonl`，分析使用原goal和原Q。
终态predicted Q / actual RGB Q如下（描述性校准，不是新独立样本）：

| root | arm | predicted Q MSE | actual RGB Q MSE |
|---|---|---:|---:|
|0|zero|1.381571|1.361122|
|0|GN|1.155790|1.245310|
|0|gradient|1.273638|1.260441|
|0|two-stage|1.195592|1.258005|
|1|zero|2.191056|2.077293|
|1|GN|1.968030|2.089442|
|1|gradient|1.830458|1.844490|
|1|two-stage|1.720243|1.725766|

root1 two-stage真实RGB Q也更低但物体pose不变，支持继续检查度量/物理任务错配，
不能全部解释为模型预测Q排序反转；GN在该根相对zero的Q改善预测未兑现。
这不是最终因果归因、不是整个理论负结果，也未打开P1。

并行复用全部6个智能体：P2组合saved-contract、P3自适应预算saved-audit、
CEM/IID采样实现、采样实现独立审阅、P4多轮真实feedback实现、P5本地修改等价性。
各自新增文件、不得修改sealed来源；CUDA/physics真实执行仍由MAIN统一调度。
P5 architecture provenance另已实填63文件，contract SHA
`6c169f0685bb227913ae6deb821d74093e15eed29f063584e919bae892cde7b0`，
51项新增测试通过；许可用途、远端权重认证、normalizer来源、真实训练成本等
仍未齐备，不自认official或零训练成本。

### 同forward调用的CEM/IID采样：MAIN实际CUDA完成

`le-wm-main/results/p1_matched_sampling_cuda_20261027_v1_cuda_main01/report.json`
SHA `cfd13ca478603c90c56b81c70f0ed161ef56d6a8de87da4c0e317388bd138e4e`。
errors=[]、source重验通过、模型未变；两根两臂共240个完整H2候选及独立赢家
nominal/whole/segmented重放，总504predict/504action_encoder，0AD/image/physics。
每根每臂126forward与witness相同，但witness另有J/VJP/SVD/solve，不是总算力匹配。
预冻结PCG种子、6批×10候选、CEM elite3以及球域不随结果调整；这是明确的小预算
局部对照，不冒充官方默认CEM复现。40工程tests通过；MAIN新增独立saved测试4通过，
重算全部240候选Q、raw tanh、两包action/latent history、完整winner、elite更新及日志计数。

| root | CEM预测Q MSE | IID预测Q MSE | two-stage预测Q MSE |
|---|---:|---:|---:|
|0|1.17114157|1.18210362|1.19559215|
|1|1.90143550|1.86180073|1.72024295|

只有root1两阶段胜出该模型评分；root0不胜采样，不能宣称普遍优势。采样物理后果
尚待NEW真实执行producer，已独立派发；当前表不提供真实任务收益或独立确认。

长预算observed-history direct v1保持B1=4/B2=8、完整32J/6144VJP不变，但其
56MiB输出+200MiB安全余量高于当前约235MiB磁盘，故不启动该v1。
NEW v2正在实现无损流式压缩：保留全部原始数值/数组/事件，Z_SYNC_FLUSH+fsync，
不能减预算或薄化证据；需完成源审阅/规模验证/新封存后才可MAIN实际CUDA。
本轮未删资产、未绕过安全余量、未CPU神经回退。

### B4/B8直接优化与真实多轮反馈：新增实际终态

MAIN已完成observed direct v2全预算主机CUDA：280predict/action_encoder、32完整
H2 Jacobian/6144数学VJP、16SVD/solve、8encoder/projector；errors=[]、模型未变。
报告 `le-wm-main/results/p1_observed_direct_cuda_20261027_v2_cuda_main01/report.json`，
SHA `954af696d88eecebdcdc7345de702809517491967263f395a8f2819753d889d5`。
保持B1=4/B2=8与全部事件不变，仅无损gzip存储；压缩完整性/密集规模12tests在
内存临时目录通过。首次磁盘mock测试7pass/5fail来自224MiB启动门，未算成方法负面。
只迁移了本轮MAIN及智能体确认的临时mock目录到主机内存临时区，未删研究证据。

| root | method | iteration4 Q | iteration8 Q |
|---|---|---:|---:|
|0|GN|1.15578971|1.15578971|
|0|gradient|1.24192876|1.21040921|
|1|GN|1.96803027|1.96803027|
|1|gradient|1.79904298|1.71640537|

两根gradient仍改善；root1 gradient8已略优于two-stage的1.72024295。这直接否定
把旧B3预算下的比较描述为“直接梯度停滞而两阶段逃逸”的证据，而不是否定整个
理论。GN固定步长被拒绝不等于全部直接优化到stationary。新的B8物理consumer
已派发，不能用模型目标表格替代真实收益。独立saved-gzip审计正在进行。

P4 live run曾在216MiB启动门被拒绝，发生于输出创建/模型/physics之前；该进程
明确退出后，暂停并发mock写入、实际满足完整门限，再启动原封存prepare02。
实际终态`two_live_20step_3round_arms_completed`，errors=[]，source重验通过、
冻结模型hash相同。报告
`le-wm-main/results/p4_feedback_live_cuda_20261027_v1_cuda_main01/receipt.json`，SHA
`e3c91195540b7465e2fcbf8431d7f182c06bfdff9da6ab160489ae6fe23bbb47`。
两臂各20selected步，step5/10/15新ownRGB编码/新J/十候选/独立重放/封存后派发；
总80predict/action_encoder、6J API、1152grad API及1152Python engine入口，8批
encoder/projector覆盖20帧；34reset、30warmup+40selected public steps全部返回。
内部600s/外部650s watchdog，实际12.56秒，显存peak allocated/reserved约151/174MB。

| arm | terminal actual Q | object xy error | wrapped angle | pose cost |
|---|---:|---:|---:|---:|
|imagined|1.57129876|57.98232140|1.48160296|87.61438057|
|actual feedback|1.50693674|90.58260255|1.33133641|117.20933073|

官方与object-only均0/2；反馈改变真实动作但Q改善未转成综合pose收益，不宣称
P4有效。单known-root配对不提供独立确认；因果/预算/物理指标独立审计仍在进行。

采样赢家另已实际执行四臂：8reset、40step、44render、12PNG，errors=[]。
报告 `le-wm-main/results/p1_sampling_physical_20261027_v1_main01/report.json`，SHA
`c3eef328bee57ce5d42b25fcf5a9935af2d2282c9de3f8caf46f2b35b53a2ef6`。
四臂物体pose均与各自初始值相同，official均false；不是成功采样逃逸见证。
真实12RGB native编码producer已派发，尚不能填入实际Q值。

### B8实际物理与采样实际RGB编码；驻点归因进一步收窄

B8实际四臂已完成40public steps、8reset、44render、12PNG，errors=[]。
`le-wm-main/results/p1_b8_physical_20261027_v1_main01/report.json` SHA
`e94a6cbf55dd9b408d350e690d17bfdb9ca693b2ef360a12ab904024f272cd7a`。
root0/1 GD8的native state distance为133.68772035/236.12397258，物体pose仍
与各自zero相同，四臂official均false；GN0/1物体位置/角误差与旧GN相同。
不能以更好的native连续距离代替成功或能力改变；独立物理saved审计正在进行。

原B8 CUDA独立gzip审计已交付，6144原始J行/tanh链、全H2 history、预算、
zero及旧B3前缀exact一致；20tests通过，报告SHA
`30726f50914506db5496300f19876de09a51e0bd2345adbc0ca2dd59d8632749`。
P4实际live独立审计核对198文件/5291事件/六次封存选择到派发链，12tests通过；
报告SHA `6bdd30c1da733417147504050beed1b6a60ddd513abee45a083fd0955d741953`。
未独立重做AD/encoder；84 guard观察中82份有保存PNG可重计，其余只可审计收据。

MAIN新增GN驻点saved诊断7tests通过：以双packet独立球投影梯度映射而非原始
梯度判别约束驻点。两根GN最后mapping为0.10750435/0.26056781，原被拒绝提案
g·delta为-0.06396976/-0.26968428。条件于saved AD，固定步长重复拒绝不是驻点
证书；负内积也不保证有限步长实际下降。NEW固定四缩步CUDA诊断已实际完成，见下。
报告`le-wm-main/results/p1_direct_stagnation_saved_main_20261027_v1_audit02.json`。

采样12帧MAIN HOST CUDA已完成，不去重12encoder/projector，0predict/AD/physics，
errors=[]、模型未变。报告SHA
`ae0fe77b628c35a129bd6035ef59771876fe4791bd12774f1e9123e30a6ef34b`，目录
`le-wm-main/results/p1_sampling_rgb_cuda_20261027_v1_cuda_main01`。
原sameQ终态实际值：root0 CEM/IID为1.22166726/1.22904271；root1为
1.98787531/1.94012334。四臂物体pose不变，不能把其Q下降写成成功逃逸。
独立原始latent保存复算正在进行；B8自身12帧producer也在实现，未回填实际Q。

本体审阅另定位：native success同时约束agent/object位置及角，native reward
还包含agent速度，而当前终点评分仅比较单帧latent。去掉goal agent不自动更符合
官方任务，object-only是另一个诊断，不能静默改成功合同。速度差可改变reward
而不改变同一几何单帧图像；这是明确的接口代理缺口，但不能解释所有失败，也
不改变success无速度门这一事实。详见
`le-wm-main/innerdocs/p1_c1_source_goal_ontology_review_20261027_v1.md`。

仍需可靠任务度量与独立 P1 非贪心机制/任务收益、完整资源匹配的强基线、
真实 P2 组合增量、P3 物理质量—成本曲线、P4 长程多轮机制效益及 P5 独立架构，
并覆盖原方案三组核心实验。当前并行实现不能替代任何科学槽位。

### GN 固定整步导致的假停滞：主机 CUDA 直接证据

`le-wm-main/results/p1_gn_step_diagnostic_cuda_20261027_v1_cuda_main01/report.json`
已终态 `cuda_fixed_segment_diagnostic_completed`，errors=[]；32 次 predict 与
32 次 action_encoder，未新增 AD、图像编码、物理执行或训练。两根全部评估
预先固定的四个缩步，不提前退出；独立重放所选动作。这是同一校准根诊断，
不是新增独立样本，也不是等总预算胜出实验。

| root | 原 GN incumbent | scale .5 | scale .25 | scale .125 | scale .0625 |
|---|---:|---:|---:|---:|---:|
|0|1.15578971087|1.17340947152|1.14426763549|1.14754898174|1.15160665679|
|1|1.96803027191|1.99330290904|1.93341396813|1.94091412597|1.95260714176|

两根均选 .25，改善 0.01152207538 / 0.03461630378。原反复拒绝固定整步
不构成驻点或直接搜索无能的证据；目前有实测支持的原因是优化协议缺少缩步，
而不是据此宣布理论失败。新版本 line-search GN、所选动作物理消费者和独立
saved 审计已分别派发；尚未完成的执行与审计不计入本节已完成证据。
latent 代价下降仍不等价于真实成功或 P1 所要求的非贪心能力改变。

主线程再次运行驻点诊断与缩步诊断共24项tests通过。GN实际报告SHA为
`2636b3a89cbe96b97d64dce853f506a837e4191e05b90dd5b754146fde658848`。
采样12RGB的新增独立审计也已由MAIN全读并实际内存重跑，16项tests通过，
12 PNG与原Q复算最大差4.44e-16；报告
`le-wm-main/results/p1_sampling_saved12_audit_20261027_v1_audit01.json` SHA
`a0c0b276eb6e25fdaf1f6f1c27829abba7c9e809cd37f2f6af181d9c13f25194`。
这验证保存数据及算术，不声称独立模型重编码。两根采样Q与native连续reward
排序反转；root1仍存在去除速度项后的位置排序失配，不能将全部问题只归为速度。

上次磁盘读数206,700,544 bytes曾阻止新增实验；目前外部清理和小范围缓存清理后
已恢复约59GiB可用，以下新增实验在原输出预算/200MiB余量门内继续执行。
清理明细见 `docs/STORAGE_CLEANUP_20261027.md`；不得将全盘大幅增长都归因于本轮缓存删除。

### 完整线搜索 GN、实际执行与同前缀能力干预

MAIN已完整读取新增物理消费者、测试与独立saved审计；首次147项联合复验中
136通过、11在测试夹具setup失败：压缩task被普通JSON reader读取。
修复仅将fixture的 `m.read(task_path)` 改为 `m.search.read(task_path)`，
随后147项全部通过。不是模型负结果。prepare01仅创建未执行，修复后使用
prepare02重新封存；旧prepare01保留但不再是当前源码闭包的有效封存。

完整线搜索模型实际报告：
`le-wm-main/results/p1_line_search_gn_cuda_20261027_v1_cuda_main01/report.json`，
SHA `5f9825f98ed40f637c9d48e49c7c53428d1380e4234e3a1e5503a76f74c4bf1e`。
固定四次fresh J、每次完整五尺度，两根尺度均 `[1,1,.25,.25]`；最终原Q代价
1.1405115382133657 / 1.9203170978443598。总156predict/action_encoder（已含
ADforward）、8J、1536VJP、8SVD/solve；不是与旧B8等总预算比较。
MAIN独立复算再次得到18,004-byte报告：
`le-wm-main/results/p1_line_search_gn_saved_audit_20261027_v1_saved02.json`，SHA
`ffaf2cf7915848f43f47eeee4925147262908fdc1e62901066a72d57dcb05deb`。
保存行/单次tanh链/完整chronology自洽不等于AD正确性的独立证明，fresh-point
方向有限差分已另派实现，尚不计完成。

对应NEW物理验证已主机完成：4reset、20public step、22render、6PNG，errors=[]。
prepare02 SHA `f36f915ccfc2687da056810d1241da433190eff430bb54fa7b7b35a6af3818d5`；
报告 `le-wm-main/results/p1_line_search_gn_physical_20261027_v1_main01/report.json`，
SHA `3a41fe83f375e5bb3f390e9f3a0e8e17dd70c2d49a7ebe3a99079f5774de85c1`。
两根native distance为301.408855429 / 321.900842012，success均false，
物体位姿从初始到终态均不变。未把latent下降升级为真实任务收益。
6帧实际RGB主机CUDA消费者正在实现，未回填原Q结果。

另完成同前缀真实能力干预：原2根 × zero/bank_greedy/two_stage ×41固定后缀，
246完整分支、492reset、2460public step、2706render、738PNG，全部native success=false。
报告 `le-wm-main/results/p1_same_prefix_physical_main_20261027_v1_main01/report.json`，
SHA `d975ef03f1b7fa540b449a403fec2fa83cf5eb9a89d4dcedbaf6555efe32c4fe`。
MAIN重跑独立saved审计，逐34,196事件、810文件绑定与object-only响应一致，
报告 `le-wm-main/results/p1_same_prefix_saved_audit_20261027_v1_main02.json`，SHA
`dcc18b35b5a4327d264a2c14a3d78edd88b3b021e6356cfa36177c3e92670f66`。
同role的step5已记录body状态/RGB强等，不覆盖solver cache等全部隐藏状态。

两根所选greedy/two-stage前缀后40个非零后缀的object-only位姿响应全零；zero
前缀有响应。root1 greedy与two-stage是同一bank prefix，不能当两个独立方法。
root0 zero最大角误差改善仅0.0787521rad，root1 zero角误差最多恶化0.1414269rad。
这些是有限轴向探针的诊断，尚未证明全局不可控；正在排查接触丢失、probe中心、
候选构造与原目标代理。更没有满足P1所需的合法两阶段成功见证或非贪心收益。

B8实际12帧原始latent也已由MAIN独立复算（60项测试纳入上述147），144文件绑定。
四臂actual终态Q为1.245309841991 /1.214026768507 /2.089441582432 /1.743841302070
（root0 GN/GD、root1 GN/GD顺序）；root1 GN实际比初始恶化0.012148140104，保留。
报告SHA `3f1b346979cd8e53e357aba0576a91d11e4268cc922dc84d63f5785622742d8e`。

P1仍未通过；P2/P3不能以缓存枚举、计账或测试代替组合增量与动态计算收益。
原三组实验、强基线等预算、独立根/跨任务与跨模型确认仍是完整目标。

### 持续后缀实际反例：稀疏零响应不等于不可控

MAIN新实现持续后缀诊断，经两轮独立审读修复planned分母/终态写盘/partial退出码，
15tests通过；与新6RGB producer合计42tests通过。固定同2根×3角色×17后缀
（zero+8方向×logit L2半径.5/1.5，五步同向）；全动作仍在原packet球内。
不是完整10维球覆盖，也不是独立根确认；较大半径同时改时间支持与幅度，不能称纯单因素消融。

主机实际report：`le-wm-main/results/p1_dense_suffix_physical_main_20261027_v1_main01/report.json`，
SHA `76fb65359a9ebf9e42a25a89aaab48ef2ae6918640e7b5e18ddcfcda5675d9a6`。
prepare01 SHA `0ad44a85d15a14ede21f1dcc79fdf041f549aa27e5a31e7885c9f1057f5e7e4f`。
204reset/1010step/1112render；102计划全部保留，96fullH2、6观察守卫失败，CLI退出2，
status=`partial_or_failed`、全局errors=[]但六行保存first_error，不能只读errors空就称成功。
六行均root0 greedy（probe4,5,6,12,13,14），canonical agent pixels absent；
不放宽守卫、不补跑、不把其缺失full endpoint混入完整分支统计。

完整分支的相对同role zero-suffix物体响应：root0 greedy最大位移33.62840117、
最大wrapped角位移0.18739343；root0 two-stage为31.47230762、0.30292560。
root1 greedy/two-stage仍精确零。root0两个selected-prefix的非零响应足以反驳
将旧小单轴探针零响应外推为“前缀后物体不可控”，但不证明新控制器、任务完成、
非贪心收益或P1通过。所有本轮分支未出现native success。
独立partial-aware审计已完成；MAIN实际重跑 saved auditor，main02 报告 SHA
`bb682de7d641ee93d6194f270e079ea79568264f1536149e7de736dca5d95cb0`。
14,056事件独立复算、实际302PNG（306仅上限），102计划/96完整/6失败。
六个失败末态均agent中心x<0；仅probe4/6的失败PNG保留且RoyalBlue为0，
其余四帧不能从缺失像素判定视觉原因。没有新增物理、替换失败或P1准入。

线搜索实际6RGB也已MAIN主机CUDA完成，6encoder/6projector、0predict/AD/physics，
errors=[]，model hash前后相同、source verified。报告
`le-wm-main/results/p1_line_search_gn_rgb_cuda_20261027_v1_cuda_main01/report.json`，SHA
`e1ec726ec0d54ef7347030abdaa10a2bdf4f5d4767e327970a2a29c819723dcd`；prepare SHA
`cf384f59734fd7ae247dd3d0bf82414d26b5d000fb5f7851eb36cca07c625e0f`。
真实端点原Q为1.268256082304 /2.028360163971；对应预测1.140511538213 /
1.920317097844。两根actual均比各自起点降低，但真实物体位姿不变、native仍失败。
不能把优化器缩步有效写成物理机制已经成立。独立raw6审计已由MAIN重跑，
saved02 SHA `cc30f13ab2f2cf71ba732950d6aa67ea5531b8c2cb7a7eb977372f81a64a0d6a`；
六NPY/六PNG/36编码事件及原目标算术一致。另物理独立审计282事件、4reset/20step/22render吻合。

同前缀接触诊断MAIN另22tests通过，实际saved复算报告
`le-wm-main/results/p1_same_prefix_contact_diagnosis_20261027_v1_main02.json`，SHA
`d68cda672705b7b3ae67980d2326d2f9975ee00adfb8f767c4d064131b0ce604`。
原稀疏40probe在agent坐标/符号/时序上均有正确响应，未发现全零dispatch或投影消除；
selected后缀820public步native contact receipt为0。native n_contacts并非精确/分pair接触计数。
“远goal-agent诱导离开接触”仍是解释假说，不把物理误差分解冒称latent Q因果归因。

Experiment1 learned-prior仍缺配对资产且当前H10会触发final-goal fallback。
官方最短非平凡配置为H50/[25,25]，低层与generator坐标/许可/真实调用须认证；
不会拿H10无调用结果或不同checkpoint论文报告值冒充等协议主表。
当前继续并行处理：fresh-J CUDA有限差分、合法构造性可达source、learned-prior实际inventory。

### Fresh-J 主机 CUDA 有限差分已经实际完成

新增固定3方向（首包e0/末包e0/事前固定20维符号单位向量）×3eps
(.001,.003,.01)，覆盖两根各4个更新前fresh incumbent。8个nominal全H2 context
与原ADforward逐值exact；152次完整H2、304predict/304action_encoder，72组central FD，
没有新AD/encoder/physics/training。28项mock/saved测试由MAIN实际复验通过。

报告 `le-wm-main/results/p1_line_search_gn_fd_cuda_20261027_v1_cuda_main01/report.json`，
SHA `1dd13d4f8ce4ac2bcdee7e8e360ef283962bae11314efb19f017cbeb92d44163`，
status=`cuda_fd_completed`、errors=[]、forbidden_AD=[]、model before/after相同、source verified。
prepare SHA `29b70d36fb5a4f49f22474a9fc5932f566c2eaab7357b4de70d2411389e974b0`。

按float32实际secant方向修正后的相对L2误差范围：

| eps | minimum | maximum | minimum cosine |
|---|---:|---:|---:|
| .001 | .000249669 | .002693677 | .999996620 |
| .003 | .000217180 | .001134559 | .999999363 |
| .01 | .000238433 | .000798154 | .999999685 |

这是实测导数方向吻合证据，进一步限制本8点/3方向上的tanh链、标准化和跨包
detach错误解释；不认证完整192×20矩阵、共享oracle的物理语义、其他模型路径或理论。
边界附近FD有部分球外网络延拓探针，不是可行控制提案，也不当作驻点证书。
没有事后导数passing阈值。MAIN实际执行独立saved audit成功，八nominal/152evaluation/
72comparison/304完整context/1824事件全部重建；与前述误差范围一致。
MAIN新增四组独立saved auditor联合测试135 passed（FD31、physical36、RGB41、dense27）。

Hi-LeWM资产目录阻断部分解除：官方Zenodo206 Range两次尾部读取共131,114 bytes，
完整414成员中央目录一致，定位标准PushT object压缩113,104,429/展开122,424,517 bytes。
没有下载整包/权重或执行pickle；不把object/weights两个容器擅称高低层配对。
报告 `le-wm-main/results/p1_learned_subgoal_remote_inventory_20261027_v1/report.json`，SHA
`7feaf480934462619149c72e293de7df09cf55835f1001a8f7d88f9194b26c66`。
现继续小范围读取官方包内registry/config/loader文本，配对认证与真实比较仍未完成。

### 已完成：构造性可达目标的主机真实重放（不是P1通过）

两根固定探索候选，root seeds 202798101/202798102，reset seeds 202798201/202798202。
不补根、不重试；每根一次构造和两次独立reset/placement/zero-warmup重建。
目标为固定合法10步见证的真实state/RGB端点；with_target=False是新渲染合同，
不能把旧任务latent/结果直接挪过来。部署仅获真实3RGB历史、10zero raw历史及目标RGB。
见证来自离线几何构造，不是冻结模型规划器的输出，不给待测模型候选答案。

prepare01 SHA `98ffb806018a270f5ef27ffa76be929393a74355c213bc274bf158c72d80b086`；
实际 sourcepack：`le-wm-main/results/p1_constructive_reachable_source_20261027_v1_main01/sourcepack.json`，
SHA `9ad3a414fd2148258524d4a7e9d4f070f59532ec2e5943cfa61f9145b3a41bd9`。
status=fixed_slots_processed/sourcepack_complete=true/errors=[]，2/2合法可达；
每根两次replay均root非terminal、首次success在第10步，端点native distance=0。
3242条journal、126PNG；12reset/120public/1224实际physics/126explicit render/
12goal render/24set_state/4goal_bind，均attempted=completed。

| slot | 物体位移 pixels | wrapped转角 rad | 两次native replay |
|---|---:|---:|---|
| 0 | 94.6680295045 | .6401514987 | success / success |
| 1 | 149.1811047521 | .8106228472 | success / success |

两项独立只读审查另确认三episode真实帧/未来state一致和blind字段白名单；
完整独立saved auditor已由MAIN实际执行，main01 SHA
`3a097772498b4617b43966011ac0e464fbbe67bac555f0249bc8b173d7098802`；
3242事件/126PNG/6commit/2certificate/2blind全链吻合，errors=[]。
已发现producer失败路径缺陷：capture先guard后保存，拒绝时可能丢失失败帧；
本次未触发，不否定2/2成功，但不能宣称普遍完整失败留存。封存源码不原地改写。
goal_bind前后不变有运行时断言和零嵌套physics事件，未双份保存独立前后快照。
复合T多边形边采样也不是严格外轮廓接触净空证明。

本批只补合法可达前提：direct困难、暂时退步、控制能力改变、强基线收益、
独立新根确认仍未证明。P1/P2/P3/P4/P5科学准入状态不由本结果自动打开。
下一实际链为blind-only CUDA重编码→冻结模型GN/D2P程序生成→预选动作真实执行。

blind-only CUDA重编码已MAIN主机实际完成；报告
`le-wm-main/results/p1_constructive_blind_encode_cuda_20261027_v1_cuda_main01/report.json`，
SHA `684456d6d3c45b99f20db354d2a0fb7b810bb61a8ebc8185816aa3d87ff40e91`。
prepare01 SHA `7470bca322e14ca7f7e94a29f9d05ef46ed0108824158a098a2760ecec58d491`。
8encoder/8projector，0predict/AD/physics/training，errors=[]，source verified，
原model tensor hash前后一致。新task分别真实编码全部observed0/5/10及goal，
raw zero历史按原float32 stats变为两包between-state token，不用padding或重复除100。
同像素保留实际独立编码调用；新task IDs不是旧根的占位替换。

部署仅接受MAIN白名单02（SHA
`18a6354b72a0a0429155e06db5df73d8e1f9d589ade438d26e1626a124bb4fd7`），
不提供sourcepack/certificate/state/witness路径或native root hash。
白名单01为旧schema、未用于任何编码，保留其未执行状态，不原地覆盖。
两task export SHA分别
`714f471761278fa654542d4bb0878ce18e3c304df6158005af46d2129533265a`、
`70865d56f9b66f0072b2ad7de01cc09d9a9275047925e188104a13d5182ce1f2`。
源码/测试由MAIN联合source saved audit复验55 tests通过；编码独立saved auditor另派发。
真实GN/D2P计划已于后续主机CUDA完成（见下），物理评估尚待完成；不把可达见证成功归给模型。

Hi-LeWM官方文本取件现已完成：33成员CRC/size/path全部核验，64,090 bytes下载、
226,727 bytes展开；MAIN离线8tests复验通过。完整合同
`le-wm-main/innerdocs/p1_hilewm_text_contract_20261027_v1.md`。
main object源码持有共同高低层；先选择性获取约113MB对象，再静态容器/class审查，
不执行未知pickle。Hi-LeWM正式D25/D50/D75预算50/100/150，不能套SAGE的H50门槛。
尚无权重tensor配对或learned-prior真实比较；许可与normalizer/历史合同需继续核验。

### 新任务GN/D2P主机CUDA计划已完成，物理效果尚待验证

MAIN实际复验planner＋physical consumer的73项测试全部通过；编码saved auditor
另外33项通过，MAIN实际saved02重建2task/8NPY/50events/8encoder+8projector一致。
这些是实现及存档一致性证据，不替代物理效能。

新planner prepare01 SHA `30b58155a59f73fdc8cc132ff069e645ed6c2bbff9434f474b1f7b9c412f7eb4`。
主机RTX5060Ti CUDA run exit0，实际报告
`le-wm-main/results/p1_constructive_blind_planners_cuda_20261027_v1_cuda_main01/report.json`，
SHA `4dd5823e9a273b820d6b7b3f6a36a4a5a19b193383eaa9cbf94442082e79d98d`。
四臂全部model_plan_sealed，errors=[]、source verified、model tensor前后相同。
GN总156predict/156action_encoder；D2P132/132；合计288/288，无新encoder、物理或训练。
MAIN实际consume以外部report SHA复核完整输出后，四套模型预选动作可交独立物理执行器。

| root slot | GN模型终态Q cost | D2P模型终态Q cost |
| --- | ---: | ---: |
| 0 | .667755629543 | .682554888707 |
| 1 | 1.291969235366 | .176434958825 |

两臂使用同任务/原checkpoint/目标Q/动作域，但预算、选择目标不同；
上表仅模型预测，不是物理收益或公平优胜结论。P1仍未通过。
接下来封存显式source_slot/blind SHA/own commands交接，再执行四个物理分支；
controller不接触目标state、证书或见证动作。

Hi-LeWM object main01下载已终止失败（exit1），不是仍在运行：首个4MiB payload
Range头部206与长度正确，实际正文少8,716 bytes，报Truncated range。
保留4,624,388-byte partial及完整hashchain；未发布正式object，不能称权重损坏。
另行实现256KiB小Range、整次墙钟与限定重试的新v2协议，先探测、再决定取件；
不原地覆盖失败证据、不执行未知pickle，也不阻塞主线CUDA。

### 四套盲模型计划的真实执行已完成：有物体响应，尚无成功

新handoff补齐consume与physical之间缺失的source_slot/blind SHA/own commands映射；
其早期compact JSON摘要与physical默认spaces不一致的问题在封存前修正，
23项测试含直接physical.selection回归通过，未把验签失败当方法负结果。
handoff01 SHA `dbcbeacc18c4ae2fbaf1fed1f9494e7bac04184c2d7b7c2f334f311906d91c60`；
selection01 SHA `3f92ee62f72bb9950ca84b8b9a6bac6341e6a390cd5b5d39f9b75e443d2541fa`；
physical prepare01 SHA `b717feb4bba220f40f93bb317d1e8894a19b672f54a5942c03f4308253245eb4`。

MAIN实际主机执行exit0，report
`le-wm-main/results/p1_constructive_plan_physical_20261027_v1_main01/report.json`，
SHA `d205850e95f639b649fba9b5c6ee8059f2507431753a538b1ce30eccbd234300`。
四分支完整10步、initial_exact=true、无错误/guard拒绝，原生成功0/4；
8reset/80public/816actual physics/88PNG/8goal render/16setter/4goal bind全部完成。

| slot | 方法 | 物体位置误差初始→终态(px) | wrapped角误差初始→终态(rad) |
| --- | --- | ---: | ---: |
| 0 | GN | 94.668→47.977 | .640151→.257051 |
| 0 | D2P | 94.668→64.251 | .640151→.476316 |
| 1 | GN | 149.181→130.929 | .810623→.607269 |
| 1 | D2P | 149.181→45.399 | .810623→.494417 |

以上只是object诊断，不替换原生success（原生位置判据含agent与object）。
原生reward距离还含速度分量，不能把它与object-only误差混用。
MAIN读取全部D2P候选确认：两根winner分别2/7，均同时为同库immediate-greedy
和model terminal最优，故本批**没有建立非贪心选择**，不能从终距改善倒推重配置机制。
完整J/候选/动作/ledger独立saved audit由MAIN复验38tests及实际saved02，
9664events一致；不等预算与P1=false边界不变。

下一具体修正检验：实际step5/10 RGB的HOST CUDA编码与预测直接对照；
然后仅从每臂自己的实际prefix5更新3state/2action上下文，重编译suffix，
保留原sealed suffix和zero incumbent，做新的已知任务反馈pilot，不读取terminal10挑动作。
这不是严格P1确认，也不自动打开后续科学阶段。

物理consumer对正常returned transition后的guard失败先保存现场；若env.step内部
部分推进后抛异常，则不保证故障后fresh frame，last_observation可能仅为此前观察。
本批没有触发该分支，但不可概括成普遍完整失现场保证。

Hi-LeWM v2小Range探测也已实际exit1：第一256KiB payload完整，第二段
262144期望/229608实收，缺32536 bytes；5attempt/4completed，0retry，
received557435 bytes，19.45秒，report SHA
`669bf41043dfae01a9bca183fe230d61baad29234c5cc73038c80409844e0b90`。
失败证据保留；不满足full-download前置，不启动完整取件，不归因权重或理论。

### 真实 RGB、反馈后缀和尺度一致 GN：均已有实际结果

本节更新上述“下一具体修正”状态，不覆盖原始封存。所有新神经调用为主机
RTX5060Ti CUDA，无CPU神经回退、训练、Q拟合或阈值修改。

原四分支step5/10 RGB已实际编码，共8encoder/8projector、0predict/AD/physics；
报告 `le-wm-main/results/p1_constructive_selected_rgb_cuda_20261027_v1_cuda_main01/report.json`，
SHA `c77a5d208acb813ad968afe24ef0c547f08a83ff9968a984b29502baf0dda770`。
模型前后tensor SHA一致、errors=[]。MAIN另复验独立saved诊断30tests，
实际8latent/88PNG/2314物理事件核对完成，无不一致。

| slot/arm | 实际Q5 | 实际Q10 | 预测对实际MSE5 | 预测对实际MSE10 |
| --- | ---: | ---: | ---: | ---: |
| 0/GN | 1.082647 | .825542 | .106013 | .147491 |
| 0/D2P | .912043 | .856466 | .069927 | .118847 |
| 1/GN | 1.412312 | 1.537921 | .057846 | .191100 |
| 1/D2P | .489668 | .385148 | .339586 | .258235 |

两根、两个边界的GN/D2P同Q排序均未反转。root1 D2P真实Q大降仍未达到
官方位置/角度success，故不能把本批失败统称为排序反转或CUDA错误。
单臂相对root的预测/实际改善方向仍可能不同，不能与跨臂排序混为一谈。

**反馈后缀干预已执行。** 新编译器只加载actual prefix5 indices0/2/4/6；
旧history后两帧+本臂actual prefix、旧末token+本臂raw5规范化token；
不加载terminal10 latent作为选优输入。固定zero/original/fresh-ridge三候选，
总20predict/20action_encoder、4J/4SVD，包含ADforward及winner独立replay。
18项MAIN回归和独立只读复审通过。prepare SHA
`5b9f970a72e3cc3ef45b31d5d35416af6cdc779189f55fd3fd8ad5670a9a3e88`；
实际CUDA报告 `le-wm-main/results/p1_constructive_observed_suffix_cuda_20261027_v1_cuda_main01/report.json`，
SHA `ac8a65900b94639f94b3797e90f80e73fa0500e13a055629750dc4dc418aaf79`。

新physical selection SHA `05ef862ed42ac6d6d9ad7b3d227ebf9bc4d04e7d3a963b3190f3345a0b8a219d`，
prepare SHA `3c1fcd64f87801533b93cf8d7f2a1477240e7065371b063239b73d4ef85f7bce`。
实际物理报告 `le-wm-main/results/p1_constructive_observed_suffix_physical_20261027_v1_main01/report.json`，
SHA `f864cc80a88eb289b578d271679d0bc895043a3b7bc798ae141097797a555b5e`。
四分支均initial_exact、完整10步、0success、errors=[]；80public/816physics/88PNG。
MAIN实际独立saved physical audit状态complete_saved_protocol_verified、findings=[]。

| slot/原臂 | 模型选定后缀 | 原物体终距→反馈终距(px) | 原wrapped角误差→反馈(rad) |
| --- | --- | ---: | ---: |
| 0/GN | original | 47.977→47.977 | .257051→.257051 |
| 0/D2P | fresh ridge | 64.251→65.231 | .476316→.514473 |
| 1/GN | original | 130.929→130.929 | .607269→.607269 |
| 1/D2P | zero | 45.399→63.090 | .494417→.364185 |

这不是在线runner，而是已知任务saved-prefix反馈反事实重放。不能因角度或
含速度native distance改善就称物体任务成功；新后缀在两条D2P臂上未改善物体位置。
原prefix各步state/PNG逐字节再现另有独立审计任务，不仅凭命令相同推定。

**GN正则单位差异已用独立variant检验。** 原GN在A/sqrt192、r/sqrt192上加
.002，等效未归一化λ=.384；D2P ridge用未归一化λ=.002。新GN明确
λmean=.002/192，保留原4更新/5尺度/动作球/任务/模型及每根78predict预算。
28项MAIN测试和独立复审通过；不修改旧模块globals或sealed源码。
这匹配系数单位，不匹配线性化点、维度、增量与suffix、预算或全部目标。

prepare SHA `cec47cb477b06cbbecc9a402f6f17415d3a67cec576679aab906dc68a733bf05`；
实际CUDA报告 `le-wm-main/results/p1_constructive_matched_ridge_gn_cuda_20261027_v1_cuda_main01/report.json`，
SHA `7eed1f6ab7543e6dc4be367c5141ab23cb44dc014fae770fc58fe813422137d5`。
总156predict/156action_encoder、8J/1536VJP，model unchanged、errors=[]；
MAIN实际consume完整核验通过。两根model terminal Q=1.014635、1.455195，
均高于原GN的.667756、1.291969，故不能宣布正则差异已解释或修复失败。

新physical selection SHA `d75e40c3032f19e899e4982c8867a4c9073d62c59bf429241f973a3c5b016dcc`，
prepare SHA `ec4582defd0e2dafa97815ce6691778324e05c4f1f2e0816adf5f54e08daf965`。
实际physical报告 `le-wm-main/results/p1_constructive_matched_ridge_gn_physical_20261027_v1_main01/report.json`，
SHA `47d46ffd9664d63e19a1906696b448aa9f06c9259f216d9a343b42ac554752ff`。
两分支完整10步、initial_exact、0success、errors=[]，40public/408physics/44PNG；
物体终距58.279779、146.200775，wrapped角误差.306245、.785011，均劣于原GN。
MAIN实际独立saved physical audit complete、findings=[]。正则算术独立saved
审计另行推进；不把普通测试或producer consume称独立神经导数验证。

当前具体剩余：反馈/新GN实际RGB复算、精确prefix再现、冻结模型训练渲染
合同核查及同任务强采样对照。核心P1仍缺严格非贪心、共同真实能力探针、
独立确认与强等资源对照；本节没有打开P1，也没有把负结果升级成理论否定。

### 上述剩余项续报：新RGB与精确prefix均完成

通用RGB v2为独立新源码，MAIN实核31tests；它不猜不同compiler的预测字段，
predicted_Q/MSE显式unknown，actual_Q/root_Q独立计算。
反馈8图prepare SHA `7127b5eda8c086a7a6334c08a846c124b78be9edce76bbc6ddefc715776be10d`；
主机CUDA报告 `le-wm-main/results/p1_constructive_observed_suffix_rgb_20261027_v2_cuda_main01/report.json`，
SHA `2cc16ea94dc76640a09ddb6d3d3cf8c70f171d9423b3896702715d48132306b1`。
新GN4图prepare SHA `3ecc41c3a4d2274c5e81136cab663aaaa0dbe2e8545d4d557006ac9b0b9ca1cb`；
主机CUDA报告 `le-wm-main/results/p1_constructive_matched_ridge_gn_rgb_20261027_v2_cuda_main01/report.json`，
SHA `f6dbf29177769e8ec0be9e5ea38b928d201812a5ea44c07fbcedd0ebd3ebabbe`。
两轮errors=[]、source verified、model unchanged，分别8/8及4/4 encoder/projector，
0predict/AD/physics。新GN实际Q5/Q10为根0 1.353260/1.342290、根1 1.477724/1.487518。

MAIN另复验独立prefix auditor23tests并实际运行，保存
`le-wm-main/results/p1_constructive_feedback_prefix_replay_audit_20261027_v1_main01.json`：
四臂0..5 state/snapshot精确到float64位、前5commands精确到float32位，
24对prefix PNG完整文件bytes及12对warmup检查点完全一致。
因此本次反馈物理比较不是错根或前缀重放漂移；这不认证未保存的外部过程。

**新的具体失配在same-prefix后缀比较中出现，不与原跨臂比较混淆：**

| D2P根 | observed-context模型Q：原后缀→选中后缀 | 实际Q：原后缀→选中后缀 |
| --- | ---: | ---: |
| 0 | .586243→.539916 (ridge) | .856466→.875436 |
| 1 | .496337→.487855 (zero) | .385148→.428327 |

模型都预测下降，实际同Q都上升。这个已知两根反馈pilot的排序反转支持继续查
局部模型调用/历史接线/导数/渲染合同，而非把反馈重编译宣称为成功修复。
也不能据此推翻原开环GN/D2P跨臂同Q排序未反转的结果；比较对象不同。
独立saved RGB v2算术审计与backend接线复核已派发，尚不能称全部代码可能性穷尽。

训练渲染合同只读审计发现：本地HDF5固定三帧均可见绿色目标标记，而
constructive明确with_target=False；checkpoint→原训练run→数据revision/渲染设置
绑定缺失，故checkpoint实际训练overlay合同仍unknown。不能把环境默认True
或这三帧现象直接称当前失败根因。详见
`le-wm-main/innerdocs/p1_constructive_render_training_contract_audit_20261027_v1.md`。
新的同任务Random Shooting/CEM与原始零点GN中心差分核查并行实现，
均不得读取witness/native outcome挑动作，MAIN负责后续实际CUDA。

### 强采样与中心差分：已完成实际 CUDA、物理及独立复算

**不是只新增runner。** 同两任务Random Shooting64与CEM4×16/top4，种子固定，
每臂每根128候选predict+6赢家三重replay，总536predict/536action_encoder；
MAIN40tests与独立review通过，实际CUDA errors=[]、model/source unchanged，
完整consume核验通过后才下发物理。prepare SHA
`4c209a1a2fb1b6c9a9e3067b21b5bc3dd6e3364234cada2278b6071d6668b4ce`；
报告 `le-wm-main/results/p1_constructive_sampling_cuda_20261027_v1_cuda_main01/report.json`，
SHA `0c2db96dc506a7a2d71e4dcd43f5990fe0cc8ad0ad27f8483388cecdf47b9698`。

物理selection SHA `bab483616c0fe888da3b5142ca8898191154b874f417422f676ba8703aafe5cc`，
prepare SHA `b84ff2297c1e51522e527c537b603b9a60bfda6c0b705d679f86684ce04ff668`；
报告 `le-wm-main/results/p1_constructive_sampling_physical_20261027_v1_main01/report.json`，
SHA `b67710a09b109c2276556d10b5fdbc3bca8e4ce44d31b302cd65ee849950c3ef`。
四臂initial_exact、完整10步、errors=[]、0native success，80public/816physics/88PNG；
MAIN实际独立physical saved audit complete、findings=[]。

| slot/方法 | 模型终态Q | 实际终态Q | 物体终距px | wrapped角误差rad |
| --- | ---: | ---: | ---: | ---: |
| 0/Random Shooting | .350869 | .944553 | 41.404063 | .269803 |
| 0/CEM | .386848 | .175161 | 20.566964 | .174164 |
| 1/Random Shooting | .395743 | .829263 | 95.277352 | .247132 |
| 1/CEM | .130895 | .259745 | 32.131889 | .022991 |

实际RGB8图prepare SHA `c691061a839ca0fa328e9239fd49333c60e00045670e798ede2cd052680b69a2`；
报告 `le-wm-main/results/p1_constructive_sampling_rgb_20261027_v2_cuda_main01/report.json`，
SHA `70749751378933daf4639fffbe7c2f36dd2d4f00cedbcfdcfb3608b29f52d829`。
主机CUDA8encoder/8projector完成，模型未变、errors=[]；MAIN另实核独立RGB auditor
36tests并实际复算sampling8图，状态saved_rgb_v2_bindings_metrics_verified。
同一auditor的feedback8图与matched4图也由MAIN实际复验通过。

CEM两根物体位置/角误差及实际Q均优于本批原D2P，必须保留这一不利对照；
但每根134predict vs D2P66，不能称公平总资源排名或独立泛化。
root0 CEM物体终距仍20.566964，原生位置还包含agent；不能改20阈值追认为成功。
这一结果也说明不能把所有失败简单归为冻结模型完全不能控制这两任务。

**原GN首个零点的中心差分实测已完成。** 两根各3固定方向×3eps×minus/plus，
加各1独立nominal，38完整H2=76predict/76action_encoder；无新AD/J/encoder/physics。
MAIN32tests与独立review通过。prepare SHA
`9485b6ddc08f246b27649744ac745f2bdecfb6bbc6dd809686cff4974351754a`；
报告 `le-wm-main/results/p1_constructive_gn_fd_cuda_20261027_v1_cuda_main01/report.json`，
SHA `107b958988530df78592aa2a98701cc091ba9daaddbabb31aa984402e76cc23d`。
两个nominal与原保存AD完整context精确相等，全部18central comparisons保留。
有效float32方向relative L2范围约4.84e-5..6.86e-4，cosine最小约.999999778；
这是受测方向数值证据，不认证全部导数或物理模型正确，也不设事后精度门槛。
独立saved FD复算报告
`le-wm-main/results/p1_constructive_gn_fd_saved_audit_20261027_v1_actual01.json`，
SHA `7d6d6b829c5dbf526c547590e4d20ff5a174f47bf531e5b82d159135cd7aacdb`，
核验456events、38H2、76/76、18比较；MAIN后续可再次复验。

新GN正则的独立saved算术亦由MAIN33tests及实际main01完成：8J/1536原始行、
8734events、156predict/156action_encoder闭合，未归一化λ=.002确实执行。
“系数匹配但无改善”是实测，不再仅依赖producer声明。

主线程将反馈两次排序反转写成独立纯saved contrast，补齐外部报告SHA、
actual dispatch→suffix winner→full plan及exact prefix latent绑定；17tests包含
完整actual分析的内存篡改拒绝。独立review捕获的winner命令漏检已修补，
不解释成实际错动作。补强后report
`le-wm-main/results/p1_constructive_feedback_contrast_20261027_v1_main02.json`，
SHA `f6f061e117fbf37a465044d69d57cac17a6b57ba2df751dc66a578f514557149`，
固定四分支中2changed、2strict same-Q rank reversals；main01保留。

另查到normalizer来源尚未闭合：MAIN只读全H5 action得到mean
[-.00781256438,.00686068723]、std[.208467442845,.206748626374]，
比public std大约3.25%/3.27%。本地public字面常数可追到JEPA-WMs precomputed stats，
但缺dataset版本/统计子集身份；当前LeWM train/eval从实际数据列求统计。
这不支持再除100，也不能仅凭差异替换旧任务常数；checkpoint真实训练stats仍unknown。
详见 `le-wm-main/innerdocs/p1_constructive_action_stats_provenance_audit_20261027_v1.md`。

当前关键在实现中的新增项：真正发生反转的observed H1处中心差分，以及
离线HDF5原始观测/动作的冻结模型fidelity核查。前者不能用已完成的根零点H2
FD冒充；后者是输入合同诊断而非新根科学确认。P1仍未通过。

### MAIN续报：observed H1 FD与归档fidelity已实际完成

以下实际产物取代上段两项“实现中”的状态。代码审查及保存算术复核由MAIN承担；
遵照用户最新要求，不再专设代码审计子智能体，现有名额改用于实验实现。
实际新增agent请求被平台thread limit拒绝，没有宣称增加成功；继续复用现有六个名额。

**Observed-prefix H1导数检查：** MAIN阅读producer/tests/doc，38 tests通过；
prepare SHA `9b8e0914f10a7a8785563ad6c3d867a290187a0685a1b4f64e2ff8c9487ccc3e`。
主机CUDA报告 `le-wm-main/results/p1_constructive_observed_suffix_fd_cuda_20261027_v1_cuda_main01/report.json`，
SHA `4fc95db60dbd62d88a9e512b6120379264780123813118736ac93e8256ee5245`。
38 H1 predict/38 action_encoder完成，errors=[]、模型未变，无新AD/图像/物理。
两根nominal完整context精确重现原反馈zero；18中心比较的有效float32方向relative L2
为 `5.5585063e-5..6.8615995e-4`，cosine最小 `.9999998049052912`。
MAIN另用不调用producer分析的NumPy算术检查266保存events、38完整evaluations、
3 artifact SHA，18份FD逐元素差为0，两个原anchor相等。
这削弱了受测observed H1零点/三方向的导数实现错误解释，不认证全部J、
全动作球或物理模型，也不解释唯一的后缀反转根因。

**固定归档fidelity pilot：** MAIN阅读producer/tests/doc，25 tests通过；
prepare SHA `acd022d451250c51de4497ebfb4713ac37af2faa7903e0b25f2d0a3ee341637d`。
主机CUDA报告 `le-wm-main/results/p1_constructive_offline_fidelity_cuda_20261027_v1_cuda_main01/report.json`，
SHA `e2d4bbb6e4f593c7fb8b2c2246bdfc16db54afbea7e9696aa264cb951154a385`。
20 encoder/20 projector/20 predict/20 action_encoder完成，errors=[]、模型未变，
无训练/AD/物理/动作选择。MAIN核验51 artifact SHA并从实际NPY独立重算全部误差，精确一致。

| 固定episode | H1 factual MSE | H1 zero对factual标签 | H1 sign对factual标签 | H1 persistence | H2 factual | H2 persistence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | .00269542 | .10269463 | .22215993 | .12797125 | .00792686 | .59530126 |
| 1 | .00252851 | .09940851 | .21758287 | .12427374 | .00636555 | .57756950 |
| 2 | .00259891 | .10059461 | .21928248 | .12559479 | .00830911 | .57251025 |
| 3 | .00262041 | .10073228 | .21903543 | .12575374 | .00781010 | .56135131 |

只支持这几段本地归档及当前public mapping下，实际动作条件预测优于固定消融和持久参考。
zero/sign没有反事实真实标签，不能把表中比较称因果物理动作效果。
checkpoint训练normalizer/overlay与collector时间来源仍unknown，不能据此认证训练复现。
这也不是先前source上闭环成功，不能反推live renderer或所有动作接线正确。

**新增相关性警告：** 四条动作相对episode0的最大逐坐标差仅0、.00265586、.00154841、.00266731；
episode1与3的frame0/frame5原始像素SHA完全相同。不扩大成四个独立演示或泛化样本。
已派NEW v2实现事前固定的四分散索引、W10/W15 teacher-forced窗口与F/Z/embedding-mask，
当前仅任务分派，未声称v2已完成。v1与该设计的episode/窗口/消融不同，结果不得混报。

P1仍缺严格非贪心、共同真实能力改变、实际任务收益、强全资源对照及独立根确认。
下一批并行实现包括共同prefix固定物理probe、66-forward采样、MPPI、gradient MPC；
均待MAIN核查与真实执行，不把mock/代码交付当研究闭环。

### MAIN续报：66-forward随机/CEM完成真实执行与RGB

MAIN完整审阅NEW matched_sampling producer/tests/doc，52 tests通过；
prepare SHA `c992ba39020d2c9f11cd534d695c5aba73a921972a4664201bc22bd12f15c8aa`。
主机CUDA报告 `le-wm-main/results/p1_constructive_matched_sampling_cuda_20261027_v1_cuda_main01/report.json`，
SHA `53b2a4e7c0ba4307b06ca8c9ff5374e97a91a04ac28dc608c99b7b8eb0f3a5fd`。
固定两根×random32/CEM2x16，每臂/根66predict与66action_encoder，合计264/264，
errors=[]、模型未变；完整consume重建随机流、候选、选择与实际H2 journal通过。
只匹配forward次数，D2P的AD/J/SVD仍另计，不称等总算力。

MAIN新物理selection固定在首次物理之前，prepare SHA
`19aa91e5e4cac5b0dc1c9bc21b63b0aaa68f865931dfb99f95f0685398d8142a`。
实际physical报告 `le-wm-main/results/p1_constructive_matched_sampling_physical_20261027_v1_main01/report.json`，
SHA `b735655a98ecb52f10967d2a2f9cbfe53ef72a00ccc064799e0be275fa1a01a7`。
四分支initial_exact/full10、errors=[]、0native success；80public/816physics/88PNG。
MAIN实际saved物理核查complete_saved_protocol_verified、findings=[]。

RGB prepare SHA `7f45f983596b8ba2d5763670f313f42ad990ef9868e9bafdb798ba9bb9d7d195`；
主机CUDA8encoder/8projector完成，errors=[]、模型未变。报告
`le-wm-main/results/p1_constructive_matched_sampling_rgb_20261027_v2_cuda_main01/report.json`，
SHA `8a14c1c6b1400e9e3713524dcb6ecfa001dea3f98edafa6901976a9136f7ccd5`。

| root/方法 | 模型terminal Q | 实际terminal Q | 物体终距px | wrapped角误差rad |
| --- | ---: | ---: | ---: | ---: |
| 0/random32 | .350869 | .944553 | 41.404063 | .269803 |
| 0/CEM2x16 | .571099 | .286735 | 35.874988 | .188161 |
| 1/random32 | .568486 | .705005 | 27.595935 | .048092 |
| 1/CEM2x16 | .180130 | .569888 | 53.663556 | .065347 |

不能再说D2P在这两个known calibration tasks上已稳定优于强采样：
root0的CEM实际Q/物体终距优于原D2P；root1的random物体终距优于原D2P，
但原D2P实际Q较低，指标并不一致。四臂均未达到原生成功；不调阈值、不选最有利指标作结论。
原134-forward采样结果完整保留，不被新低预算结果替换；两批共有seed/候选祖先，不扩大独立样本量。

主线程另指出MPPI-inspired v1使用best-evaluated候选派发，不是标准nominal MPPI，
因此不会冒充论文MPPI强基线；已要求保留v1并另做明示真实预算/nominal派发的v2。
不为凑66forward而牺牲基线算法语义。

### MAIN续报：84共同物理probe完成；分散归档v2完成

**共同真实能力probe已完成，不再只是设计。** MAIN完整阅读NEW实现/tests/doc，20tests通过；
prepare SHA `70616f76f08f560301cd0b5445cad8fcb41a9e53415bcac7a9d0efbf34cf560e`。
实际报告 `le-wm-main/results/p1_constructive_common_prefix_probe_physical_20261027_v1_main01/report.json`，
SHA `450b182e4ed4b02959f74593099cf85c728619bfc5ddc3678ea1940c846e1210`。
84/84完整、errors=[]、source_verified=true，实际233.05秒；84prefix精确复现，0native success。
168reset、1680public、17136physics、1848PNG、168goalrender、336setter、84goalbind全部实际attempt/return闭合。
MAIN不用producer分析函数，另核48974事件hashchain、1850artifact SHA、操作进入/返回身份、
每position全21分母/精确prefix签名，并从真实终点复算40列B/E与共同方向a；a最大差3.56e-15（算术顺序舍入）。

| root/真实prefix位置 | 角色 | 共同方向secant向量L2 | 其中物体贡献L2 | 物体响应矩阵Frobenius |
| --- | --- | ---: | ---: | ---: |
| 0/zero | Z | 91.113593 | 21.248240 | 44.954589 |
| 0/原D2P prefix | G=D | 72.834651 | 3.212725 | 22.841458 |
| 1/zero | Z | 89.672202 | 28.027481 | 46.342782 |
| 1/原D2P prefix | G=D | 93.044915 | 31.065062 | 57.016479 |

共同方向由原root/goal固定，不是各position own residual；物体贡献使用同一四维方向的物体分量，
未重归一化/拟合有利权重。B为rho=.25、5步真实有限secant，非无穷小J、全局可控性或任务成功。
真实响应确实随prefix改变，但root0的受测能力幅度下降，root1部分上升；不能称稳定能力扩大。
G=D是同干预alias，因此仍无非贪心证据；84条不变成84独立根。已分派同批固定probe的
NEW CUDA实际RGB/模型响应对应实验，尚未完成，不能用旧J冒充已预测这些响应。

**归档分散v2已完成：** MAIN完整读实现/tests/doc，51tests通过；
prepare SHA `580dd9e35431423885041da76bbdbca3a033808dae7db40aabbaff25e1c34adb`。
实际主机CUDA报告 `le-wm-main/results/p1_constructive_offline_fidelity_cuda_20261027_v2_cuda_main01/report.json`，
SHA `e2ef4ce025d52f62e1688b268d2dab03021c248c4c8c5a7e2bf54ae98352b5e9`。
固定episode[0,6228,12456,18684]，两teacher-forced窗口各F/Z/outgoing-embedding-M；
20encoder/projector、24predict/actionencoder/predictor/pred_proj均完成（predict API与predictor不相加），
每预测3tokens共72，errors=[]、模型未变，无新物理/AD/训练。
MAIN核217artifact SHA；从保存NPY独立复算24误差（差0）、完整输入history/time-major publictokens与mask。
8/8窗口factual误差均低于Z、M和persistence；F/Z/M八窗口均值分别
`.0038933154/.0705292411/.1086163351`。四分散episode相同步无相同RGB，但演示祖先独立性仍unknown。
这支持记录数据当前mapping下动作条件预测有效，不认证训练来源、live泛化、P1或反事实物理标签。

MAIN同时发现v2明确telemetry代码问题：初始化Limits默认free为测试常数FLOOR+CAP，
真实disk_usage注入前先check，导致minimum_observed_free_bytes=243269632不是实际主机最小剩余量。
实际df约61GiB；后续dispatch仍检查实际disk floor。此问题不改变24预测/NPY/误差，
但该字段不得作为真实最小磁盘证据。已要求保留v2封存，NEW v3从首次check就注入真实disk_usage，
不能改旧report抹掉错误，也不把telemetry错误归为理论或实际预测负结果。

### MAIN续报：同轨迹目标标记开关实测完成；审计留在主线程

按用户最新要求，代码阅读、源码绑定和保存数组复算由MAIN承担，不再分派专门的审计子智能体。
子智能体继续实现gradient MPC、nominal MPPI、GRASP适配与共同能力probe等互不覆盖的实验。

live overlay renderer报告
`le-wm-main/results/p1_constructive_live_overlay_fidelity_20261027_v1_main01/report.json`
SHA `567e58769764f68107e099f4e2494159d9ef0076c7bbd73445a62245be6d4e80`。
原四分支off渲染88帧/state/snapshot/文件字节均精确复现；新增16帧on图像，
同一状态渲染期间无advance/setter，104explicit render、80public、816physics。
on使用原生默认goal_pose，不是独立任务goal；训练时overlay协议仍unknown。

实际主机CUDA报告
`le-wm-main/results/p1_constructive_live_overlay_fidelity_20261027_v1_cuda_main01/report.json`
SHA `d563fe54ae16a04ff184bc87cfeba3450ca98fd0a87e24d0084955965b7f3a58`。
32encoder/projector、8predict/actionencoder完成，errors=[]、模型未变；不优化、不选择、无新增物理。
MAIN独立核42artifact SHA，从保存NPY/预测重算全部8组四项误差，最大差0；
同时逐组核输入仅为自身view的warm5/warm10/prefix5，不含terminal10标签。

| root/arm | off预测MSE | on预测MSE | off保持当前状态MSE | on保持当前状态MSE |
| --- | ---: | ---: | ---: | ---: |
| 0/GN | .102225 | .096900 | .098444 | .077041 |
| 0/D2P | .090957 | .060627 | .100036 | .089722 |
| 1/GN | .208782 | .166769 | .252048 | .288906 |
| 1/D2P | .062703 | .072957 | .191660 | .241162 |

四分支三项预测误差下降、一项上升；开启标记也改变标签的编码坐标，不能直接解释为动力学改善。
root0/GN两view均不及保持当前状态。此轮仅factual，无zero/mask动作对照，不能证明动作依赖，
不能归因为训练渲染不一致，更不打开P1。已分派NEW v2补F/Z/M控制，保留全部v1封存。

### MAIN续报：梯度MPC主机CUDA完成（尚无该臂真实执行）

MAIN完整阅读gradient MPC源码、tests与协议，36项纯数组/RAM测试通过；
prepare SHA `83371ef03c1b18727d8632b6a69085f04638189d7991ecd7831fa3affb62ca8f`。
主机CUDA报告
`le-wm-main/results/p1_constructive_gradient_mpc_cuda_20261027_v1_cuda_main01/report.json`，
SHA `8eadbc6068ee4949f280892442eda45298934ece87dd7ae38a44b4f93d78d728`。
两根272predict/actionencoder、64scalar backward、64proposal与2selection全部attempt/complete闭合，
errors=[]、模型未变。64条grad API receipt对应同64次backward，不能相加。
固定zero初值、32次eta=1投影更新，全部33迭代后模型选优；不训练、无线搜索、不接物理结果。
实际消费gate完整检查保存迭代/梯度/因果日志与三路径重放，命令释放通过。
root0模型最佳iteration32、terminal meanQ=.18272003826943647；
root1最佳iteration23、terminal meanQ=.09783419858162906。
这仅证明完整H2可微优化器运行和模型目标下降；不是闭环反馈MPC，不是forward/总资源等预算胜出。
本臂尚未进行真实物理执行或terminal RGB编码，不推断真实收益，不增加独立根，不打开P1。

### MAIN续报：共同probe被跨注意力内核exact gate拦截，已定位数值协议问题

MAIN完整阅读common-prefix CUDA v1/source/tests/doc，21tests通过；
prepare SHA `f151267742dca9b6e05492eac155fc63840cc1bf55b86de37203e7bc122bdad2`。
真实HOST CUDA报告
`le-wm-main/results/p1_constructive_common_prefix_probe_cuda_20261027_v1_cuda_main01/report.json`，
SHA `68e890823fa2ac1c75a0d885c4ed3e4eff716788696c86fe713aeb10faf63173`。
92encoder/projector完成、0predict/actionencoder，模型未变；在旧task warm5/10逐bit认证时拒绝。
不是84个预测失败或理论负结果。4个warm的最大绝对差为
`1.430511474609375e-6,1.430511474609375e-6,9.5367431640625e-7,9.5367431640625e-7`。

MAIN源码定位：原blind encoder只禁matmul TF32，未固定SDP backend；新v1对encoder和predictor都固定MATH。
随后使用同4张封存PNG做实际主机CUDA交叉诊断（仅终端输出，无另建sealed producer）：
8次编码、0predict、同一冻结模型，默认SDP/cudnnTF32=True与MATH两条件。
4/4默认编码逐bit等于旧task，4/4MATH编码逐bit等于本次拒绝现场，模型tensor hash未变。
该诊断确认此exact gate拒绝来自数值backend协议差异；并不证明MATH/default能解释以往大幅排序负结果。

修复另建v2，保留v1及全部失败现场：增加4次原协议编码exact认证旧task，
再用92次固定MATH编码组成真实own-context与labels，继续84预测；合计96encoder/projector。
不修改旧artifact、不放宽容差、不用旧latent代替实际prefix；额外成本和数值差异须显式保存。
v2实现已分派，尚未完成运行，不能将修复建议当成功结果。

### MAIN续报：nominal MPPI模型运行完成；gradient MPC真实执行出现1/2成功

MPPI v2完整源码/tests/doc和原kernel已由MAIN阅读，33tests通过；
prepare SHA `139b0000336f5ed6764b46ea1ca8c8fd5882346009f8fb86829ea3f2886deccc`。
实际HOST CUDA报告
`le-wm-main/results/p1_constructive_mppi_cuda_20261027_v2_cuda_main01/report.json`，
SHA `0db92a35da17b21eca3fa9d11ab90025bc8610aabe7e634af55c18ddaa9c82f9`。
144predict/actionencoder全部完成、errors=[]、模型未变，consume完整参考kernel/nominal日志重算通过。
两根按预定index34末轮nominal派发，模型Q分别1.3186646514482832、1.4738642569294973。
不因分数差偷换best sample；此为项目clipped-Gaussian kernel baseline，非官方或density-corrected MPPI。
尚无其实际物理结果，不填论文强MPPI复现行，不声称相同总计算资源。

gradient MPC physical adapter经MAIN完整阅读，测试首次出现fixture闭包引用mutable manifest的问题，
篡改同时改变expected；MAIN在测试内deepcopy冻结预期值，修后32tests通过。
这是测试代码问题，不是生产物理/理论失败，修复发生在physical prepare前；原CUDA封存未改。
selection SHA `667f6fb6ee305e88428e8227812c44c312623076357b9c238b4db403a6a37c38`，
physical prepare SHA `cabfc3a3ab6560ed9ca13e7ca4fa2960197ace7ad4b98c308927276b21c05984`。
主机真实物理报告
`le-wm-main/results/p1_constructive_gradient_mpc_physical_20261027_v1_main01/report.json`，
SHA `134356eb4524998055e46b83d30cfb44369195356b8890ac1a6aea170208a75a`。
两根完整10own steps、initial_exact、errors=[]；40public、408physics、44PNG、4reset、8setter、4goalrender、2goalbind。

| root | 最终agent+object四维XY误差 | wrapped角误差rad | 物体终距px | 原生成功 |
| --- | ---: | ---: | ---: | --- |
| 0 | 40.569058 | .237491 | 34.911915 | false |
| 1 | 17.435800 | .189585 | 13.663915 | true |

MAIN直接从state/goal数组复算原四维XY<20且wrapped角<pi/9，两根与native标记一致；
root1不是历史负角误判。此前D2P两根0成功，因此本批不能声称D2P优于普通动作梯度优化；
且gradient MPC单根136predict+32scalar backward不与D2P66等资源。
这提示原这两个校准任务不足以证明“普通梯度停滞而reconfiguration获益”。
仅两旧known tasks，不能把1/2当泛化成功率；新物理terminal RGB评估尚待完成。

### MAIN续报：common probe v2数值修复实测闭合；局部响应与跨配置增益分开

MAIN完整阅读v2源码/tests/doc，15tests通过；prepare SHA
`f0b90e73408d6e64b488d9e35553db8663d38d1d11aa43bfe9e9532f00e2764f`。
实际HOST CUDA报告
`le-wm-main/results/p1_constructive_common_prefix_probe_cuda_20261027_v2_cuda_main01/report.json`，
SHA `73959e915161c3e756068bc30606c7c14a84c9806b52d47ca918b1dbcf598b46`。
4默认backend认证+92MATH编码、84MATH预测均完成、errors=[]、模型未变；
4默认warm float32字节与原task严格相同，无容差。MAIN核两artifact SHA、1092journal事件，
360实际module attempt/return、96image receipt与84prediction receipt逐条对report完全一致；
从保存输出独立复算4组192×10中心secant，最大差0。

共同latent方向固定原goal−root，表中为该方向下10维中心secant投影向量的L2，
不是各prefix自己的残差方向，也不是物理XY方向；rho=.25、固定21probe，非无穷小AD。

| root/position | 模型方向幅度L2 | 真实编码方向幅度L2 | 同位置方向向量cos | 完整响应矩阵cos |
| --- | ---: | ---: | ---: | ---: |
| 0/Z | 3.054263 | 6.047182 | .991361 | .843735 |
| 0/G=D | 4.686259 | 3.470730 | .926085 | .885166 |
| 1/Z | 3.550284 | 1.311188 | .988830 | .601855 |
| 1/G=D | 2.610313 | 3.385523 | .861582 | .731737 |

局部投影方向在多个position一致，但跨prefix能力幅度增减两根均与实测反向：
root0模型预测增加、真实下降；root1模型预测下降、真实增加。
故“模型完全没有动作响应”不是准确描述；“局部方向较对”也不能推出重配置增益估计可信。
按10列投影符号一致数分别10/10、10/10、10/10、5/10。
这是固定两knownroot、4position有限干预的失配证据，不是独立确认、理论不可行证明或P1通过。
4position中G=D仍是同一动作alias，无非贪心证据。下一步需区分局部/有限幅度估计与控制目标失配，
不得通过事后选position或更改成功阈值掩盖。

### MAIN续报：overlay F/Z/M动作条件消融实际完成

MAIN完整阅读v2源码/tests/doc，28tests通过；prepare SHA
`ca71842b0277e75b23d6044b4adbbb55a123b19a479a9afa8ccc30c453f74587`。
HOST CUDA报告
`le-wm-main/results/p1_constructive_live_overlay_fidelity_20261027_v2_cuda_main01/report.json`，
SHA `068ed8f1bc7da1a92850738df7a6af061cceb0ec6047c16ea5ee30455eb73631`。
32encoder/projector、24predict/actionencoder，errors=[]、模型未变，无新增物理/AD/训练。
MAIN核60artifact SHA，从原NPY与24保存预测独立复算全部F/Z/M MSE，差0；
逐项核M只清最后action embedding、前两history embedding不变、F/Z不改embedding。

8个branch/view中，factual误差7/8低于raw-zero控制、7/8低于outgoing-embedding-mask控制；
root0/GN/on的zero略优于F，root1/GN/off的mask优于F，保留这些反例。
Z/M均用同条factual终态作条件消融评分，并无Z/M真实反事实终态，不能叫其真实动力学误差。
两view标签坐标不同；不能把on的部分低误差解释为训练overlay问题已确认。
此结果排除本批“模型完全忽略动作”的简单描述，但不认证反事实排序/重配置增益/真实任务收益。

### MAIN续报：gradient MPC真实RGB评价链完成

MAIN完整阅读RGB adapter source/tests/doc，26tests通过；
prepare SHA `f63b02c64b16d0c5669484fc8bd191579cbf19abdce9bc5d28391bc7a79c9615`。
实际HOST CUDA报告
`le-wm-main/results/p1_constructive_gradient_mpc_rgb_cuda_20261027_v1_cuda_main01/report.json`，
SHA `d3cb06fa8aed651e74ec11d6415505fd5313d47d980a62835ff048895eef8fd0`。
4encoder/projector完成（两根各真实prefix5/terminal10），errors=[]、模型未变，0新predict/AD/physics。
MAIN核6artifact SHA，从4NPY与原task goal独立复算actual Q，最大差0。
两根prefix/terminal actual meanQ分别`1.11408753/.40203708`、`.46849874/.03526502`。
结合此前物理root0未成功/root1成功，gradient MPC的模型→独立真实执行→真实RGB链已完成；
这不是独立新根、非贪心机制或等预算对比，也不是整套研究闭环。
producer的predicted_Q仍null（未在此路径重新预测），不改旧报告假装已计算该字段。

### MAIN续报：nominal MPPI真实物理完成，未成功

MAIN完整阅读physical adapter/source/tests/doc，38tests通过。
selection SHA `61fefc1d36d2ddd87ccbaf26bffb038168e34b290dbd62bffaa6d79d3b7fa79c`；
prepare SHA `c7b9bead505e9cd1e84f7b626ae129db3aa764083aae242000dc3914070d7d02`。
实际物理报告
`le-wm-main/results/p1_constructive_mppi_physical_20261027_v1_main01/report.json`，
SHA `2f5221e8684100ab8ad78b96ce77e6a17a6b5f0ee908efca3a5fa9fba544ffdc`。
两根完整10own、initial_exact、errors=[]；40public/408physics/44PNG/4reset/8setter/4goalrender/2goalbind。
MAIN从state/goal独立复算native位置/角度成功条件，与两根native=false及wrapped=false一致。
root0/root1四维XY误差176.710682/215.434536，角误差.445786/.729439，物体终距72.662224/139.491876px。
0/2是这套固定项目kernel参数/nominal方案的真实结果，不调温度/噪声后替换，不上升为所有MPPI不行。
真实RGB编码尚未执行，已分派新adapter实现；不得写作完整主机CUDA→物理→RGB链已完成。

### MAIN续报：职责调整与nominal MPPI真实RGB链完成

按用户要求，代码审计、实验准入和结果复算均由MAIN承担，不再另设审计子智能体；
子智能体只实现互不冲突的新实验/基线，实际神经实验仍仅主机CUDA，不降级CPU。
MAIN完整阅读MPPI RGB adapter source/tests/doc并重跑32项RAM/mock tests，全部通过。
prepare SHA `764ba056feb952e5f8ad2d4c56e435369b1e9970e7bcd4fc81af5c61f24dacf7`。
实际HOST CUDA报告
`le-wm-main/results/p1_constructive_mppi_rgb_cuda_20261027_v1_cuda_main01/report.json`，
SHA `d730e5845b0d06edbbfb00a6dc1f6490ac4d5e6662ceef6abae3588fbeaf3dcf`。
固定两根各prefix5/terminal10，4 encoder/projector完成，errors=[]、模型未变；
无新增predict/AD/physics/训练。MAIN核6项artifact SHA，并从4份NPY与原task goal独立
复算actual meanQ，差0；两根prefix/terminal分别`1.120399365/1.211865580`、
`1.469183882/1.510607247`。复算初次误以为原export直接含gn_task字段，读取诊断报KeyError；
改用既有load_export接口后完成，不涉及producer代码、实验重跑或结果替换。
由此固定项目kernel nominal MPPI的CUDA→自身真实执行→真实RGB评价链完成，仍0/2 native成功。
这不等于官方/density-corrected MPPI、独立新根、等总资源对比或P1通过。

GRASP_H3_bounded已由MAIN复核并重跑40项测试（104.91s）；完整默认解析fixture不是CPU神经实验。
prepare SHA `4ee8a2a6526d4ecdf534a54b790150117022b53737323b0d78f0c603a4d710ff`。
实际主机CUDA已启动，输出stem `p1_constructive_grasp_baseline_20261027_v1_cuda_main01`；
在最终report、consume及真实执行完成前，不计为已完成基线或科学证据。

### MAIN续报：GRASP_H3_bounded实际CUDA与消费验证完成

上述GRASP作业已终止且完整成功，不再只是“已启动”。实际报告SHA
`501651be13c6dd6b6082bd20513d464c0ffc8aed053bb7a40f1c4ee9cd9388c5`，
路径`le-wm-main/results/p1_constructive_grasp_baseline_20261027_v1_cuda_main01/report.json`。
3604 predict、3604 action_encoder、1800 scalar backward的attempt/completed全部配平，
errors=[]、前后模型tensor hash相同；无训练、无CPU神经fallback。
MAIN执行原producer consume完整通过，并独立核6项artifact SHA、最终logit球可行性、
逐float32 tanh与全部10 raw commands完全相等、末端meanQ复算差0。
两根最终model meanQ为`.181553096920/.064177260527`；不能把model Q低直接解释为控制成功。
本版保留官方虚状态/soft dynamics/state-input stopgrad/noise/periodic full-chain sync与Adam reset，
真实H3和有界动作坐标是明示的本地适配，不冒称官方H1/论文数值复现。
新physical adapter正在由子智能体实现，MAIN随后审计与运行；当前尚无此GRASP计划的
自身真实终态与RGB评价，也没有独立新根、等总预算或P1非贪心机制证据。

### MAIN续报：GRASP自身真实执行完成，0/2原生成功

MAIN完整阅读physical adapter source/tests/doc并重跑46项RAM/mock tests通过。
selection SHA `ab6dcf51965091c388b0004037fc554f9fa7992011869f8cbdcbc8c31868e8f9`；
prepare SHA `6c617a16ff889ab06673b6dc9feb5f44fb7829aeb01bc1d21b9dfcb211d6d3c9`。
实际主机物理报告
`le-wm-main/results/p1_constructive_grasp_physical_20261027_v1_main01/report.json`，
SHA `bf7f15cf9d5aad4ca7923c4a116f6703c5e559b496c50488de55a4a94394f606`。
两根自身10条命令全部执行，initial_exact、errors=[]；40public/408physics/44PNG/4reset/8setter/4goalrender/2goalbind。
MAIN从终态/goal独立复算四维XY与wrapped角度成功：root0位置46.927528、角.146515、
物体终距18.835903；root1位置53.096669、角.284353、物体终距42.131246。均未成功。
物体位置/角度改善不能替代含pusher位置的原生成功条件；root0物体终距小于20也不能冒称成功。
GRASP RGB adapter已由MAIN完整读source/tests/doc，40tests通过，prepare SHA
`1ca4bc51b39db377b9ef6bc7d42101b0bf8f035c25e6fc79a2ef6f846b54a1e2`。
主机CUDA4图编码已派发，最终结果另记；不能在报告完成前计作完整RGB链。

### MAIN续报：固定四半径模型响应实际完成，84点逐位重放

MAIN完整读multiradius source/tests/doc，11tests通过；prepare SHA
`80a13ee858736501635b32d61647a473d8ad6b0ac184851cff88246e4e2f39c0`。
实际HOST CUDA报告
`le-wm-main/results/p1_constructive_common_prefix_multiradius_cuda_20261027_v1_cuda_main01/report.json`，
SHA `5379546ede547fe72af2b6a7182960d51b8c822f351531175e48092594bb5d0b`。
324 predict/action_encoder完成，errors=[]、模型未变；原zero/rho=.25共84点完整context逐float32位复现。
MAIN核2artifact SHA，独立复算16个192×10中心secant全部数组差0；
投影向量范数独立矩阵乘法与逐列点积算法仅有最大8.88e-16浮点归约差，不修改producer。

| position | rho=.0625 | .125 | .25 | .5 |
| --- | ---: | ---: | ---: | ---: |
| root0/Z，共同latent方向幅度L2 | 2.964973 | 2.978551 | 3.054263 | 3.168926 |
| root0/G=D | 4.541122 | 4.573264 | 4.686259 | 4.694265 |
| root1/Z | 3.749686 | 3.688658 | 3.550284 | 3.339522 |
| root1/G=D | 2.633489 | 2.626913 | 2.610313 | 2.578520 |

四半径模型均预测root0跨prefix增大/root1减小；同位置矩阵相对最小半径变化约1.5%–25.1%，
最大半径与最小半径矩阵cos仍约.972–.989。由此不能靠单改模型probe半径反转其配置间增益判断。
新增半径没有真实物理标签，不能据此排除真实接触非线性、认证局部物理导数或否定理论。
这是模型内部有限logit响应稳定性证据，不是AD/新根/P1通过。

### MAIN续报：archive4/live4固定输入统计落盘，发现真实静止历史差异

MAIN完整读source/tests/doc，36tests通过；实际saved-input报告
`le-wm-main/results/p1_constructive_archive_live_input_summary_20261027_v1_main01.json`，
SHA `8d8ec5b20938a619179234a16d9e8736aa93b340693ce9066edb589e33b85b20`。
4/4 archive、4/4 live描述完成，errors=[]；27小文件绑定与固定HDF5片段认证，0神经/物理调用。
MAIN独立重算两组汇总直方图counts与原uint8 RGB均值全部一致，差0。
每个live root历史三帧只有1种RGB，两个历史packet真实执行raw-zero；原四分支来自两根，
跨分支whole RGB history只有2种、whole raw history只有1种。archive四个W10各3帧/3packet均不同。
这不是虚构padding错误：静止图像与zero历史来自实际warmup记录；但揭示了具体部署context差异。
RGB边际TV约[.028206,.022112,.024693]，raw动作norm均值live−archive=.226358；
不能据此推断checkpoint训练身份、独立样本量或因果归因。
下一项固定同当前状态/同outgoing factual动作、只干预旧状态/历史动作的模型消融已分派实现，
用于检验历史输入敏感性；不训练、不选有利窗口，不提前称静止历史就是失败根因。

### MAIN续报：GRASP真实RGB编码完成，完整评价链闭合但未成功

GRASP RGB实际主机CUDA已终止且完成：4encoder/projector，errors=[]、模型未变。
报告`le-wm-main/results/p1_constructive_grasp_rgb_cuda_20261027_v1_cuda_main01/report.json`，
SHA `c1a74f179fc367e45e65e8e38c429d4b8a416efa9f1dd74453e6aa838d1dfd77`。
MAIN核6项artifact SHA，从原NPY和原goal独立复算meanQ差0：
root0 prefix/terminal `.635643652/.184152515`；root1 `.690425847/.295876365`。
两根model terminal meanQ为`.181553097/.064177261`，第二根的真实编码终态明显更差；
本RGB producer的predicted_Q仍null，不回写旧字段，以上模型值来自原封存GRASP结果。
root0真实latent接近model预测代价也不意味着native成功：物理四维XY误差仍46.93。

目前这批known-root上，gradient MPC为1/2原生成功，GRASP_H3_bounded与nominal MPPI为0/2；
不作显著性/独立泛化结论。每根搜索分别136预测+32标量反向、1802预测+900标量反向、
72预测+0反向，不是等总资源比较，也不能据适配版本主张已复现官方论文强基线。
完整研究闭环的缺口仍是独立根非贪心重配置机制与真实收益、强基线忠实性及等预算、
组合/自适应预算的独立增量和多环境/跨模型证据；这些不因本轮基线评价链完成而关闭。

### MAIN续报：归档历史F/S/A/SA消融已真实CUDA完成，静止历史不是充分解释

MAIN完整阅读新source/tests/doc，21项RAM tests通过；prepare SHA
`93537522ef3d800cba980f8b05779dd1cd86aba4c4f5249a3361a2f486d523f8`。
实际主机CUDA报告
`le-wm-main/results/p1_constructive_archive_history_ablation_cuda_20261027_v1_cuda_main01/report.json`，
SHA `2d20dadc8ff67515f59ba3495731d4d90a93f49d3fbe57b1cd7d4f4c53fff783`。
固定4episode×W10/W15×F/S/A/SA=32 H1，8个F的完整预测/动作embedding均逐位复现旧v2；
0新encoder/projector/AD/physics，32 action_encoder/predict/predictor/pred_proj完成、每类96tokens，
predict API与predictor不能相加成64次预测。errors=[]、模型未变、实际磁盘探针初始化，无CPU神经降级。
MAIN核229项artifact SHA，并从32保存prediction NPY/原label独立复算全部MSE，差0；
全部32保存history/raw/token与预声明干预逐项相同，128实际module attempt/return索引完整配平。

| 条件 | 固定8窗口平均对同一factual successor的MSE |
| --- | ---: |
| F：原真实历史 | .003893315391 |
| S：只将两旧latent改为当前latent | .004302927737 |
| A：只将两包历史raw改为0后按原统计归一化 | .004498737894 |
| SA：同时执行S/A | .003837776112 |

S−F在6/8窗口为正，A−F在5/8为正，SA−F在4/8为正；SA均值略低于F，不能挑有利窗口归因。
这些是在固定当前latent/outgoing factual action下的输入条件消融，不是新物理反事实的准确率。
结果不支持“历史静止本身足以解释当前live大幅失配”，更不能据此反推训练来源。
没有以提高阈值、改变normalizer或结果筛选修复；保留此前静止历史差异作为尚需受控检验的线索。

已预声明最小真实moving-history源协议：同原两个seed/placement，实际执行float32 raw
前5条[.1,0]、后5条[0,.1]，从新warmup终态构造新goal并独立重放两次；
不是改写旧payload或保持同root/goal，标记reused-recipe calibration而非独立新根。
旧blind loader/planner interface/physical branch均含zero-only假设，不能只换NPZ便假称接线完成；
子智能体分工实现NEW source、编码和planner薄适配，MAIN承担审读/actual运行，旧封存链不改。
本协议是端到端可行性测试，不预设非零历史能修复P1，也不据本32预测打开P1/P2。

### MAIN续报：moving-history真实源与CUDA盲编码均已完成

非零warmup源报告`le-wm-main/results/p1_constructive_moving_history_source_20261027_v1_main01/sourcepack.json`，
SHA `c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6`。
固定两slot均可达且两次独立重放成功；12 reset、120 public step、1224 physics step、126 render，errors=[]。
MAIN独立核验3428条事件哈希链、126PNG文件/像素SHA、两NPZ严格字段/float32 raw10、
construction/replay0/replay1的0/5/10历史像素、root完整动态snapshot及后续十步动态/像素逐位一致。
这只证明构造任务实际可达；witness动作不进入盲规划输入。复用旧placement recipe且改变root/goal，
不是独立新根、不是相同任务的历史因果比较，P1_admission=false。

MAIN完整审读新encoder/tests/doc，57项RAM tests通过；自行创建仅含两NPZ路径/SHA的审批manifest，
prepare SHA `4386cf3babd40df0eb21abdf8bed52913596f012fb017ddb4986af96ddafa599`。
主机CUDA报告`le-wm-main/results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/report.json`，
SHA `e7fb554327b3bb5a96190f6291f8450b2b0723cd981817715cfb7e50aa266bf6`。
真实8encoder/8projector、0predict/AD/physics、errors=[]、模型前后SHA相同，无CPU神经回退。
MAIN核12artifact SHA，8个原NPY对任务出口latent逐项相同、raw float32与两normalized token独立复算差0。
task00/01文件SHA分别`a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b`、
`101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74`。
新GN/D2P及非零warmup物理consumer实现中，尚无本批规划/控制结论。
主线程承担全部代码审计与实际运行核验；子智能体仅分担独立实验实现，不专设审计智能体。

### MAIN续报：moving-history GN/D2P已完成CUDA→四分支真实执行，尚未修复P1

MAIN完整读新planner/tests/doc，41tests通过、source冻结SHA
`f8f1d5507e2474fbbbfa56cae98f7bfa99f5929829946a2d7760c2c30ccc03c1`。
prepare SHA `dfe63b34bbca00131dd874633720b06ef5d332da0f3da2e6a29e739016d80080`；
主机CUDA报告`le-wm-main/results/moving_history_planners_cuda_20261027_v1_cuda_main01/report.json`，
SHA `cd8432c06288577d25d9d9b92c7eb3673de592499ee64cae93238156f3f19aa0`。
GN每根78predict/78action_encoder（两根总156，不能误写156/root）；D2P每根66，总计288/288。
四分支全部封存、模型未变、errors=[]，own consume核对全部实际forward的完整latent/token、
候选与AD/重放时序及原资源attempt/return。没有更换旧module globals或使用旧zero-only loader。
MAIN额外独立核各计划float32 tanh/raw、动作球及terminal meanQ，数值差0。
两个D2P winner分别6/7，均等于自身bank即时stage_cost最优者，不构成非贪心证据。

MAIN完整读新physical/tests/doc，39tests通过；source SHA
`44d90599eab881c773d5efa09177dfe6b395fa3f4eef41070340681fea5ec96e`。
以成功own consume创建NEW handoff：只显式省略大prediction字段，保留源report SHA/全部动作、
任务身份、selection seals与scalar model cost；完整prediction仍在原封存planner输出。
prepare SHA `226749e2bf8acdcd7a4a4a08be7336445daad0b6ad12185b24cbc0fb4560b1ff`；
真实物理报告`le-wm-main/results/p1_constructive_moving_history_physical_20261027_v1_main01/report.json`，
SHA `a5322dc4f483e82b5ed9d7b3996cf8a63845261d65774f90a4ae341773016aaf`。
四initial_exact/full10、errors=[]；80public、816physics、88render、8reset、16setter、8goalrender、4goalbind。
MAIN独立核2354事件哈希链、88PNG文件/像素SHA、四实际命令与各自selected动作完全一致，
以四维agent+object XY和wrapped角度重算原生成功均false：

| slot | arm | model terminal meanQ | 实际四维XY误差 | 实际物体XY误差 | wrapped角误差 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | GN | .600329583 | 89.030875 | 39.788663 | .279523 |
| 0 | D2P | .654757470 | 56.846847 | 16.925102 | .001532 |
| 1 | GN | 1.200545247 | 213.115680 | 134.735301 | .445289 |
| 1 | D2P | .166066805 | 56.015615 | 36.788671 | .074637 |

D2P这两根实际终距较GN小但仍0/2成功；不能用物体误差<20替代完整原生判据。
这是NEW任务上的校准结果，不是相同root/goal受控历史对比或独立泛化/等预算结论。
目前没有证据称“只换非零历史已修复失败”，也不能称理论已被否定；
四分支真实RGB编码和两固定构造轨迹factual successor保真诊断正由实现agent推进，MAIN待审读执行。
P1非贪心机制/真实能力改变、独立真实收益和后续研究闭环仍未成立。

### MAIN续报：固定moving-history factual successor保真诊断已真实完成

MAIN完整审读source/tests/doc，29tests通过；source SHA
`2d4b66eb5b40e4c9a87af67ae95b230ee05919d7fe1afd51fc51108a33945af4`，
prepare SHA `baf273a3ef54e1b7b8d4b675e2749ae2e22aa6dd30aff129396c8e68f208a784`。
主机CUDA报告`le-wm-main/results/p1_constructive_moving_history_fidelity_cuda_20261027_v1_cuda_main01/report.json`，
SHA `81bd7359b40216aae3d224f5d97a381d6f30ac73e0d15dbaea7d152d591e311f`。
固定两slot、每slot四预测，不按planner outcome选轨迹；只新增两个真实step15的encoder/projector，
实际8predict API/8predictor/8action_encoder/8pred_proj，各24tokens；不是16次预测。
errors=[]、冻结tensor SHA未变、0AD/physics/训练，无CPU神经回退。
MAIN核67artifact SHA、全部8保存raw/token/latent（含teacher-forced与fresh open-H2分叉）及8NPY MSE，差0。
每根root_H1与重新执行open_H2_first完整3×192输出逐位相同，非直接复用输出。

| slot | root→真实15 MSE | 接真实15后→20 MSE | 从root完整想象→20 MSE | root→20 persistence MSE |
| --- | ---: | ---: | ---: | ---: |
| 0 | .413610665 | .105979064 | .462631624 | 1.154187284 |
| 1 | .354393397 | .145747749 | .126554382 | 1.388804530 |

八次预测均优于各自persistence对照；模型不是完全没有动作预测能力，但保留明显失配。
teacher forcing在slot0改善终态误差，slot1反而略差，不能把负结果全部归为累积滚动误差。
这是构造轨迹的factual evaluator，动作见证仅作为已执行动作条件，不进入planner候选。
它不是独立根泛化、不证明排序可靠或P1通过，也不是同root历史干预因果实验。
真实选中计划的RGB评价已MAIN完整读三文件并38tests通过，尚待本次CUDA产物。
另行实现从真实RGBprefix重编译suffix的最小反馈协议，保持原prefix/root/goal，不拟合任何模型或Q；
仅用于原方案Execute→Observe→Recompile路径的校准，不绕过P1/P4科学门。

### MAIN续报：moving-history四计划真实RGB评价完成，发现一根跨arm预测排序反转

MAIN完整读新RGB source/tests/doc，38tests通过，source SHA
`ed9b80bebb212434d60ddd55d3a3cba01e5cc55306170a0ea6f9a167bb156469`；
prepare SHA `5b5de956f6a6bc45f9569e40d85546f809efd06334f6a8fc5aeaf2fb66947d7a`。
主机CUDA报告`le-wm-main/results/p1_constructive_moving_history_rgb_cuda_20261027_v1_cuda_main01/report.json`，
SHA `7ec03c0a6a6e8411ff0dd0115044a872d099106edc95e4529b97c38c2b9b8f71`。
8encoder/8projector、0新predict/AD/physics，errors=[]、模型未变；
MAIN核10artifact SHA，从8个原float32 NPY独立复算actual/predicted meanQ与预测对真实MSE，差0。
MAIN首次只读复算误用latent.path，按已声明receipt的相对file字段修正后通过；
这是核验命令字段误用，不是producer错误，也未重跑神经实验或改封存输出。

| slot | arm | 真实prefix Q | 真实terminal Q | 模型terminal Q | terminal预测对真实MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | GN | 1.061724386 | .482174596 | .600329583 | .103353289 |
| 0 | D2P | .225843386 | .182658563 | .654757470 | .333385971 |
| 1 | GN | 1.546443848 | 1.374904060 | 1.200545247 | .071256910 |
| 1 | D2P | .262040134 | .198951944 | .166066805 | .166581275 |

slot0模型terminal预测GN优于D2P，实际Q及四维XY误差均D2P更好，构成固定这两arm间的排序反转；
slot1两者排序一致。D2P两根实际Q/四维终距更好仍均原生失败，不能将latent改善写作成功。
四branch的prefix真实RGB已封存，可作为新观测反馈suffix编译输入；旧prefix动作、root、goal保持不变，
不对旧report回填，不根据真实terminal重选prefix，不把这里四branch当独立新样本。

评价合同特别澄清：本批直接沿封存native `env.eval_state`，要求agent+object四维XY误差<20且角差<π/9。
历史2026-10-26的object-pose evaluator只用state[2:5]，是不同评价合同；
不能把历史“物理仅约束物体”的文字推广到本批native成功，也不能互换成功数。
native state_dist还含agent velocity且角差代码有既知wrap边界，应分别报告真实native与wrapped诊断，不改原门。

MAIN联合回归：moving encoder、planner、physical、factual fidelity、selected RGB五模块
合计204tests通过（27.33s），仅RAM/解析合同测试，不替代上述HOST CUDA/真实物理证据。
磁盘实查仍约61GiB可用，无本轮删除/下载/权重复制。新观测后缀编译/真实执行器和固定
archive action-time replay探针正在实现，均尚未prepare或实际运行，不计为新增成功或准入。

### MAIN续报：观测后重编译后缀的四分支CUDA与真实执行完成

用户明确代码审计不单开智能体：MAIN负责源码/接口审读、测试、准入、主机执行与独立复算；
agent只交付限定实现。没有训练、拟合Q、CPU神经回退或修改旧封存。
MAIN联合测试104项通过。新compiler source SHA
`3e72cef9375c5fa5b010593e2285e8eeb8d6b51b71310b70663b7f7406a7eb56`；
prepare SHA `1610eae470b0277429b829eb9827731adbe3f9cde77060338682ff7a3ac58c66`。
prepare实际702362字节，四显式Q矩阵需要2MiB工程上限；不改变科学协议，run仍8MiB。
主机CUDA报告`le-wm-main/results/moving_history_observed_suffix_cuda_20261027_v1_cuda_main01/report.json`，
SHA `79f07b28722991ce40a01cdfa2c40fa0e51f149803db00e1617d1ccecb0966fc`。
20 predict/AE/pred_proj、4 Jacobian、4 SVD、768数学VJP，errors=[]、冻结模型未变。
成功own consume原样封存，不由MAIN伪造status/blind；218 artifact哈希已核。
MAIN从保存J/nominal独立重解bounded ridge，params/raw及所有meanQ数值差0。
新后缀固定无条件派发，旧suffix仅诊断：slot0 GN/D2P预测Q .604785→.490136 / .199897→.060671；
slot1 GN/D2P 1.284849→1.330095 / .220835→.260056，不事后挑选预测改善支。

新physical source SHA `07215161b9ea620efe83ff7ca7ae2838d2f27b40ca16f5cc3aa969626813b2ce`；
prepare SHA `5bcd47b14a02a07162640349b26d315e11b7ee7b016628556eef42162f760972`。
报告`le-wm-main/results/moving_history_observed_suffix_physical_20261027_v1_main01/report.json`，
SHA `3376eda7d0a88b31e8fb56a5f3b2cc19ef4854ef5b12f35ae196da0744c71f0e`。
四full10、prefix_exact/prefix_goal_exact全部true、errors=[]、0native成功；
80public/816physics/88render/8reset/16setter/8goalrender/4goalbind，旧control不额外重跑。
MAIN核2426事件哈希链、88PNG文件/像素哈希、68原prefix帧完整state/body逐位相同、
40条实际plan命令与成功consume一致、原native距离与wrapped成功判据算术完全一致。

| slot | arm | 四维XY误差 | wrapped角误差 | native距离变化（新−旧） |
| --- | --- | ---: | ---: | ---: |
| 0 | GN | 98.214253 | .083553 | +96.311668 |
| 0 | D2P | 20.245456 | .060849 | −30.378485 |
| 1 | GN | 201.681217 | .453206 | +13.929889 |
| 1 | D2P | 45.911567 | .292395 | −27.146670 |

D2P这两支native距离改善但仍失败；native距离含速度，不等同四维XY、物体误差或latentQ。
slot0 D2P接近但未达到原<20门，不调阈值、不计成功。只能说明固定已知根反馈校准已可执行，
不能声称独立收益、非贪心能力机制或P1/P4通过，也不能把未成功解释为理论否定。
新terminal RGB4适配由实现agent推进；archive native alignment探针已交付，MAIN尚待审读/运行。

### MAIN续报：archive/native探针完成，发现目标overlay的明确协议差异

MAIN亲自审读完整source/tests/doc，真实只读build捕获两项工程问题：
1. 严格字段集合遗漏实际H5的proprio float32[2336736,4]；
2. 未注册已安装hdf5plugin，读取压缩pixels时报OSError。
二者均在prepare前修复；新增proprio缺失/shape/dtype/额外字段回归，20tests与7subtests通过。
没有安装/下载/改H5；codec入口源码绑定，proprio只绑schema、不读取或用于控制。
新probe source SHA `62f8fcbffacbc14cdf3641654eb2c4adb9fcadea1c41a6e762a1a0c79034d172`；
prepare SHA `899485c8ae93827ef02100f64429dd293422f617055e81c37dc1f1859b6db0d7`。
主机物理报告`le-wm-main/results/p1_archive_native_alignment_probe_20261027_v1_main01/report.json`，
SHA `affb90a6e67c096623b62ac1d4a1f9f2e11ceaa5d5fe8436f365d39536de2e05`。
固定episode1/6229×lag0/1全部full20、errors=[]；84PNG/80public/816physics/8reset/
16state setter/8goal setter/8goal render；20固定checkpoint全部保留，不选择最佳lag作为准入。
MAIN核2338事件哈希链、84PNG文件/像素SHA、80条实际raw bits、全部20checkpoint算术，均通过。

两episode初态state数值完全一致但RGB不一致；lag0各checkpoint state最大绝对误差
分别<=4.95910645e-5 / <=3.05175781e-5，lag1则分别达25.28173 /35.59590。
这支持本批原archive动作顺序可真实重建，不支持擅自整体错移一帧；不推广为全部archive因果证明。
lag0 RGB MAE仍约1.595–1.722 /2.236–2.294（0..255尺度）。
MAIN直接并排查看相同state0，archive存在绿色目标轮廓，native with_targetFalse没有。
按预先非选择性int16颜色诊断G>R+20且G>B+20，两episode archive0绿色像素988/1355，native均0；
archive bbox集中y94..153、x70..130附近。初次临时计数误用uint8加20环绕，已当场改int16重算，
不使用错误计数。其他RGB差异可能含压缩/边缘，不能把全部MAE归因overlay。
checkpoint训练成员/collector完整配置仍未知，因此这里只能确认archive/live可见协议差异，
不是已证明checkpoint训练分布或已修复模型排序/原P1。
新有界off→on→off同body渲染对照已分派实现，MAIN独占审计/准入/执行，
不得改旧sealed实验为with_targetTrue或把新增可见目标当作原合同效果。

### MAIN续报：反馈新终态RGB4已主机CUDA完成，配对后缀方向3/4仍失配

MAIN完整审读新RGB adapter source/tests/doc，35tests通过，真实只读build175877字节通过；
source SHA `52979d3618626c3a23da2d18cd8affe814cac4c11d9d5f10c8625c6e9746799b`；
prepare SHA `7c6c4377a66760853b6a07ac578af1c1ea8081c166cab3bdc8a1e2bf64a89277`。
主机CUDA报告`le-wm-main/results/moving_history_observed_suffix_rgb_cuda_20261027_v1_cuda_main01/report.json`，
SHA `3c3e2f66b48bf4cd8c427d318b0a456e2498c5499937148c6a8a1ca06e547caf`。
只新增4encoder/4projector，0predict/AD/physics，原prefix不重编码；errors=[]、冻结模型未变。
MAIN核6artifact SHA及4真实float32 NPY，独立复算actualQ/predictedQ/MSE与原封存目标精确一致。
配对数值另存`le-wm-main/results/moving_history_observed_suffix_paired_analysis_main01.json`，
比较的模型old/new都来自同一实际prefix条件预测，不混用旧open-loop预测：

| slot | arm | 条件modelQ旧→新 | 真实Q旧→新 | model/真实Q方向反转 | 新预测MSE |
| --- | --- | ---: | ---: | --- | ---: |
| 0 | GN | .604785→.490136 | .482175→.565551 | 是 | .150446 |
| 0 | D2P | .199897→.060671 | .182659→.039441 | 否 | .050770 |
| 1 | GN | 1.284849→1.330095 | 1.374904→1.332705 | 是 | .019559 |
| 1 | D2P | .220835→.260056 | .198952→.129025 | 是 | .090498 |

两根跨arm终态排序均模型/真实D2P较优，但这不等于同arm旧/新suffix局部排序可靠；后者3/4反转。
slot1 GN真实latentQ改善而native距离恶化，度量不一致仍存在。0/4native成功不改。
这些known-root reused calibration不能当独立确认，P1/P4仍false，也未排除最新overlay协议线索。
MAIN批准下一轮固定2episode原lag0真实20步回放，在0/5/10/15/20同body off→on→off、
新off对旧checkpoint file/pixel/state/body逐位核对。物理预计40public/408physics/30PNG；
独立实现agent准备三condition（archive/off/on）的冻结CUDA factual保真消费者，MAIN负责审计和运行。
均尚无新实际产物，不把实现或已派发任务当作完成证据。

MAIN联合回归另覆盖九模块（moving encoder/planner/physical/fidelity/RGB、observed suffix
compiler/physical/RGB、archive alignment），367tests及7subtests通过（37.31s）。
这是工程回归，不补P1科学证据。MAIN核native env默认goal位姿[256,256,π/4]及
with_target/render_goal分支；固定绿色overlay并非当前构造任务的物理成功目标。
对照只切显示，不让旧任务目标跟随该轮廓，不能将其引入写成原合同“修复成功”。

### MAIN续报：同body目标轮廓显示对照已实际完成，解释大部分archive像素差异

MAIN完整审读full20 producer三文件，19tests通过；审读中要求修正合法native早停被fullcap
census误标engineering_failed的问题，新增早停回归。它没有解释旧full20结果，只修失败分类。
最终source SHA `9845afec0c0ae3cf5636606f3bbd7e2e75734dbd6b7c2648eae25e46fc8f9586`；
prepare SHA `1bcc1a9e4002cc2887ac1796c62f543a8fb24d6115b2d66de47b8ce6df7a4fb6`，16038字节。
主机物理报告`le-wm-main/results/p1_archive_target_overlay_probe_20261027_v1_main01/report.json`，
SHA `184809d72b4e4a6194cb9d7591b26f4a2fe09d423a826ab02c1fcb5cf340c430`。
两full20、30PNG、40public/408physics/4reset/8state setter/4goal setter/4goal render，errors=[]。
固定原episode1/6229的lag0动作，不改goal_pose/goalstate/相机/颜色/动作；同body五checkpoint分别
off→on→off。MAIN核1074事件哈希链、30PNG文件/pixel SHA、40实际float32 raw命令，
全部10triplet前后完整snapshot/goal/renderer一致、off replay逐位一致，两次off均与旧lag0封存一致。
独立复算另存`le-wm-main/results/p1_archive_target_overlay_physical_analysis_main01.json`。

| episode | checkpoint | off RGB MAE | on RGB MAE |
| --- | --- | ---: | ---: |
| 1 | 0/5/10/15/20 | 1.71846 /1.72206 /1.67686 /1.66645 /1.59501 | .41570 /.41846 /.37088 /.36011 /.38021 |
| 6229 | 0/5/10/15/20 | 2.29403 /2.26785 /2.23555 /2.23723 /2.29203 | .46510 /.43484 /.40446 /.40624 /.46112 |

十checkpoint平均MAE从1.970554降至.411711（约79.1%），剩余非零差异仍未归因。
这是同物理状态显示干预对像素失配的证据，不是checkpoint训练配置已认证、神经预测修复、
规划排序修复或P1成立。没有把旧target-free科学合同改名为target-on正结果。
三条件archive/nativeoff/nativeon的matched factual冻结CUDA评价器继续实现，固定30编码/24预测，
尚未prepare/运行；只有该实际结果才能判断overlay如何影响这两条轨迹的预测保真。

### MAIN续报：matched factual主机CUDA完成，显示协议显著影响冻结预测

上述“尚未运行”状态已由本节实际产物替代。MAIN审读完整三文件，prepare前捕获
`sealed.core`不存在的运行期接线错误，修为`enc.core.hash_value`并补外部SHA拒绝回归；
最终30tests通过。最终source SHA `f536cf5fe9baa4c652b17888f6ff0112bccc5aa241c567581a034b5ec8ccc31c`；
prepare SHA `f45d89d60e6da7e6ef97b1bc33ae8c3d1f335ddc92405a6e73873fde5ee2f25f`。
主机CUDA报告`le-wm-main/results/p1_archive_target_overlay_fidelity_cuda_20261027_v1_cuda_main01/report.json`，
SHA `d8cbd4b6e8ff1497421e6cdaf8644f4b39ed2cd2b695bb10cd15b57fbae121f0`。
实际30encoder/30projector、24predict及对应action_encoder/predictor/pred_proj，
0AD/physics/training，errors=[]、模型tensorhash前后一致。原生encoder输出保留[1,257,192]。

| 固定预测模式 | archive MSE | nativeoff MSE | nativeon MSE | on相对off下降 |
| --- | ---: | ---: | ---: | ---: |
| H1→15 | .00337773619 | .02360527779 | .00806385932 | 65.84% |
| fresh H2→20 | .00827052067 | .03901472734 | .00963172965 | 75.31% |
| teacher-forced→20 | .00829561897 | .06876958697 | .02032198522 | 70.45% |

固定episode1/6229、lag0、同condition label；全部六个episode×mode的on MSE低于off。
独立H1与freshH2首跳逐位相同，不重复计作独立样本。on仍比archive误差大，残差原因未知。
MAIN独立核522artifact、516数组的压缩/NPY/raw三层SHA与shape/dtype、30原RGB、
全部ImageNet预处理（最大差0）、24真实raw及归一化packet、teacher/imagined context接线、
模块返回/156 ledger/577journal事件，以及24MSE和persistence分数、10组off/on latent距离，均通过。
独立数值结果保存`le-wm-main/results/p1_archive_target_overlay_fidelity_independent_analysis_main01.json`。

这证实观测显示协议对这两条轨迹预测保真的贡献，不认证checkpoint训练来源，
不证明构造任务的规划排序、非贪心机制或独立泛化，P1仍未通过。
下一步固定旧moving根/动作/任务，仅增加同body可见轮廓的真实history/goal及terminal配对，
测试排序能否修复。新schema明确target-on合同，不覆盖target-free旧负结果。
MAIN独占审计/准入/真实运行/独立复算；两个实现agent分别负责新物理producer与CUDA consumer，
不设置专门审计agent、不训练、不降级CPU神经实验。

MAIN另做事后描述性尺度核查，避免把latent整体收缩误当预测修复：
每mode两episode的平均预测MSE/平均last-input persistence MSE，off→on分别
.168616→.080056、.196215→.061123、.462466→.137003；全部六个episode×mode的
单根该比值也下降。freshH2的last-input为预测15，另用固定origin10作持久性基线，
其比值.084825→.023301，同样下降。此计算不新增神经调用，不构成独立确认或预注册检验。
完整数值`le-wm-main/results/p1_archive_target_overlay_persistence_normalized_main01.json`。
三新模块联合回归84tests通过（3.82s），是工程验证，不提升科学准入状态。

### MAIN续报：moving同任务配对source与GN/D2P真实回放均完成

原六个子智能体容量已全部使用；额外spawn被`agent thread limit reached`拒绝。
三路推进paired source/plans/CUDA，另三路实现原target-free moving的CEM/nominalMPPI、
gradient、kinodynamic基线，互不改旧seal。交付空闲worker已续接统一baseline物理/RGB消费者，
并非新增超过容量的agent。MAIN继续独占审计、实际主机执行与独立复算。

MAIN读source完整三文件，发现on15缺口并要求prepare前补齐，固定186render合同后
23tests通过、实际build/verify155bindings通过。source SHA
`768a45d127ad01a0e3ee8b55a6c7c535c92134211af10914e3e99bb4674e472b`；
prepare SHA `88957b9d251595e5e35ed46ba4a9aa003eb937bbfff0f58ae12540418dcb39a1`。
真实主机报告`le-wm-main/results/p1_moving_target_overlay_source_20261027_v1_main01/report.json`，
SHA `85ca1df3de910628bfd6c5bd1a0a482f61880a2ee76bfd404732a77dd8c7666a`。
6full20/186PNG、120public/1224physics/12reset/24state setter/12goalrender/
16goalsetter/4goalbind，errors=[]。MAIN独立核3520事件哈希链、186PNG、120literal float32
命令、30off/on/off完整invariant、4严格盲包；off全与旧source逐位一致，
on goal确为新construction真实terminal20图像，replay0/1精确复现。独立结果
`le-wm-main/results/p1_moving_target_overlay_source_independent_main01.json`。

MAIN另完整读GN/D2P plans producer三文件，47tests通过，实际build118bindings通过。
source SHA `ec5c9d383b09f9db2475710645f680946c6ebabf966d8785405e96751a2f200b`；
prepare SHA `72c10ab251ab426c2e4e6d1b2a7ef8a19d697075c4e2e2d4b8bd76314885d048`。
真实报告`le-wm-main/results/p1_moving_target_overlay_plans_physical_20261027_v1_main01/report.json`，
SHA `d40c32e8c59df7c5552937cf76ee43525ae2aae0b595340cd325662b8ea686c0`。
4full10/112PNG/12triplet，80public/816physics/8reset/16setter/8goalrender/4goalbind，errors=[]。
MAIN独立核2434事件链、全部112PNG/80实际命令、88原off逐位一致、12triplet完整invariant，
终态state/metrics均等旧封存，仍0/4native success。独立结果
`le-wm-main/results/p1_moving_target_overlay_plans_independent_main01.json`。

MAIN还在未封存CUDA consumer发现真实事件接口问题：producer先emit observation再加
old_off_match字段，故journal该字段在guard事件（checkpoint report则有完整字段）。
消费者必须按phase/step/mode/png join实际guard，不能从observation直接取值或将完整checkpoint
与早先observation dict强行判等；已交实现者prepare前修正，不改旧producer seal。
这不是科学负结果。本轮两个物理producer均0神经调用；新CUDA预测/排序评价尚未完成，
known-root配对仍非独立确认，P1及完整C1--C3/三实验/P0--P5闭环均未完成。

### MAIN续报：配对CUDA完成，三臂基线进入真实回放

主线程实际发现并修复第二个consumer接线错误：plans中的compiler_report_SHA是
命令handoff的52c475…，不是原CUDA报告cd843…；此前混用导致phaseB material拒绝。
新增反例回归，65tests通过；保留旧prepare01，新建prepare02，不修改旧producer。
这是工程失败，不是科学负结果。

主机CUDA两个phase均完成，模型哈希未变、errors=[]：

- factual报告SHA `cff57afe05e44a720151ed912a3345f35c5c98936310fc46bbc9c6b563474605`，20编码/16预测。
- fixed-plans报告SHA `618379d20fe6fa0575a22ae1e1675756c9aec1d6ecf0c8a6fab5536c32aee40c`，32编码/24预测。

均在`le-wm-main/results/p1_moving_target_overlay_fidelity_cuda_20261027_v1_{factual,plans}_cuda_main02/`。
MAIN使用不导入producer或模型的NumPy脚本独立核896artifact hash、872数组三层hash/shape/dtype、
40误差、16目标Q、8排序及token/context/真实模块输入输出绑定；结果保存
`le-wm-main/results/p1_moving_overlay_independent_recomputation_main02.json`。
此核验不冒充完整独立module chronology审计。

原动作固定时，slot0开放H2的GN/D2P排序在off/on均反转；两条件teacher排序均一致；
slot1两模式两条件均一致。因此overlay未修复该开放H2排序，不能外推archive上的改善。
factual独立H2末态MSE on/off变化slot0 +7.23%、slot1 -53.61%，并非一致改善。
提示继续核验多步滚动误差/反馈接线，不证明理论不成立，也不宣称根因已经穷尽。

MAIN完整审读并运行sampling29tests，实际CUDA完成两根CEM+nominalMPPI：
report SHA `9ac889d6be76bf3241f85c9bfcac13936784dcf80171b3fe48abf5c9cfcd319c`，
412predict/AE、ownconsume4rows真实通过。此前gradient已完成272predict/AE与64reverse，
report SHA `96c140caee14bba43bf766d981c21be1724a36c0047fde4abcc2892c24bf5b7d`。
不将模型Q当物理效果；nominalMPPI是本地参考核、非官方完整复现，三臂预算不同。

物理consumer三文件MAIN完整审读、39tests通过，预先固定gradient/CEM/nominalMPPI三臂×两根，
不等待kino/GRASP且不按成本选臂。manifest SHA
`0242766b513ed7959e142daef827ebe9162193164791a56979380f23960c692c`；prepare SHA
`2e7304f11420ed65b6f3f039dd49522fee042e30bf6e97c82db791addd3fa50e`。
真实物理执行及后续RGB各以最终报告判定，不能仅凭prepare/启动认定完成。

六子智能体循环复用：物理consumer、RGBconsumer、无组合SRA、movingGRASP、结果汇总、
预算调度提案；主线程承担实际代码审计、准入、CUDA/物理与独立复算。并发容量仍为六，不宣称新增第七个。

三臂物理实际已完成（不是仅启动）：report SHA
`6a9f4cf602a328cf5cb0fbe82296fc989a109a6859287c2efb703bebf44511db`，
6full10、132PNG、120public/1224physics/12reset/24setter/12goalrender/6goalbind，errors=[]，
native success **0/6**。MAIN独立核3530事件链、132PNG、60实际selected commands。
gradient独立复算66候选成本、64tanh-chain与投影更新，赢家iteration32/30，成本
.2613191018/.0720113415；仅验证保存算术，不冒充重新证明AD导数正确性。
保存`le-wm-main/results/p1_moving_baselines_independent_main01.json`。

RGB consumer MAIN完整审读、42tests通过，实际prepare179835B，主机CUDA12encoder/12projector
完成，report SHA `ab549f1d1a75c88b64a1c8c8697b004f8e6bf84d603b6e350406cf439fa305cd`，
目录`le-wm-main/results/p1_moving_baseline_rgb_cuda_20261027_v1_cuda_main01/`。
errors=[]且模型未变；MAIN独立核14artifact/12native192数组和全部12指标，保存
`le-wm-main/results/p1_moving_baseline_rgb_independent_main01.json`。

| 预声明基线 | slot0真实终态Q | slot1真实终态Q | 原生成功 |
| --- | ---: | ---: | ---: |
| gradient | .454879964 | .046495811 | 0/2 |
| CEM | .364320541 | .125224110 | 0/2 |
| 本地nominalMPPI | 1.106068765 | 1.350350558 | 0/2 |

Q越小越好，原rootQ为1.154187284/1.388804530；以上是旧两根描述性结果，不是新独立样本、
等总预算或P1机制通过。部分物理位置明显靠近仍未满足agent+objectXY/angle联合原生标准，
不能把object-position单项替换成功合同。prediction字段保持unknown/null，未重新预测。

新一轮worker已续接无组合SRA实现、physical/RGB comparison v2、新根协议、
step5真观测反馈协议、P3预算预注册和RGB纯saved接口交接。旧模块继续冻结。

## 满槽轮转后新增实际主机CUDA证据（2026-10-27）

MAIN继续按experiment-code的固定协议/真实日志执行，不采用其通用“成功率不得零”要求；native零成功原样保留。六worker已接入连续反馈core/driver、预算scheduler、extended baseline physical/RGB、comparison v2，具体见PARALLEL_DISPATCH最新节。实现、测试、运行与汇总重叠，审计仍由MAIN承担。

- GRASP moving-H3 bounded本地适配：26项测试通过，实际双根CUDA完整3604predict/AE、1800scalar backward、模型不变、两计划已封存；own consume已通过。报告 `le-wm-main/results/p1_moving_grasp_baseline_cuda_20261027_v1_cuda_main01/report.json`，SHA `7593940c1503416f42ce87a5d8135e8512763121e027c0ef98dbe614bfb31f91`。H3指观测历史，未来仍两包10动作；非官方数值等价、非等预算，物理效果尚缺。
- 无组合joint-H2谱bank后逐packet投影：MAIN完整审读新adapter及测试，24项测试通过；实际双根CUDA完整176predict/AE、2J、384数学VJP、4SVD、2bounded solve，errors=[]、模型不变。报告 `le-wm-main/results/p1_moving_no_composition_baseline_cuda_20261027_v1_cuda_main01/report.json`，SHA `ec39956ca0c90cc5dbf955b1a65bf300b68d166627a748a82314a2176a481c34`；own consume运行中。保留原未归一化ridge damping .002，支持域映射与原joint-ball分布区别明确记录。
- Kinodynamic：168predict/AE完整搜索后，两分支真实物理完整执行，native成功0/2。物理report SHA `2db7370214bf62810c21ad8778c1d58bc397dbe8f08c9f5ef1db435468abc655`。四张真实RGB实际HOST CUDA编码4encoder/4projector完成，report SHA `71dafeeebd1d2bd2567028e65800f4f27f479c72b066da3d2ace6a35ef8e7869`。MAIN不导入实验模块独立核对6个artifact哈希、4个float32 native192数组及4组Q算术，真实终态Q slot0 `.5584198381473228`、slot1 `.29292652750561465`；预测字段unknown/null不填零。只复算保存数组，不冒称独立神经/物理重跑。

全部为同两根known-root calibration，不能增加独立样本数；模型内Q/真实RGB Q下降不是native任务成功，更不是非贪心改变未来能力的P1机制通过。GRASP与无组合新计划的真实物理/RGB链由独占新consumer继续实现，不改旧白名单/封存。

后续同轮核验：无组合谱基线own consume已终结通过，完整重建两根40槽谱bank/选择/上下文与重放链；不是独立神经重跑。`git diff --check`通过。comparison v2交付后worker立即接v3补两个新provider，v2不回填改封存。

## 连续推进：真实综合表与GRASP/无组合物理执行已补齐

四算法comparison v2经MAIN完整审读source/tests、40项RAM测试通过后，已实际saved-only汇总：
`le-wm-main/results/p1_moving_baseline_comparison_20261027_v2_main01/comparison.json` SHA
`a0fb977445b79d72ebb062a78c454debabfa0e6560eb0f5957e7a8676f85424c`；
同目录中文表SHA `7fe2cc34153ba6b25a18151f2882c0199c4e6fcbe4715c3b097e11d9044ae987`。
显式manifest SHA `e2dcb6023a8c9f93c7e244febf5fa68bd2a99b1e7efb3aaf6c6d586a8f1a6a31`。
MAIN另用不导入实验模块的脚本复核8个实际物理分支、16份float32 latent、16项双根均值和预测调用求和，全部通过。
综合状态recorded仅指数据链完整，P1仍false；各算法native0/2不能解释为同预算失败，也不证明理论失败。

GRASP/no-composition新physical consumer已冻结（source SHA `a0f7348ebe8c6e061b5d63a1ba4a8ecb064a26e7f7a1ef4be57fe8b07270d799`，tests `91587a9a8eaf0b7ce738dc4407b6b318f9d7d3aca37afc207a16e889641154a1`，doc `73ae46ab471e261d5f9535e28c583e34dd50c5d1bbf17e479df12ad8dec6472d`）；MAIN44测试通过，确认固定原两根×两臂而非运行后挑选。
事前selection SHA `431bebaa7f95439b40583b1476ad78209023324a5c6f10a963b2101d04aae351`；
prepare44158B，SHA `9819cd6857e043492d27a239728fb3eee6d6ed5542546796467bf0ae83425fd7`，verify通过。
随后实际HOST物理四分支完整结束，report
`le-wm-main/results/p1_moving_extended_baseline_physical_20261027_v1_main01/report.json` SHA
`f05ce7e2d773bb046c6c8ba905c9e86c367c7e19ad09174d4509e1823a297038`。
errors=[]、full10=4、native成功0/4；80public、816physics、88PNG、8reset、16setter、8goal-render、4goal-bind、4construct/close均实际attempt/return一致。
MAIN独立复核2370项journal hashchain、88PNG文件与像素hash、40条literal选中动作和全部resource计数通过。
构造半途失败未覆盖的内部工作不能由hook计数0解释为真实0；本次所有构造成功且符合封存原生无reset/step/render构造合同。
这两个新基线的8图RGB编码正在接新consumer，不挪用旧endpoint或修改旧白名单。

连续反馈窄core经MAIN完整审读及111项联合回归通过。driver尚在联调：MAIN发现过run入口缺runtime、假定suffix提供encode的实际接线缺口，已交由driver实现方补齐同进程加载/独立视觉Handler及入口测试；这些是代码问题，不计入科学负结果。只有新driver完整审查、封存、真实同episode执行及因果日志通过后，才能讨论R−I/R−O的真实反馈效应。

## 本轮新增终结证据：连续反馈、六基线、预算调度

以下覆盖上节的“正在接/尚在联调”，不修改旧封存文件。

1. **六基线完整模型→物理→RGB已汇总**。extended RGB主机CUDA完成8encoder/8projector，report SHA `8e72ca50d43a5460330dafff8b07568e204ffb43fdd3e0b53d8b9d6906b929a9`；MAIN独立核18artifact、56event、8PNG/8native192和Q/pixel算术。comparison v3完整审读、56tests后实际assemble：`le-wm-main/results/p1_moving_baseline_comparison_20261027_v3_main01/comparison.json`，SHA `96cf27b143bb1a839edf7bc1e4de02f4a97e433c6475c4c731c78af76300d32e`；manifest SHA `388143639950875317be2f5d25ebaeb8b6b9c1c409290c92763dc10ceef600d5`。12model/12physical/24RGB全部接齐，原生成功0/12。MAIN另以无实验模块导入脚本重核12物理selection、24NPY和24双根均值。GRASP/无组合谱终态真实Q均值分别 `.236275930840675`/`.4437224331512617`，不等预算、不作独立确认、不排名为正式科学结论。

2. **12分支真正连续反馈已实际完成**。driver source `6742a1b35102e433d7772e4a208b4f97b37b76f815b991cfc0e472763754ace9`，MAIN逐字审读三文件、22项RAM测试通过；prepare SHA `bc3fbc91d7a19ae256b24db22ad05b2526fcbe454e3b55d904136eba2f8fdff5`。主机CUDA+physics实际report `le-wm-main/results/p1_moving_live_feedback_physical_20261027_v1_main01/report.json`，SHA `990843e8bba15955db5163b9b5d77c1f8eecaa1870d3745b49d46cc22ca895e3`：errors=[]、12full10、native0/12、24encoder/projector、40predict/AE/predictor/pred_proj、8ADforward/8J/8SVD/1536数学VJP、240public/2448physics/264PNG。全job80.363665s、execution66.711600s，模型原hash未变，12条原prefix lineage均精确匹配。

   MAIN不导入实验模块独立核8334事件hashchain/operation attempt-return、264PNG与24NPY、每episode先observe5→encode5→seal→dispatch6、prefix之后无reset/setter/goal替换、literal5suffix与实际执行一致。同两已知根四配对层，Q10的R−I依次 `−.0683736287, −.1432171055, −.0453253296, −.0699266200`；R−O第一层仍 `+.0833765534`，其余为负。真实观测优于imagined重编译的编码距离不等于原生任务成功、非贪心或能力改变；不把4层当4独立任务。P1/P4仍false。

3. **预算调度已实际完成520次HOST CUDA预测和8分支物理**。scheduler source `f4f4d5e013e3fc3aac896da58faaafadcf5e2d4e53f5f2f7b61203707e09ff9d`，MAIN全文审读source/tests/doc、51tests通过；prepare SHA `330bb32be9cc3ccf330835b94e336fff9aad057a6b85ce468678d65397369aaf`，report `le-wm-main/results/moving_history_budget_scheduler_cuda_20261027_v1_cuda_main01/report.json` SHA `710892008bbd47aa188a9b9391972976d3a94d2a0067e46911ff37f05be8369b`。520predict/AE/predictor/pred_proj、2J/384逐行grad与数学方向/2SVD，errors=[]且模型不变。MAIN独立SVD/配额/全部candidate cost/argmin/replay复算：权重162.32405169996156与156.2867146609139，D与交换权重的S经预声明half-up取整均为[10,10]，与U/N相同；赢家各根index1/2。因此本panel**未形成分配干预差异**，不能据四者相同宣称谱调度无效，也不按结果事后放大权重或调配额。

   physical新consumer MAIN全文审读、70tests通过，prepare `bddd73792da6ecabcc669e75d9c466940f763571990498f3363375336c7d6344`。真实8分支report `le-wm-main/results/moving_budget_scheduler_physical_20261027_v1_main01/report.json` SHA `c9169b685e4e7f7d79ea2101aa81ad55b95593f96d6158ed8ff0815de136bd70`：8full10/0native、176PNG/160public/1632physics、8construct/close，errors=[]。即使命令相同也未去重/退款；后续新RGB evaluator由独立worker实现，未用旧endpoint填新结果。

4. **真实捕获的代码问题另列**：feedback analysis v1的27项5维RAM测试通过，但MAIN实际saved运行发现它硬编码state/goal为5维，真实native checkpoint均7维（含速度），导致shape错误。无分析产物发布；该失败属于接口/测试覆盖错误，不是理论负结果。原native距离使用全7维norm，success使用前4维XY与angle，禁止截5维冒充原生度量。v1保留封存，修复使用新v2及7维回归，不改原driver或实验结果。

5. 新根排重实现48tests仅工程交付，现有六显式authority smoke仍970unknown，不能认证freshness。两worker并行补历史authority inventory/显式身份桥；不按结果选根、不绕过unknown发起独立确认。原研究goal、P1及完整C3依旧未完成。

## 六工作位持续轮转后的实际新增结果

- analysis v2修复已完成：MAIN全文核验、32tests通过，实际读取原live报告成功，结果 `le-wm-main/results/p1_moving_live_feedback_analysis_20261027_v2_main01.json` SHA `5df7dd90b3da010d1bafbb66198334272ec6470b7b0f7bc62991e4b0dae0b5d8`。保留全7维native distance，没有修改物理结果或成功判据。
- 六基线读者表实际生成至 `le-wm-main/results/p1_moving_baseline_reader_table_20261027_v1_main01/`，31tests通过，12行；reader_rows.csv SHA `254b7be5137f82b5239944452d6dd86be8d25525f87e1813b5c8eb7d728c791f`。不同预算/本地适配/两已知根限制明确保留，不用作者跨协议reported值填本地缺项。
- scheduler saved comparison：MAIN完整代码核验、38tests通过后实际汇总8物理分支，报告 `le-wm-main/results/moving_budget_scheduler_comparison_20261027_v1_main01/report.json` SHA `1eb2c51d6d4459b4aa6723fafed40ca8034729aa4f27d4359c28964125840b64`。该封存v1中的RGB仍missing，随后新编码独立衔接，不回填修改旧报告。
- scheduler RGB：MAIN完整审读三文件、56tests通过，prepare SHA `cfa1326c7d16f9617e999a66153c72545792cae82783c82a26b0de087f249672`。实际HOST CUDA结束，报告 `le-wm-main/results/moving_budget_scheduler_rgb_cuda_20261027_v1_cuda_main01/report.json` SHA `08b96eb2f861be9a84e96738e030aa423ac4a0e4b6541f95355bd01b2d2554e7`：16encoder/16projector、errors=[]、模型未变，34artifact+report。ownconsume成功；MAIN不导入实验模块的独立脚本再核34hash、112event、16PNG/16float32 native192和全部Q/pixel算术。四策略slot0终态Q均 `.4821745958776284`，slot1均 `.19895194420946052`，符合相同命令的真实重执行；不是缺dispatch，不增加独立样本，不构成P3理论负结果。
- scheduler物理journal独立复核已通过4738事件、176PNG和80条literal命令。新根排重v1经MAIN完整审读与48tests通过，但这不是完整历史coverage或科学freshness认证。

并行配置：六位worker分别推进配额离散化后续设计、RGB综合adapter、历史来源清单、来源身份桥、作者reported值核对、teacher-only冻结模型轨迹保真度诊断实现；MAIN专注代码审读/主机CUDA/独立复算。teacher诊断不训练、不拟合、不反馈控制，不能替代独立新根P1确认。

历史清单已实际交付618个path/SHA，960条继承引用对应480个不同路径；包含8次既有calibration尝试，不按选中结果删根。MAIN实际执行8项显式request：`le-wm-main/results/moving_root_exclusion_manifest_20261027_v1_inventory8_main01.json` SHA `a17bc732c9a021a4ca81a7e4da573c752ad1f66eafe08b1a3dad2b64464d7060`，975 unresolved＝960 inherited＋6身份缺项＋2schema投影失败＋3缺family＋4显式范围声明。它不是原970被修复后增加5个新错误，而是扩大显式输入/声明后的不同分母；状态blocked准确保留。后续分别补旧reset上界证据与schema/身份桥，不能靠缩范围清零。清单总体282406682B及618项超旧v1预算，不能擅删106文件伪装完整覆盖。

作者reported数值及原文页表已落在 `docs/D2P_REPORTED_BASELINE_NUMBERS_HANDOFF_20261027.md` 和新增PRIMARY_LOCATORS补充文档；来自GRASP/Hi-LeWM/LeWM/PlanWAM固定版本原文，与本地matched结果分栏，不填造missing。交付后该worker立即续接六基线saved预算矩阵实现。

## Teacher factual实际CUDA与并行槽位恢复

Teacher v1实际prepare在任何神经调用前因357081B语义内容超过256KiB封存上限而失败；这是序列化代码问题，不是模型负结果。保留v1，v2采用完整语义SHA绑定的有界gzip/base64封装（不删证据、不改模型输入，单gzip/CRC/尾随成员与解压上限均验证）。MAIN审读与42tests通过；实际prepare64526B，SHA `b83bccf3b5edba63e59c0c3f4efc4e796fe99bfdd43a9a6096c8eb57f1b9fae6`。

实际主机CUDA报告 `le-wm-main/results/p1_moving_live_feedback_teacher_fidelity_cuda_20261027_v2_cuda_main01/report.json`，SHA `7323347a9e74c8f1751550642503426d07ddce55549ddaef4331f3c362567128`。12条各自真实prefix3+实际suffix5完成12AE/predict/predictor/pred_proj，errors=[]、模型未变；无训练/AD/新encoder/物理。124artifact+report、48ledger、170事件；12预测先封存后评分。I的真实prefix只用于离线factual诊断，不赋予I部署时的观察权限。

MAIN独立脚本复核124文件、170事件、全部12条输入动作/目标/评分，以及4条R的完整投影输出与原compiler new_suffix_replay逐元素完全一致。四配对层R−O预测/实际Q差为：

| 层 | 预测R−O | 真实编码R−O | 方向反转 |
| --- | --- | --- | --- |
| 0 | -0.11464941383908783 | +0.08337655337607058 | 是 |
| 1 | -0.13922638198214798 | -0.14321710547049737 | 否 |
| 2 | +0.045246326523013636 | -0.04219944671806286 | 是 |
| 3 | +0.03922147667041345 | -0.06992661996426389 | 是 |

是3/4反转，不是2/4；已更正此前向解释worker的层2口误。仅两已知校准根，四层不是四独立任务。逐元素等价排除了本次teacher/controller接线漂移解释，但不等于排除全部代码/协议问题，更不证明理论不可行或真实物理收益。P1保持未通过。

其他实际交付：scheduler RGB综合report SHA `e9ec5cc7818ecaa62c4898aec3e3eb6eff7fb3f40321b70eab6f3643267f17d0`，8行16图、所有D对U/S/N的Q差为0（相同配额/命令）；六基线budget_matrix.json SHA `721d978d29516f8ecf1d1056e159736ef231723e6849490a83678bd76baa1ebc`，30tests通过，15对总成本均不等或未建立，未称等预算优胜。

历史resolver实际capture `le-wm-main/results/root_exclusion_authority_resolution_20261027_v1_MAIN01.json` SHA `032ea722f53da7c93db90377d775b75f7ff7a40771a43fdfa38dfd7761e27d00`：480/480来源捕获、6/6原记录physical身份桥成功、errors=[]。614引用无直接identity，8来源16引用缺旧reset上界、4family coverage未获认证仍保留；不是freshness通过。下一步完整catalog接线和旧上界证据追溯并行进行，不删unknown凑通过。

并行状态更正：四旧worker因服务403余额不足停止，已关闭并补派四新实现worker，与原两路合计六工作槽；新派发不冒充已运行完成。主线程实际完成prospective queue51项RAM测试与teacher独立复算；无CPU神经降级、无新鲜根科学结果。最新分工见DISPATCH末节。

### 新并行批次实际交付与MAIN接线

四新worker本批均实际交付了新source/tests/innerdoc，而非仅派发：618项inventory桥、teacher误差分解、六基线前瞻预算合同、P2组合准入合同。MAIN已全文审读inventory桥与teacher分解三文件；预算/P2为实现与测试交付，不预填真实实验。

MAIN实际执行完整inventory618 resolver，新报告 `le-wm-main/results/root_exclusion_authority_resolution_20261027_v1_inventory618_MAIN01.json`，SHA `e2afed12a1d8f788ad16d0223482377b5f54d281e67739766475228b53ac2591`。捕获618文件、344有元数据、199直接身份、123显式输入绑定、6原记录物理映射；285077447B有界读取。614原引用无直接身份与16旧reset引用仍保留，不发freshness。

MAIN审读teacher分解源码/tests/doc，29tests通过后独立实际再生成 `le-wm-main/results/p1_teacher_fidelity_error_decomposition_20261027_v1_saved_MAIN02.json`，与worker结果逐字相同，SHA `483c40c33eb856c45b380e7d0892e555784236d8e7584c198b3881098c16a0e7`。798文件SHA、132NPY数组SHA、12episode与3/4反转均复算。Q偏差分解为 `2<actual-goal,pred-actual>/192 + ||pred-actual||²/192`，残差最大约1.53e-16。当前suffix本身是H1=5raw一步；三个context投影不是三步自由rollout，不能笼统以长滚动累积解释本次反转。

MAIN实际联合测试192pass/1fail：旧比较器断言整个进程未导入torch，被同进程teacher收集时导入污染。单独比较器+inventory桥60pass。保留原封存测试，不将此失败算理论负结果；MAIN新增独立进程并行验证入口，最多4进程/BLAS各1线程，九套原测试448项全部通过、16.06057秒，入口自身6项通过。报告 `le-wm-main/results/moving_metadata_parallel_validation_20261027_v1_MAIN01.json` SHA `04b0b98dac2f502cbbb3714e5449fd2e436ad847d40ca93b5762a8e8d8e42e90`。原单进程联合命令隔离问题仍存在；新入口没有改原断言或实验产物。

完成者已续接身份绑定图闭包、预测度量与物理指标对齐、资源/真实结果Pareto表、P2动作见证图搜索核心；原两路推进条件reset隔离清单和合法packet时域诊断。Poincare的条件隔离清单27项测试与新metadata已交付，MAIN待审读；不能据11条件seed区间追认历史完整coverage。研究goal及P1/P2/P3科学结论仍未完成；本节没有新增神经实验，未以CPU代替CUDA。

## MAIN实际H1/H2定位：36次主机CUDA，非单纯滚动累积

前一goal轮为实际进展（618清单接线、并行回归与teacher误差分解），不是无进展或等待。本轮重读完整原方案715行，仍以C1/C2/C3、三实验族、强基线等资源与原P1机制为完成标准，不将metadata/测试交付替代原目标。

MAIN全文审读新horizon source/tests/doc，32项RAM测试通过。source SHA `8b8e4bdb708b914b3fc50bba9e0b6c3b624cc724396c05d660cc79432879e1ed`，tests `3c2666a150208f0a231896dc76bc585a777ff6ade573b99eebd872e77f9866c4`，doc `ff5c9fa4a6ae9bf9f6ffe6f9fc814e10260a19df169dc10811ed31be7d0890c5`。实际prepare89416B，SHA `c87b35732ebaa3a3f6c74ec2e65ec817d9b15f319e0b687cc3ffa453f03935ec`，verify成功。

真实主机CUDA报告：`le-wm-main/results/p1_moving_teacher_horizon_localization_cuda_20261027_v1_cuda_MAIN01/report.json`，SHA `e319035d9e13a6ae2dc4cbfa5cefb9eb2e6ccd49dbcf35408fda1704f5f34253`。12固定episode各prefix H1、open H2、teacher H1，共36AE/API/predictor/pred_proj完成，144ledger；errors=[]、模型未变，0训练/AD/encoder/physics。整个作业13.35149秒，352artifact+report，482事件，916绑定。无CPU神经降级。

ownconsume成功；MAIN另写不导入实验执行器的标准库/NumPy独立脚本，核全部artifact与916输入文件SHA、事件链/144attempt-return、所有36份完整module输入输出桥接、raw归一化/历史时间顺序、全seal先于评分、每条自身标签及指标。12个teacher H1输出与旧teacher逐位相同，原H1证据没有被新实验替换。独立报告 `le-wm-main/results/p1_moving_teacher_horizon_independent_MAIN01.json`，SHA `371c59c40bb67482257ba210f2ac5d1a71e78c2970a7d9fff8463d3a914130c7`。

| 层 | 真实编码R−O | H1预测R−O | H2预测R−O |
| --- | --- | --- | --- |
| 0 | +0.08337655337607058 | -0.11464941383908783 | -0.09461784016586139 |
| 1 | -0.14321710547049737 | -0.13922638198214798 | +0.04758828414839511 |
| 2 | -0.04219944671806286 | +0.045246326523013636 | +0.07598683498908931 |
| 3 | -0.06992661996426389 | +0.03922147667041345 | +0.10731037241853303 |

H1排序3/4反转，H2变为4/4；固定12自身轨迹均值MSE：prefix H1 `.19308612179218534`、teacher suffix H1 `.07422531920061208`、open H2 `.19168943692672857`。这是两已知根相关分支的描述，不能称独立失败率。H2通常更差但并非逐条单调（e00_R略降）；gap和cross交叉项均独立重算，不将gap称“累积占比”。已有H1反转不能单由滚动累积解释，依然未排除全部代码/训练合同/度量原因，更不推导理论不可行。

MAIN继续检查当前安装的原LeWM rollout实现与手写packet路径：原API支持cached embedding、H3上下文与T4动作token的两步rollout，action_encoder一次计算全序列，与手写逐packet调用的计量不同。已交Jason实现新原API等价诊断，必须复用真实RGB/已付cached embedding且保存完整module账，不造dummy输入，不修改本次封存。公开normalizer训练来源仍unknown，不能因公共常数自洽就排除其代码/导出合同缺口；Hegel实现有界官方/镜像metadata刷新器，待MAIN审读再联网。没有重训、替换统计或放宽P1。

并行其他新交付：来源身份图618节点/370显式path+SHA边/278关联（报告SHA `7c7e1792c5a17986aaf3a38539a280ec46d5910eb4d9351babbe9bf2a70b3576`），以及12episode物理指标对齐（SHA `fcf6d1f7f272cfcd3972886cc4b51214d0b2d72ab369318b8d0627cc3d97ad63`）与P2 saved动作见证图核心，均为worker交付、MAIN尚未完整集成，不冒充新的实际控制实验。metadata关联不证明physical等价或旧reset覆盖。P1、P2组合增量、完整C3及原三实验族仍未闭环。

## MAIN实际官方rollout等价CUDA与公开镜像核验

为增加有效并行度，空闲工作位已派发新overlay完整重编译、8计划物理consumer、真实结果saved比较；官方rollout和metadata交付后又续接真实RGB CUDA evaluator及同预测调用预算采样基线。六个worker工作位均有接续任务，不能解释为六个GPU实验同时运行。MAIN仍承担审读、实际执行与独立复算，无专职代码审计agent。

MAIN实际23项RAM回归通过；官方rollout三文件冻结SHA分别为source `b644621e14c66b3c4ede4ace2481fd44a52816d55d888fd7d90969813c850dc0`、tests `92b6efa6edadadcd9861c606a34311c9df4428507262535ee86985124f7c36dd`、doc `2ffba77dc39a819547e6091b5bf58a2262b37f0deff5c556dd60b3644853f7b3`。实际prepare100770B，SHA `a0026125f7157ea6de4b960e291e72da90a495bc59bddcfb46ee097348e68211`，verify通过。

实际HOST CUDA report：`le-wm-main/results/p1_moving_native_rollout_equivalence_cuda_20261027_v1_cuda_MAIN01/report.json`，SHA `71b67ec61f9a054ebf4ed99ab24ca5899307a3075ce45611fa34fe95e0a5caef`。24次实际安装类`LeWM.rollout`，cached真实embedding并提供自身真实过去RGB，24AE/36predict API/36predictor/36pred_proj，156嵌套ledger，不相加冒称预测次数。14.590148秒，errors=[]、模型tensor前后不变，无训练/encoder/AD/physics或CPU神经降级。

24/24完整`predicted_emb`逐位等于原手写packet路径（12root H2+12suffix H1），每份maxabs=0/maxULP=0。36个step的AE窗口、predictor两输入/输出、完整三token pred_proj均逐位一致；官方一次AE(T4)与原逐步AE(T3)在本批并未产生差异。official sum192与原mean192仅按保存数组的float64分别复算，不冒充实际criterion调用。

ownconsume通过。MAIN另写不导入执行器/模型的独立NumPy脚本，实际核460artifact、1273文件绑定、482事件链/156attempt-return、24全预测先封存再比较、所有模块窗口、action prefix及目标代价，全部通过。独立结果 `le-wm-main/results/p1_moving_native_rollout_independent_MAIN01.json`，SHA `d339ce200acd664f664c1f5897de6d10402581e161fc17b3f3f7f63b198bb87a`。这排除**本批cached官方rollout与手写packet数值差异**的解释，不排除全部代码或checkpoint训练合同问题，不将已有反转升级成理论负结果。

metadata MAIN全文审读并实际50tests通过后，在主机匿名请求固定official/mirror API和镜像commit固定README/config，共4次。report `le-wm-main/results/p1_checkpoint_public_metadata_refresh_20261027_v1_MAIN01/report.json`，SHA `4beda578da61c972b8468171a74aa7462f9d3d0fad1110a22717ff3b0b2e221c`。官方15s超时；mirror返回200、commit `22b330c28c27ead4bfd1888615af1340e3fe9052`；声明weights LFS SHA `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`与本地文件一致，config字节SHA和Git blob均一致。MAIN独立核rawreceipt SHA/长度、API身份、权重hash对照和config blob通过。

镜像不等于独立官方背书；API未列allowlist五类stats文件，不能推出所有位置都没有统计，更不能认证training normalizer。训练run/source/dependency/dataset绑定依旧unknown，未修改mean/std或模型。新overlay干预仅在两个已知root上重编译，不改default goal_pose，不冒称独立P1。研究goal保持active，所有科学未完成项保留。

### 新八计划实验：首次运行前捕获的代码错误与实际启动

MAIN审读未封存的新overlay compiler时发现consume与RAM mock都硬编码ViT hidden shape `[1,197,192]`。冻结checkpoint config实际为image224/patch14，必须为16×16+CLS=257；MAIN另对旧真实CUDA encoder receipt独立验SHA `5363644ac0555f03e8916ee8227d2b2d39a5c9bb15f53799f3f9cab609573911`，实际shape `[1,257,192]`。此为代码/mock接口漏测，不是理论负结果，也不是本轮已执行控制失败。

修复仅进入尚未执行的新三文件：prepare由固定config推导257，run hook与consume均严格检查，并新增197/256/258/错误维度拒绝和257完整CLS桥接测试。MAIN实际41项完整回归通过。source SHA `ce0e5ddb9a3ffc223efd94dbbee93a97bffa44b225a3bbb28afba41075a67ab6`、tests `7220f84e8176839f840b2d4ec47791a42842a13438068cd4376602114edccb60`、doc `800e4940538fa5c81bea8fe58c2e5f5ab4200399da97685d0c09d09444ced96c`；MAIN新预声明 `docs/D2P_MOVING_OVERLAY_RECOMPILE_PROTOCOL_20261027.md` SHA `242e3277831ce3b5ac8db7d1da27d384e02a2485a218911bb3dbe80a0db7af95`亦进入prepare绑定。

实际prepare24926B，SHA `8d622196815f90476d02d7ca6750817116e2100e81945f192a46f740156a9a20`；verify通过，随后主机CUDA正式运行。固定两known roots×off/on×GN/D2P，16真实图重新编码、576predict预算，8份全新优化动作，不是对旧动作换评分。记录时作业handle36416仍在运行；不预填成功或实际收益。后续动作必须经全八consume才放行物理。64MiB总cap、4MiB final reserve、900s、200MiB磁盘floor，不训练、不改统计/目标/goal_pose，不降CPU。

预算模块MAIN另实际58tests通过，确认其v1只适用原六方法12组sealed配置，不能套用新CEM/MPPI四组。新采样runner将独立预声明实际hooks，避免伪造八个额外组或借旧配置SHA。所有AD/SVD/wall等异质成本继续另列，不用旧all-axes相等阻止合理的同预测调用预算比较，也不冒称等总FLOPs。

## MAIN续接：实际控制、RGB接口修复与固定能力探针

本节更新上文“八计划运行中”的历史状态；旧封存源、协议和结果均未覆盖。四独立侧边栏对话仅有交接文件，当前工具没有create_thread/send_message_to_thread，不能称已派发；没有新开子智能体。

### 已完成的真实产物（非独立确认）

- 八计划overlay CUDA报告SHA `2b2e5064f74f8aca712ae0d0415fcbfb64a2c8e6325bdffe24ae4fdc3b373381`；实际物理报告SHA `56962e9370fcf249a2ed88f004a400414b491f503a4c3a9db1bd951cf984a521`，8/8完整、native success 0/8；真实RGB CUDA报告SHA `18d2f192a75e397c01e6c7eb7f6422e9bb073af865107ef469debe5b8ae32d1e`，28 encoder/projector。三个产物均已有实际consume，不再标运行中。
- 等预测调用预算CEM与local nominal-guided sampled-winner MPPI：CUDA报告SHA `60872ad114700731d82d293a616a4dede54a53b7be6514ad0cdf231d838d3ed8`，四组各144 API/预测样本。实际物理 `le-wm-main/results/moving_prediction_matched_sampling_physical_20261027_v1_MAIN01/report.json` SHA `8ce8634eddcd3e2e5b25d5b55572d80d06da1fd5d3513ee06aecec89001efa41`，4/4完整、success 0/4、errors=[]，实际consume通过4行/12checkpoint。MAIN此前实际79项物理测试通过。不是官方MPPI复现，也不是与GN/D2P等总FLOPs。
- 采样真实RGB **v2** 报告 `le-wm-main/results/moving_prediction_matched_sampling_rgb_cuda_20261027_v2_MAIN01/report.json` SHA `63c48c16b91d16b8f20c5f6ba08010d366544558f6e7eb9dd7a9542a5e50fa2f`，14 encoder/projector、无predict/AD/physics，errors=[]，实际consume通过。MAIN实际47测试通过。
- overlay saved任务指标桥 `le-wm-main/results/moving_overlay_task_metric_bridge_20261027_v1_MAIN01.json` SHA `2fa115f012dadebcca8bedefb6b6e65396f17042945d7f89c84a470a1840e045`，MAIN实际35测试通过。
- MAIN独立12分支复算 `le-wm-main/results/audit_moving_p1_task_alignment_MAIN_20261027_v1_MAIN01.json` SHA `cdec68bef38d0868e03a4bea76b2ab94e8e5e7b356bb3193a86defd1d1e116de`，18测试通过；核保存预测/真实编码/goal的mean192及误差交叉项、任务身份、native7指标，无新神经或物理。

### 捕获的代码／mock错误，不记作理论负结果

采样RGB v1绑定旧physical源码SHA，期待过期的operation_attempted/completed顶层resource布局，并误认branch_closed有顶层branch；最终真实producer使用operation_dispatch_intent/operation_returned和嵌套receipt，branch_closed身份也在receipt内。MAIN保留v1，另建v2，绑定最终physical SHA `50552d50ce3d97c9a25f938ef2b47e99bc66c88fa1644659611b5bb32843884a`，增加真实producer接缝与变异回归，执行实际physical.consume后才允许编码。v2源码SHA `dde4a59a92583161e488c906aaffe64f5f1e33c3be6bfcae9c46bd8172c83acc`。首次prepare拒绝属于接口错误，不是科学负结果。另一个采样comparison v1仍期待旧close布局，尚未实际运行，不得绕过校验或伪造事件。

### 当前证据边界与下一项已固定实验

两个root仍是known calibration roots，未获freshness认证。D2P不是所有任务轴或强采样基线上的一致赢家：root1/off的D2P真实Q约.19895、native7终距508.413，而CEM约.13811、380.814；local MPPI约.90601、140.576。native7终距含速度，native success不含速度，必须同时保留objectXY、角度、速度与原生成功，不能把不同轴混成“已修复”或“理论不可行”。overlay切换也没有一致改善全部任务轴。

MAIN已全文审读固定能力探针producer及分析器；本次分别实际31测试/53测试通过。实际probe prepare SHA `a73f841da07edaf98d6684246ddaf685d9c1860abb1d988aa7fcc93fa2e9a12b`，主机物理作业handle80516已启动。固定2 roots×off/on×GN/D2P×zero/±x/±y=40分支，每分支重建原warmup10+prefix5后执行固定raw float32 .1幅度probe5；完整51项渲染/状态轨迹匹配后才允许probe。不通过state setter近似恢复suffix，不做神经CPU降级（本项无神经）。这是确定性原路径重建，不是隐藏solver/cache序列化证明。900s/192MiB/200MiB磁盘floor保留；未有终态报告前不得称完成或推断能力收益。

完整C1/C2/C3、原三实验族、独立P1、P2组合增量与C3等预算收益仍未闭环，goal保持active。不训练冻结模型、不修改阈值或按结果挑根。

### 能力探针已完成：取代上文handle80516运行中状态

主机物理40/40完整，errors=[]、369.964127秒；报告SHA `e075d3b75c7b5a16c47d8c76c846eaa2ac4d8ebd202abd18037a7242792b076a`。saved分析SHA `19661bb010662dcfc7c9397723ecb085080fe18a7de069bdaac1f7c59e0e7557`；MAIN无producer导入的独立复算SHA `e4f668ce10f4389019f04d4fa507a6efdfa66158fc7a1609feeb74d23420d43d`，实际核32矩阵/40分支及journal资源计数通过。

D2P前缀确实改变真实后续动作响应，但root0在固定探针库没有正向object任务投影，root1则有显著正向有限响应。不能将无符号响应norm/谱增强等同任务可达改善。另MAIN全四库40候选saved评分审计SHA `f4c2f2de9022eca48d1a94b78bf18338a2d410c19f639c04ddee0791e5a0f720`：只有root0/off的线性与非线性对应评分赢家不同，符合已声明实现，不是已确认接线bug，不足以解释所有负结果，未释放或执行新winner。两项新MAIN审计合计11测试通过。详细表与边界见[D2P_MOVING_CAPABILITY_PROBE_MAIN_20261027.md](D2P_MOVING_CAPABILITY_PROBE_MAIN_20261027.md)。P1及原整体闭环保持未完成，不推导理论不可行。

### MAIN固定40探针方向预测已实际完成，四支线恢复子智能体

用户授权恢复子智能体，四既有工作位已分别续接强基线/freshness/P2/P3，独占新前缀、不派生下级、不启动神经/物理作业；MAIN继续亲自审计与CUDA验收。派发记录见`D2P_SUBAGENT_RESUME_DISPATCH_20261027.md`，派发不是科学交付。

MAIN预声明后新实现并完成40编码/80 teacher-open预测主机CUDA，report SHA `eac431fca369c977c57d38a32a600d1ef02ae2030b4112636d4c3df92bf143fe`，14.441633秒，无错误/训练/新物理/CPU神经降级。320实际模块桥接saved复核report SHA `f564c6771c1940dc3f6925999647569670a9dbbb4381bb6da5a37a5636a0c5bd`；接口/独立消费者/任务轴实际22测试通过。

八锚点teacher MSE均低于open，均值.01945718 vs .16457413；固定Q两两排序一致72/80 vs62/80。真实prefix能显著改变本批预测误差，但仍有反转，不称P1通过。进一步按原生联合agent+object位置目标复算，report SHA `ee0fdadbd99de36681bd3dbf93a4cbcbd77b4e69dbe7b8a73b35a93d1bf39bc2`：32非zero对照的真实Q/联合位置改善方向一致30/32，teacher预测29/32、open24/32。原生成功0/40。**object-only投影不正不能独自证明官方任务无进展**；root0部分无物体移动动作仍改善推杆位置。详见[D2P_FIXED_CAPABILITY_DIRECTION_RESULT_MAIN_20261027.md](D2P_FIXED_CAPABILITY_DIRECTION_RESULT_MAIN_20261027.md)。这是真实校准证据，不是fresh泛化或反馈控制修复完成。

### 2026-10-28 MAIN：实际24后缀修复、物理与验收已完成

v1 CUDA/NumPy tanh重建ULP差异导致235次token不等，保留失败且未放行物理；v2保存实际CUDA raw buffer，24case/312forward/936模块严格消费者通过，report SHA `fc620c9fdcbb5046c1d076c79cd0703da999275ec921de799f3747c0204d69ab`。

物理24分母全部执行、工程错误0，23完整后缀+1原生成功早停；实际report SHA `cc92465141b400659e024e052b05683153b4dc6c013b81bdf1e3e29355fd2663`。全saved journal/PNG/51前缀/raw动作bits消费者与原生指标复算通过，分析SHA `970bb7071d3bfb25ab5d01e5b4e87ebc07377b01e52c704799372739d2fa6c8f`。open/teacher_full/teacher_small原生成功分别0/8、1/8、0/8；联合位置误差相对原后缀均值变化-39.91193/-54.68591/+7.14731px。teacher相对open仍有3/8前缀变差，不能以单例宣称修复普适。详细证据和解释见[D2P_SUFFIX_FACTORIAL_PHYSICAL_RESULT_MAIN_20261028.md](D2P_SUFFIX_FACTORIAL_PHYSICAL_RESULT_MAIN_20261028.md)。

这取代“尚无新反馈优化成功”的旧快照；仍是两个known calibration roots，无独立P1或原C1/C2/C3全闭环。主线程审计/执行，五个子智能体已实际续接并陆续交付：P2新工程合同MAIN109测试与saved verify通过；P3冷启动新对照MAIN33测试及verify通过，其CUDA产物另验收。强基线与freshness还须按实际交付核实，作者报告值与本地结果不得混列。goal保持active，不训练、不降低门槛、不按结果选根。

### 2026-10-28 后续实际验收补充

P3冷启动512次主机CUDA已完成，report SHA `5bb52423c796713899a4947f9c0167ad9dfa32d1b336b7abf858d17dd45f8462`，严格consumer SHA `0403affb31fa137308502f5875a05960222c60540b9cf28b91012984310cd65c`；四策略各128次预测，候选/配额/1536模块完整验收，仍动作碰撞、modeldelta0，保留零结果。原C3同任务的程序类型/horizon决策仍未完整实验，不能以跨根路由替代。

四旧observed-prefix CEM64主机CUDA260调用也完成并验收，report SHA `83f9b6fa5ad991022af6bc38063c0fa6c72fca350578692b3c30f4a7e4854a6b`，exact consume SHA `b81abd4c2f6600bae69513228a9b6896e2a67f09c3e0ab46fc6901a4034f47d3`。物理adapter待交付，模型Q不能充当真实任务收益；该四prefix与最新moving8不是同一协议，不混比。238作者报告cells继续与本地结果分栏。

C2关键限制补核：最新moving四个D2P前缀均已改善官方联合位置，唯一成功分支首段179.73780→62.28799px。因此该1/8控制成功仍不是暂时退步后改变未来能力的机制见证。新根工作者新增1033文件metadata与原返回链，解决5个原子状态请求，但完整历史reset/加载闭包/scope仍缺，当前0认证fresh candidate；不以新seed或找不到旧记录追认freshness。

完整证据与当前并行任务见[D2P_PARALLEL_ACTUAL_VALIDATION_MAIN_20261028.md](D2P_PARALLEL_ACTUAL_VALIDATION_MAIN_20261028.md)。P1、独立根、强基线物理、P2/C3真实增量及原三实验族仍未全闭环，goal保持active。
