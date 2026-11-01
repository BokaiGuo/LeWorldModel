# 四方法强基线：分栏证据、协议矩阵与可执行补充验证

日期：2026-10-27（Asia/Shanghai）。本对话的有界交付是证据表与独占接口，**不是整套研究或 P1 完成**。
先读仓库 README 及用户指定六文档，再读取其引用的真实 JSON、原文 HTML 和 PDF 字节。
按 research-repo 的来源链审计原则复用既有档案；不重建来源库、不重新采集、不修改共享文件。
无子智能体、网络、训练、神经推理、物理执行、提交、推送或删除。

## 1. 已有产物与不重复范围

- 2026-10-26 两份来源审计的“未取得原文”是历史状态，不能覆盖 2026-10-27 后来的真实产物。
- 原始作者值入口：[封存原表 registry](/home/gbk/code/LeWorldModel/le-wm-main/results/p2_reported_baseline_table_20261027_v3/export/report.json)，SHA `52e687811e6d6f728ab3eda60836ea05eff11f36da2bb4af975d1dd19ef380f1`。
  共238单元格＝236数值＋2原文缺失；GRASP论文112、Hi-LeWM论文106、LeWM8、PlanWAM12。
  计数包含对照方法/配置，不是238次独立实验，也不是112个GRASP目标方法实验。
- [moving v3 已执行综合](/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_baseline_comparison_20261027_v3_main01/comparison.json)，SHA `96cf27b143bb1a839edf7bc1e4de02f4a97e433c6475c4c731c78af76300d32e`。
  当前状态 `comparison_recorded_not_admitted`，6算法×2根＝12模型行、12物理行、24 RGB图；不能仍按旧交接称“尚未assemble”。
  本轮仅读取/分栏/核算摘要，不调用主线程的 prepare、consume、assemble 或代码审计。
- [GRASP-H1 保存物理独立核算](/home/gbk/code/LeWorldModel/le-wm-main/results/p2_grasp_h1_saved_outcome_audit_20261027_v1_main_verify01/report.json)，SHA `e6f4d3a1b32b37f65048067b116f63f5d72581d85c4c31dec9a4eca6ffbf303e`。
  原分母8分支/4根对、4完成/4动作越界拒绝、仅1完整配对，0新独立根。它不是moving两根，也不是官方同协议复现；不合并、不重跑。
- [Reacher P0 sample-matched](/home/gbk/code/LeWorldModel/le-wm-main/results/reacher_d2p_p0_predict_sample_matched_calibration2_v1_20261026_cuda/report.json)，SHA `e938ba6d3f8cf6f7c1e8698fffe59c8270d1be805d022b391235662d522bf981`。
  两校准seed20261040/41，H2/history3/block5。restart-Adam仍为local proxy；536/536 predict samples不等于全资源一致。无重跑。
- [Hi-LeWM v2真实probe](/home/gbk/code/LeWorldModel/le-wm-main/results/p1_hilewm_bounded_object_fetch_20261027_v2_probe_main01/report.json)，SHA `669bf41043dfae01a9bca183fe230d61baad29234c5cc73038c80409844e0b90`。
  `failed_preserved`，首错 `TransferError: Truncated range`，`unpickled=false`。未定位到同名v2 full输出；不续传、不重复下载、不把部分字节当可用权重。

## 2. 原文版本、页表、原始来源

固定版本，不声称是最新在线版本。PDF页号沿用指定的PRIMARY_LOCATORS逐页核验记录；本轮重新验证三份PDF字节SHA，未重新执行PDF页提取。
原始文件都在 [既有 sources](/home/gbk/code/LeWorldModel/le-wm-main/results/p2_reported_baseline_table_20261027_v3/sources)；完整HTML/PDF SHA进入本轮JSON `input_sha256`。

| 方法、完整题名 | 官方固定版本 | 结果定位 | 协议定位 |
| --- | --- | --- | --- |
| GRASP，Parallel Stochastic Gradient-Based Planning for World Models | [2602.00475v1](https://arxiv.org/html/2602.00475v1) | Table1 p7，Tables3/4 p8；示例子表S4.T1.3 | §4.1–4.2 p6–7；S4.SS1.p2.1超参sweep；S4.SS2.p2.1模型内成功 |
| Hi-LeWM，Mind the Gap: Promises and Pitfalls of Hierarchical Planning in LeWorldModel | [2607.12547v2](https://arxiv.org/html/2607.12547v2) | Tables3/4 p13，Table5 p14；示例A3.T4.3 | B.1–B.3 p9–10训练；C.2 p12分母；C.6 p14 empirical bank |
| LeWM，LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels | [2603.19312v3](https://arxiv.org/html/2603.19312v3) | Table10 p26，A7.T10.6；Table5 p24另列 | D p17–18架构/CEM；E p18数据；F.1任务；**本次Table10不是v1** |
| PlanWAM，Planning-Shaped Future Representations for End-to-End Autonomous Driving | [2610.11382v1](https://arxiv.org/html/2610.11382v1) | Tables1/2/3，S4.T1.6/T2.1/T3.1/T3.3；PDF页missing | Tables11/12/13；Table13实际数据表ID为A3.SS4.p2.1，caption为A3.T13 |

三个PDF SHA依次为：

```text
GRASP  633bff79e5913401f578d4f13cbee65526221fef859d865cdaab6cfc0fa802a8
HiLeWM 03f0cf2e39166157c21baebb7808992f3481f9edf2b6cb4d771103e87b586533
LeWMv3 553a01501ac74ec56721cfdb936d3825ac941734133a1fd761f614c01b6887e9
```

## 3. 作者值与本地值分栏（禁止据此跨列排名）

示例是指定原配置，不是从完整sweep逐格挑最好。完整238单元格继续以既有registry为准。
本地列只是已保存JSON的实际值投影，本轮验证SHA、双根分母、摘要算术；不是重审全部producer/NPY/物理闭包。

| 方法 | 作者报告值 paper_reported/context_only | 本地已保存值 locally_measured/calibration | 实现身份与边界 |
| --- | --- | --- | --- |
| GRASP | Table1 Push-T H40：SR59.0%，成功trial中位8.5s；每setting500trials；H50/60/70/80 SR43.4/26.2/16.0/10.4% | moving `GRASP_H3_bounded`：两根mean modelQ=0.1626559966364103，mean actual terminalQ=0.236275930840675，native成功0/2；predict3604、backward1800（两根合计） | reimplementation / **local_variant**，不是完整official复现。H3指观察历史3帧，未来为2blocks/10步；不能当15步规划 |
| Hi-LeWM | Table4 staged d50 `(hh,hreplan,hl,rl)=(2,5,2,1)`：PushT Hi-LeWM42.0±2.8%，Hi-LeWM-C64.0±3.3%；±为跨seed SD | missing，数值null，不是0% | 有官方来源代码文本/资产指针，但完整高层对象与低层配对未验证；不能把普通bank冒充 learned prior |
| LeWM | v3 Table10 PushT：CEM96.0±2.83、SGD26±4.32、RMSProp67.33±2.49、Adam84±7.12%；该表±定义未确认 | LeWM-backbone上的本地moving CEM：两根mean modelQ=0.30700590568169106，mean actual terminalQ=0.24477232582649805，native成功0/2；predict268、backward0 | 骨干来自LeWM；本地CEM控制协议为 **local_variant**，不是Table10 official reproduction。不把模型来源认证扩展成整实验认证 |
| PlanWAM | NAVSIM-v2 EPDMS90.9；v1 PDMS93.8；HUGSIM RC53.7、HD-Score38.7 | official driving pipeline missing，数值null | 本地名字含planwam的PointMaze修复实验不是官方PlanWAM；本轮不录为该方法结果 |

GRASP的模型SR不是物理native成功；native成功计数也不与Q阈值互换。0/2只是两根校准执行事实，不推断普遍成功率或算法无效。
不计算作者值→本地值的提升百分比、配对CI、显著性或同预算排名。Hi-LeWM Table3的flat LeWM是Hi-LeWM作者复现，不计作LeWM原论文额外独立重复。

## 4. 原文协议可比性矩阵

unknown/missing表示未建立绑定，不表示论文没有说明。以下是论文通用协议；不能自动灌入未绑定的每个结果单元格。

| 维度 | GRASP v1 | Hi-LeWM v2 | LeWM v3 | PlanWAM v1 |
| --- | --- | --- | --- | --- |
| 环境/评价域 | Push-T、PointMaze、Wall-Single；learned-model success | PushT-v1、OGBCube-v0真实环境goal任务 | PushT等物理goal任务 | NAVSIM-v1/v2 proposal simulation；HUGSIM闭环驾驶 |
| 数据/划分 | DINO-wm原设置；本表精确训练/测试manifest missing | expert同轨迹起终，d25/50/75；精确episode IDs missing | PushT20k expert episodes，平均196步；TwoRoom10k、Cube/Reacher各10k；表格精确split绑定missing | navtrain103288（原文约85k/18k）；navtest12146；HUGSIM436 public scenarios |
| checkpoint/额外训练 | DINO-wm，原表精确weight digest missing | frozen low LeWM＋训练macro encoder/high predictor；-C另需empirical macro bank；30.5M中12.5M可训练；高低层配对digest missing | 端到端LeWM；训练通常10epochs；表格精确weight digest missing | frozen DINOv2 ViT-S base＋rank32 LoRA及规划模块训练；两阶段各50epochs；官方发布权重digest未认证 |
| 观察/动作历史 | 视觉latent/goal；DINO-wm 5动作堆叠；原表历史接口未绑定 | 当前/目标图像＋宏动作/低层历史；原表到本地history/normalizer绑定missing | PushT history3；5-action block；具体表格版本/动作normalizer绑定missing | 4历史帧×4相机，含ego feature；不是单RGB；部署prior不能与训练future posterior混淆 |
| 动作接口 | DINO-wm堆叠动作；raw单位/边界到本地未绑定 | 高层latent macro＋低层动作；latent32，最大chunk15，N5 waypoints/span15 | 5个primitive动作一个block；本地raw/tanh/normalization需另绑定 | 64条候选×8 waypoints/4秒，非PushT推杆或Reacher torque |
| horizon/反馈 | Table1 open-loop H40–80；H到primitive/model-step准确映射missing | d是目标环境步offset，**不是horizon**；hh/hl和staged/online另列 | 通用H5blocks＝25环境步，执行整段后replan；goal offset25、执行budget50 | 规划4秒、0.5秒waypoint间隔；NAVSIM非反应proposal与HUGSIM闭环分开 |
| 预算/优化 | 每环境/H sweep后最好配置；Table1时间仅成功trial；Table4时间分母unknown | d25/50/75执行50/100/150；高CEM900×20/1500×40/1200×60；低300×30/900×30/1200×30；编码/bank另计 | 通用CEM300 candidates，PushT30轮、elite30；Table10梯度solver预算missing，不套CEM预算 | K64≠64完整forward；训练16×A800/globalbatch128；推理完整调用/AD/墙钟missing |
| 指标/误差 | SR%模型内；successful-only中位秒；不借图中Wald CI给其它表 | SR%；Table4跨seed SD；不把最优sweep当固定配置均值 | Table10 SR%与reported ±，不借Table5的variance/seed定义 | EPDMS/PDMS/RC/HD-Score 0–100；RC/HD不能改名success |
| 样本数/统计单位 | Table1每setting500trials；seed IDs missing | 每配置50episodes、通常3seeds；seed列表missing | Table5三训练seed/同50trajectories；Table10具体seed/±绑定missing | navtest12146；HUGSIM80/157/96/103共436；不能充当436训练seed |
| 与本地直接比较 | 不可：模型/评价域/时域/预算未绑定 | 不可：本地完整高层资产缺失 | 不可：本地两根短suffix不是原任务SR协议 | 不可：环境、输入、动作、训练和指标均非本地控制协议 |

本地moving合同另列：非零动作历史，观察steps0/5/10；未来10×2 raw actions，2×5 packets；modelQ与actualQ分栏，0新增根。
GRASP每根1802 predict/900 backward，CEM每根134 predict；不声称全compute匹配。
综合JSON记录PushT权重SHA `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`、config SHA `2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09`。
这两项是继承收据，本轮未重hash大权重，不证明与各原论文单元格精确相同。

## 5. 本轮独占实现与实际补充验证

接口：[chat_baselines_20261027_bridge.py](/home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/chat_baselines_20261027_bridge.py)。测试：[chat_baselines_20261027_bridge_test.py](/home/gbk/code/LeWorldModel/le-wm-main/tests/chat_baselines_20261027_bridge_test.py)。

1. `build()`只读10个pinned输入（3 JSON、4原HTML、3 PDF），保留238条的权威引用，不重复全文提取/建registry。
2. `local_projection()`只分栏已有GRASP/CEM saved值；验证两根完整分母及摘要算术，不认证整个P1下游闭包，不给主线程做重复代码审计。
3. `compare_contracts(left,right)`要求14维完整schema，每项`{"value": ..., "source": "原文/收据定位"}`；嵌套缺失仍unknown，unknown==unknown不能match。
   environment、data_split、checkpoint、observations、action_history、action_mapping、horizon、feedback、budget、metric、sample_ids、statistical_unit、training_and_adaptation、hardware_and_timing逐项保留。
   即使metadata全match也不授权提升率、配对统计或科学准入。机器矩阵只填写本轮确实绑定的单元格字段，通用原文细节见上表，不强行填入特定cell。
4. **实际新补充验证**：从PlanWAM原始HTML Table13真实数据表提取四source/四难度计数，检查各行及各列加总；再将Table3所有难度与Avg按一位小数±0.05区间联检。
   RC舍入格加权中心53.6814220183，与作者53.7区间兼容；HD中心38.6993119266，与作者38.7区间兼容。
   原reported Avg不被重写，不计算等权难度均值，不把舍入区间当CI，不声称恢复未舍入原值。这是已执行的saved-data数值补充，不是神经实验或新模型效能证据。

实际验证：36 tests passed（0.22秒）；覆盖14维不匹配/unknown、嵌套缺失、同义open-loop结构化、真实sealed读取、全部来源SHA、双根分母、NaN/越界/精度拒绝、nested HTML表不合并、舍入相容与反例、独占输出拒绝覆盖。
GPU预检：主机RTX5060Ti，16311MiB，377MiB占用、GPU util0%、无CUDA计算进程（瞬时快照，不当资源预留）。因无必要的新同协议神经工作，未加载模型；没有CPU神经降级。

## 6. 下游可消费入口、SHA与精确缺口

最终：[机器证据表/可比性矩阵/算术结果](/home/gbk/code/LeWorldModel/le-wm-main/results/chat_baselines_20261027_delivery_v2/chat_baselines_20261027_report.json)。
JSON顶层 `reported_column` 与 `local_column` 完全分栏；`comparability_matrix`逐方法保留原cell定位/contract/拒绝理由；`input_sha256`供只读校验；`planwam_arithmetic_supplement`提供实际新增验证。
早期独占delivery_v1保留为中间快照；其中字符串schema不一致可被记为mismatch，**不可用于断言真实环境或反馈不同**。最终v2以未绑定environment/horizon记unknown，GRASP双方open-loop统一结构后记match；全协议仍不可比。没有覆盖或删除v1。

```text
bridge.py  0a5f1185f3d64fd6c41bda5d3f147f8095560527e3ebd545db784b66cf5fa55a
test.py    4670b7c7fa3b91f96022bce4256f2a09b2503e575e67cc500be508016b877ab1
v2 JSON    dc5cf07f74c8e7ca2fcb8b017da5361eac967c7d1f1d1c94f7d5f2c8055157fe
```

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/chat_baselines_20261027_bridge_test.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.chat_baselines_20261027_bridge
```

第二条只写stdout、可安全重复，不重跑实验。带`--output`时只能指定results直接子目录且前缀匹配、目录必须不存在；不要重用delivery_v1/v2。
未来有具体预训练资产/配对合同后可直接import上述preflight接口；本模块没有隐含executor、训练或下载功能。

尚缺且未被工程测试“解决”：GRASP官方协议全绑定与独立root确认；Hi-LeWM完整预训练高层对象/低层/normalizer/empirical bank配对；LeWM Table10梯度solver完整预算/误差定义及精确checkpoint/episode绑定；PlanWAM PDF页与官方驾驶pipeline/权重，以及所有方法完整硬件/总compute合同。
在禁止训练且资产不完整的范围内，不造神经数值，也不为补一行启动额外训练/下载或侵入P1。可消费交付已完成；上述效能复现仍partial/missing，研究科学门槛未变。
