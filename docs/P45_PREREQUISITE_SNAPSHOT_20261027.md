# P4/P5 前置条件与强基线证据快照（2026-10-27）

结论：**P1 未通过，Full D2P P4/P5 保持 blocked；未闭环。** 本次只新增指定名字的脚本、测试、本文与对应新 results 目录；不修改任何既有源码、测试、报告或 seal，不 commit/push/delete，不训练或学习 readout，不选新 outcome 根，不启动神经或物理实验。磁盘紧张时只生成小型清单，不下载/加载/复制权重，不占用 RTX5060Ti。

## 本次实质新增，不替代旧清单

机器产物：`le-wm-main/results/p45_prerequisite_snapshot_20261027/manifest.json`，schema `p45_prerequisite_snapshot_20261027.v1`。

复用既有 `d2p_remaining_stage_evidence` 的严格 JSON、根身份比较、SRA 配对与搜索核账函数，不重建 admission 合同。新增有限 8 MB 读取器，单独核读旧清单因 2 MB 限制未读的 3.3 MB 组合报告；不改变旧清单或其限制。固定 allowlist，不递归搜“最新结果”，不追随任意报告内路径，不读取权重、数据集或新 outcome。报告及文档绑定实际 bytes/SHA256、schema/status/gates、精确根字段与 JSON pointer、协议、checkpoint 声明；allowlist 内的上游绑定重新比对，其他路径明确 `not_rehashed`。

manifest 为每条 prerequisite 列出具体 artifact 路径/hash、读取检查与 root/protocol/model/task 差异；完整性门严格要求 `is True`，不接受整数1。缺失/坏JSON/非有限数/重复键/越界路径/超限报告均失败关闭。输入第二次读取检测并行修改；这不是原子仓库事务。`source_verification` 只是旧报告保存的声明，本轮不重新执行 seal verifier，不认证当前源码与旧 seal 相同。生成器与所复用模块本身另有源码 hash。

## 本次保存内容复算结果及阶段职责

以下路径均相对 `le-wm-main/results/`，manifest 保存完整路径。

| 确切 artifact | 本次离线复算 | 可支持 / 仍缺 |
| --- | --- | --- |
| `pusht_composition_replay4_v4_20261026_cuda/report.json` | schema v2、6 门；4根×16固定candidate；64份 receipt SHA、动作 hash、128条相邻 context/action 链边与终态 hash；predict calls/samples 384/384 | 固定已保存 witness 的内容一致性，**没有本轮模型 transition replay**；不证明 progress 类型、组合搜索收益或 P2。四根全部复用 calibration |
| `pusht_representation_runtime_audit4_v1_20261027_cuda/report.json` | 7完整性门、5数值检查；12图、64程序、448保存预测样本；保存 direct/official 最大差0，projected embedding 最大差2.1457672119140625e-6 | 最新有限样本表示/时间接线证据，不是任务度量修复或所有代码正确。数值是保存 CUDA 报告的复核，不是本轮神经重执行 |
| `pusht_program_search4_v2_20261026_cuda/report.json` | 原8门；逐根82/82/81/82；planning320 + replay7 =327 | 固定库模型搜索/成本接线，非 faithful 官方 kinodynamic 比较。缺等反馈/等资源强对照和真实任务增量 |
| `pusht_search_physical4_v1_20261026_cuda/report.json` | 保留全部4条：10/10、10/10、5/5、5/10；最后根 observation_unavailable，5条未执行；official/object-only 均false | 有真实单次封存计划执行，不得再写“从未物理执行”；30/35完成也不是多轮 observe→recompile 到终止或 P4 成功率 |
| `pusht_observed_prefix_prediction4_v1_20261026_cuda/report.json`、`pusht_d2p_budget_guard_probe4x16_v1_20261026_cuda/report.json` | 分别6/8门；均与 v5 calibration 四个RGB根重合 | 固定后缀预测/执行前预算拦截，不是新 outcome、质量—成本优势或长程任务收益 |
| `reacher_d2p_p0_predict_accounting_calibration2_v1_20261026_cuda/report.json`、`reacher_d2p_p0_predict_sample_matched_calibration2_v1_20261026_cuda/report.json` | 保留两个 calibration seed20261040/41、所有方法计数。默认 D2P/CEM samples536/256；sample-matched536/536，但calls仍122/8；restart-Adam34calls/40samples、random0/0 | 只匹配一类前向样本，不是编码/backward/耗时/反馈的完整资源匹配；不能证明官方 GRASP 或 held-out 策略优势 |
| `sra_p4_rgb_closed_loop_r3_full_20261024_cuda/report.json` | 5门；64 calibration/144既有 outcome 索引不交，immediate/spectral完整配对 | 保留 PointMaze-SRA 受限 P4，不选新根、不打开新outcome；不跨 checkpoint/archive、四 block bank 或分支移用到 Full D2P/P5 |
| `pusht_d2p_p1_outcome_physical_20261024_cuda/report.json` | 历史 fail/非贪心门false保留；v5 model preflight 不覆盖此失败 | 不能以新接口/成本校准追认 P1；正式独立机制与闭环效能证据仍缺 |

根相等只在明确命名空间内比较；RGB hash 相等表明复用，不相等也不自动证明 source episode/训练/修复/校准独立。PointMaze整数 archive index、Reacher seed与RGB身份、PushT root SHA不互相转换。checkpoint 路径/报告内 digest 不等于本轮加载或权重真实性认证；各任务 LeWM checkpoint 也不等于预声明独立架构复制。

## 最新 representation/runtime 边界

依据新 `docs/D2P_REPRESENTATION_RUNTIME_AUDIT_20261027.md` 及其确切报告：当前 root-anchored context 为 **emb3 + between-state actions2**，再接新动作 B,C；不能拼 incoming3 多预测一步。旧P0 causal/bootstrap约定另有真实调用语义，本次不把它笼统判为索引bug，也不修改历史源码。

当前 Q metric 仍可能在物体静止时将 agent 运动算作很大的能力响应。沿用 Q 的 recurrence、guard、成本统计可独立核验；**Q-latent 增益不能宣称物体 task efficacy 修复、C2成立或P1通过**。

## 哪些可以不依赖 P1 真实推进

本轮已实际完成：组合保存 witness 内容核账、搜索/强基线资源收据复算、四根复用及上游报告 hash 对比、新 runtime 结果登记、失败与 guard-stop 全根保留、公开基线协议归类。这些是已完成的离线审计工作，不是“准备下一步”的代称。

仍可独立并行的窄任务是：取得 GRASP/Hi-LeWM/LeWM 缺失一手表与版本，核对官方 GRASP 算法映射、资产许可/checkpoint revision，核验 RGB context/动作/normalizer 接口与资源单位。未取得材料仍留unknown；不可由此启动正式 P5 或训练模块。若主线程另行安排新的神经接口/成本诊断，必须主机CUDA、全新目录且由其调度；本清单不需要任何新CUDA实验，故不提供重复的GPU重跑命令。

正式 Full D2P P4 仍缺各环境自己的 P1及采用组件的P2/P3准入、冻结起终分布与总预算、RGB多轮重编译至终止、同checkpoint同反馈强对照、根级配对success/cost及PushT接近/接触/推送/闭合分解。P5仍缺预声明独立冻结架构、合法可用且接口验证的资产、每模型P0/任务前提、复制核心机制与真实收益；同模型多任务另报。代码通过不补这些科学槽位。

## 四项 reported baseline 审计 / 使用勘误

检查范围严格为现有 GRASP/Hi-LeWM、LeWM/PlanWAM 两份 named reported-baseline 文档，以及 baseline policy/faithfulness 与闭环文档。manifest记录原文行号/hash。**这两份 reported 文档中未发现把公开值冒充本地同协议结果的表述**；不据此宣称全仓库没有此类问题，也未在线重新核验论文原文。

| 名称 | 当前能确认的来源状态 | 禁止挪用；应如何标记 |
| --- | --- | --- |
| GRASP | 现有报告明确已核实外部性能条目为0；不是性能为0 | `insufficient_metadata`；本地 `grasp_style_restart_gd` 只能叫restart-Adam proxy，不是官方GRASP复现或官方成绩对应臂 |
| Hi-LeWM | 原文身份、版本、表格未取得 | `insufficient_metadata`；不能将内部PushT改善描述写作已核验数值，也不能自动和本地suffix协议比较 |
| LeWM | v1身份有来源，但Table10数值及完整protocol未逐项核验 | `insufficient_metadata`；附件优化器数值不升级为paper_reported，本地权重存在不证明原表revision相同 |
| PlanWAM | 既有文档按缓存v1转录：NAVSIM-v2 EPDMS90.9、v1 PDMS93.8、HUGSIM RC53.7/HD-Score38.7 | `paper_reported/context_only`；驾驶任务、训练规划模块、4摄像头历史、trajectory输出、指标与本地Reacher/PushT不同，不算相对提升率、不混入配对统计/等预算排名 |

PlanWAM上述数值的证据在本轮止于现有来源审计文档，缓存/PDF未在本轮重新认证，manifest明确 `primary_source_reverified_this_run=false`。全部数值对应的原始表行保存于 `reported_values_source_rows`，不是本次生成的性能成绩。本文只作新勘误/使用澄清，不改原文。

## 可运行命令、资源与验证

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p45_prerequisite_snapshot_20261027.py tests/test_d2p_remaining_stage_evidence.py tests/test_d2p_reported_baselines.py
.venv/bin/python -B -m experiments.spectral_reachability.p45_prerequisite_snapshot_20261027 --write
```

测试实跑 **120 passed**（含26个新增测试，另94个旧测试），无神经工作。新增测试包括真实既有报告回归、保存 witness 损坏/漏项/重复/链边/计数/整数假布尔、严格JSON、有限大报告、路径越界、缺来源、固定allowlist、不解锁阶段、stdout只读和输出拒绝覆盖。

生成命令退出 **1为预期的阶段blocked**，不是CUDA失败。`--write`只创建指定新results目录，存在则拒绝覆盖；后续可去掉`--write`重算到stdout，不修改已产出的manifest。预计显存0、磁盘新增远低于1MB、内存为有界JSON解析级而非模型级、通常秒级；无GPU验证待启动。测试与脚本使用 `-B`、禁pytest cache以避免写入既有源码目录。

最终必须以manifest中的 `all_read_bytes_stable`、artifact/check/binding/recomputed字段区分实际读取、内容复算、未重新哈希的权重与未满足的科学要求；任何completed或正向完整性flag都不开放P1/P4/P5。

实际生成已完成：manifest约196KiB，13个固定报告槽中11个通过保存报告检查；搜索v1缺失完成报告、历史P1物理失败两个槽保持blocked。allowlist内已重算的上游报告绑定没有发现hash冲突；全部二次读取一致，`all_read_bytes_stable=true`。退出1符合预期，Full D2P P4/P5仍blocked。未重新认证seal、权重、原论文PDF或CUDA trace；该有限结果不推出仓库其余结果不存在。
