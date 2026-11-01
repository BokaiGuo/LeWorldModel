# D2P P4/P5 前置证据清单（2026-10-26）

结论：正式 P1 尚未通过，Full D2P 的 P4/P5 仍为 `blocked`。本轮只新增离线、可机读的报告清单生成器及逻辑测试，没有训练、神经推理、物理执行、GPU 占用或 CPU 神经降级；没有修改准入规则、封存文件或 runner。报告中的 completed/全 true 门不等于研究阶段通过。

## 定义与来源

以现行 `docs/SPECTRAL_REACHABILITY_ATLAS_RESEARCH_PLAN_20261023.md` 的“阶段和解锁门槛”表、`docs/D2P_RESEARCH_CLOSURE_AUDIT_20261025.md` 的“原提案要求与当前证据”表、以及 `docs/D2P_RESEARCH_REQUIREMENT_EVIDENCE_MATRIX_20261026.md` §5 为准。生成器记录原文行号、表格行和有限文档 SHA256，便于后续定位定义漂移，不重新规定科学门槛。原附件只命名 P0/P1，P2–P5 是项目闭环编号，预算路线 B0–B3 不替换这些阶段。

| 阶段 | 确切范围 | 不能替代它的证据 |
| --- | --- | --- |
| P4 | PointMaze、Reacher、PushT 的 RGB-only receding-horizon 闭环；同一 checkpoint、匹配优化预算、根级配对 success/cost。各环境自身 D2P 准入和适用 P1/P2/P3 后，固定起始/目标分布与总预算，多轮 observe→recompile 至终止、保留失败；PushT 分接近、接触、推送、闭合，物理状态仅作 evaluator 标签 | 单次 suffix、固定双段执行、预测误差降低、渲染重放、旧 SRA 分支均不能填 Full D2P 的闭环收益槽 |
| P5 | 预声明的独立冻结模型/架构及任务复制；核心算法与 P1/复制组件先冻结和准入，每模型重新验证接口/P0与任务前提，复现机制和任务收益。保持 training-free，披露预训练/额外学习成本；同模型多任务另列 | 单 checkpoint 多 seed、跨论文成绩、teacher/adapter/repair/qpos readout 资产、同模型任务数均不能冒充独立架构复制 |

PointMaze-SRA 的有限 P4 证据单列，不能抹去，也不升级为 D2P、Reacher、PushT 或 P5 结论。本生成器不是 P4/P5 准入器；即使输入肯定 admission 标志，也不会解锁阶段。

## 有限清单的实际核查

以下 `results/` 均相对 Python cwd `le-wm-main`。只读取代码固定 allowlist 中的报告，不递归发现“最新目录”，不扫描数组、数据集、权重或巨量结果。对照最新文档选择明确版本，而非按 outcome 选择证据。

| 报告/版本 | 当前离线核查 | 科学边界 |
| --- | --- | --- |
| `sra_p4_rgb_closed_loop_r3_full_20261024_cuda/report.json` | `admission_pass`，5 门严格 true；重新核对 64 calibration 与 144 outcome 索引不交，immediate/spectral 各自完整覆盖 outcome、无重复；proxy 校准根一致 | 仅既有 PointMaze-SRA 分支。这里只核验同 archive 的根索引和配对，不重新认证源 episode 独立性、source seal 或效应统计 |
| `pusht_d2p_p1_root_anchored_v5_calibration_20261026_cuda/report.json` | schema v5、`pass`、7 门严格 true；作为后续已知根复用的比对来源 | calibration/preflight 不是正式 P1 收益准入 |
| `pusht_observed_prefix_prediction4_v1_20261026_cuda/report.json` | `completed`，6 门 true；四个 RGB root SHA 与校准四根完全重合 | 反馈固定后缀预测接线，不是新 outcome 或真实任务增益 |
| `pusht_composition_replay4_v4_20261026_cuda/report.json` | 路径存在，超过 2 MB 上限，明确 `blocked`/未读，不读取其大内容 | 完成状态来自 `D2P_DOWNSTREAM_CUDA_VERIFIED_PROGRESS_20261026.md` 的主线程记录，不冒称本生成器独立核验六门或根 |
| `pusht_d2p_budget_guard_probe4x16_v1_20261026_cuda/report.json` | `cuda_integration_pass`，8 门 true；四根与校准完全重合 | 硬预算拦截不是质量—成本优势，也不是 P3/P4 |
| `pusht_d2p_sealed_program_execution4_v1_20261026_cuda/report.json` | `completed`，5 门 true；四根与校准完全重合 | 已有物理固定程序校准，不是多轮规划或独立 outcome；本轮无新执行 |
| `pusht_d2p_p1_outcome_physical_20261024_cuda/report.json` | 历史 `fail`，非贪心完整性门 false，明确 `blocked` | 保留失败，不以新校准/接线追认旧 P1 |
| `pusht_program_search4_v2_20261026_cuda/report.json` | `integration_completed`，8 门 true；四根全部复用校准；独立加总保存收据得到 planning 320、完整计划 replay 7、合计 327，逐根 82/82/81/82 | 真实主机 CUDA 接线由主线程执行，本轮仅离线核账。无物理执行证据；P1/P2/P3 flags 均 false；不是官方 kinodynamic 复现、全局最优、完整算力匹配或 P4/P5 |
| `pusht_program_search4_v1_20261026_cuda/report.json` | 完成报告缺失；最新搜索文档明确首次调用 instrumentation 阶段名报错，清单保留 `failed attempt`/`blocked` | 不写成从未尝试，不以 v2 回填 v1，不把代码失败算作方法负结果；旧目录不改不删 |
| `pusht_search_physical4_v1_20261026_cuda/report.json` | `integration_completed`，7 门严格 true；全部四根保存并与校准完全重合。根0/1各完成10/10动作，根2完成5/5，根3完成5/10后 `observation_unavailable`，剩余5条未执行；official/object-only 四根均 false | 是新增真实物理接线证据，不再标为“搜索从未物理执行”。根3 guard停止不等于完成完整计划；无独立outcome、正式P1或任务收益结论 |

搜索新增依据：`docs/D2P_PROGRAM_SEARCH_CUDA_INTEGRATION_20261026.md`。v2 是固定 16 primitive、depth≤2/beam4 的模型搜索；不同根返回长度可为 1 或 2，不补零伪装等长比较。只核对保存样本数与 cap，不把预测 Q 代价当物理任务代价。无独立 CUDA trace 认证、无来源封存重认证。

后续物理执行新增依据：`docs/D2P_SEARCH_PHYSICAL_EXECUTION_20261026.md`。这是**单次有限长度封存计划执行**，观察 guard 不等于重新规划；不是 multi-round observe→recompile P4。生成器直接保留四条保存执行记录的状态、错误、计划/完成/未执行动作数和 official 标签；不调用 evaluator 重算，不把报告级 `no_execution_errors=true` 翻译成每条计划均完成。根3第5步观察缺失后停止，既不删除该根，也不补执行或改候选。`new_physical_execution=0` 仅指本审计轮无新执行，不否认主线程上述真实运行。

## 缺项如何表示

生成器输出五个正式槽：P4 PointMaze/Reacher/PushT Full D2P 闭环、P5 独立模型复制、P5 同模型多任务矩阵。现行受检文档未绑定这些正式报告，故 `path=null,status=not_run`；其精确含义是“本有限清单无正式完成报告”，不是证明全仓库无任何相关运行。阶段仍为 `blocked`，原因包括 P1 未通过和缺少 root/source/protocol 绑定的正式验证。

每份报告分别记录 availability、reported status、schema、required missing gates、所有额外失败/非布尔 gates、root audit 和 admission=false。所有门必须严格 `is True`，整数 1 不作 true；缺失、坏 JSON、重复键、NaN/溢出、越界符号链接、非普通文件、超限读均失败关闭。根比较保留类型/命名空间；RGB SHA 不交也只标 `rgb_hash_disjoint_only`，不能证明 source episode、训练/repair/calibration 独立。未知必须保持 unknown，不以 root-disjoint=true 字段代替重算。

## 复跑与测试证据

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
python -m pytest -q tests/test_d2p_remaining_stage_evidence.py
python -m experiments.spectral_reachability.d2p_remaining_stage_evidence
```

CLI 仅 stdout 输出 `d2p_remaining_stage_evidence.v1` JSON，退出码 **1 为预期的阶段阻塞**；不创建结果文件或目录。每个显式文件最多 2 MB，SHA256 仅计算实际受检小文件；不执行所读源码。

本轮新增测试覆盖缺失/坏/超限报告、状态/schema/门检查、根重复/交叠/漏配对、proxy split、校准复用、未知 source 独立性、搜索 327 样本收据核账及错误注入、v1失败与v2完成分离、物理执行四根保留/guard停止、肯定标志不放行、无目录发现和只读输出。所有 fixture 只是逻辑合同测试，不填实验结果槽。

本轮普通解释器实跑：新增测试 **38 passed**；连同 `test_d2p_stage_admission.py`、`test_d2p_downstream_frontier.py` 的联合回归 **84 passed**。实际 CLI 输出成功解析为 JSON，退出1（预期阻塞），P4/P5 均 blocked；核验搜索8门/327样本、物理7门/四根及全部 official=false。`git diff --check` 通过。主线程最终另行实跑七套联合 **251 passed**，与本轮测试数不相加，不作实验或准入证据。

本轮仅新增：`le-wm-main/experiments/spectral_reachability/d2p_remaining_stage_evidence.py`、`le-wm-main/tests/test_d2p_remaining_stage_evidence.py`、本文件。保留脏工作树，不提交、不推送、不删除。
