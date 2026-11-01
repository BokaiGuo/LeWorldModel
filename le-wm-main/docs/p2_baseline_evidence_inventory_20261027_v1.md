# P2 强基线证据清单：本地只读核验

独占同 stem Python、test、本文三个文件；未修改共享进度页、旧代码或封存结果。
适用祖先目录及仓库内未找到 AGENTS.md（空 `.agents`/`.codex` 不提供额外指令）。
已读取外层 `docs/D2P_PARALLEL_CUDA_PROGRESS_20261027.md`，以当前 JSON 为核验输入，
不把文档里的历史缺口当成当前代码不存在的证明。

## 可直接交给 main 的入口

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p2_baseline_evidence_inventory_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p2_baseline_evidence_inventory_20261027_v1
```

第二条向 stdout 输出 JSON，不写文件；无需训练、下载、模型或模拟器。
返回 0 表示清单的结构/保存指标核验通过，**不表示科研准入或强基线齐备**；
缺文件、坏 hash、错根、预算/指标不一致返回 2，并输出具体错误，不跳过坏证据。
这是固定来源快照，不自动选 latest、不扫描大结果树。来源后续改变须人工审查新版本。

## 实际复用和检查

- 复用 `results/p2_reported_baseline_table_20261027_v3/export/report.json` 及
  `comparison_readiness.json`，对照原 SHA256SUMS 核验这两文件；逐单元格调用既有
  `d2p_reported_baselines.validate_record`，检查唯一 ID、来源 HTML hash 格式和计数。
  不重做原文下载/解析，不声称本次重核所有原文与 PDF。
- 四论文共 238 单元格：GRASP 112、Hi-LeWM 106、LeWM 8、PlanWAM 12。
  每项保留原 record（环境、版本、checkpoint unknown、预算、指标、出处、统计定义），
  两项 unknown 原样保留。按论文归组不是同名 solver 的纯净实验集合；例如 GRASP
  论文也包含 CEM/GD/LatCo，Hi-LeWM 论文包含作者复现的 flat LeWM。
  CEM 索引的30条与论文集合重叠，不得再求和为独立结果数量。
- 本地复用 `p2_gradient_mpc_matched_comparison_20261027_v1_audit01/report.json`，
  按稳定 root SHA/arm 接五个原 physical report（不按 JSON 数组序号）。
  对这五份报告核对比较器保存的文件 SHA，再核对初始物理身份、固定10步完整端点、
  pose/official distance/success，以及 planning+replay+compile=total 算术。
  原表 success 的整数0/1与 physical 的 bool 明确转换；非二值拒绝。
- 保留每根 predicted Q、真实 encoder Q、pose 和 official distance，不能互相替代。
  仅 pose/official distance/success 与 physical 报告交叉核对；Q 沿用上游数组审计结果，
  本模块不重新读取 latent 数组、编码或认证神经执行。
- 读取 full20 sampling 的 protocol，列出记录的 config/weights SHA，不打开大权重，
  不冒充逐方法独立 checkpoint 认证或全传递 source seal 验证。
- GRASP 复用 `p2_grasp_h1_saved_outcome_audit_20261027_v1_main_verify01/report.json`，
  核验稳定根唯一性、full-path/unmatched 分母和差值缺失规则，保留 selected/warmup
  预算、原始拒绝及来源缺口；不是重新证明官方完整算法或物理轨迹。
- 已登记代码路径做有界只读存在检查并记录 SHA；存在不等于本次执行或算法认证。

## main 的具体缺口与已有可用资产

以下路径均相对 `le-wm-main`；Python 实现位于 `experiments/spectral_reachability/`。

| 项目 | 已有代码/保存结果 | 仍缺什么，不得如何表述 |
|---|---|---|
| CEM / random shooting | `p2_full20_sampling_20261027_v2.py`；`results/p2_full20_physical_20261027_v2_cuda_v1/report.json` | 每臂四根336预测；共享H2不代表同反馈/全成本。不能用旧constant2报告代替full20结果 |
| MPPI | `p2_full20_mppi_20261027_v1.py`；`results/p2_full20_mppi_physical_20261027_v1_cuda_main_v1/report.json` | 四根336预测；clip-Gaussian inspired，尚不能称官方MPPI。需核对噪声/控制成本/importance weighting合同 |
| gradient MPC | `p2_full20_gradient_mpc_20261027_v1.py`；`results/p2_gradient_mpc_physical_20261027_v1_cuda_main_v1/report.json` | 四根336预测另156 AD/update；开环不是receding MPC。需要同observe/replan日程，不能抹去MATH backend与反传成本 |
| GN | `run_reacher_d2p_p0_goal_progress.py` | Reacher P0同D2P/DLS求解器等价性，不是独立强GN-MPC；需独立multistart arm、同目标/动作/反馈和资源收据。本清单不认证历史GN结果 |
| no-composition SRA | `p2_no_composition_sra_20261027_v2.py`；`results/p2_no_composition_sra_physical_20261027_v2_cuda_main_v1/report.json` | 四根352预测，另Jacobian/SVD；不等同其余336预测预算，更不是已证组合增量 |
| GRASP | `p2_grasp_reference_core_20261027_v1.py`、`p2_grasp_official_contract_20261027_v1.py`、`p2_grasp_h1_cuda_runner_20261027_v1.py`；上述saved audit | 新机制子集不再只是restartAdam，但非完整论文复现。H1规划/real-H3诊断与H2不可混排；八槽四拒绝，仅一根完整配对，另三根unmatched |
| Hi-LeWM | v3 reported表及readiness | 无本地高层模型/宏动作bank/分层评估收据认证；低层冻结不等于training-free。不授权为了补表去训练 |
| LeWM | 本地checkpoint绑定及v3 Table10 reported表 | 本地使用LeWM不等于Table10复现；精确试验checkpoint/根、梯度预算、±定义未齐 |
| PlanWAM | v3 reported表及readiness | NAVSIM/HUGSIM指标和训练接口不同；不填本地PushT排名，不借64候选宣称等总计算 |

五个H2方法的每根记录均保留；全部各0/4 official success，不能据此编造正向效果。
共享四个已用校准根不是独立holdout。GRASP唯一配对不能与它们拼接成统一样本。
下一步 main 可直接消费 JSON 中 `missing_or_unmatched`、`assets`、每根 `budget` 和
`metrics`，核对所需比较的精确合同；不需要重新搜索论文或重跑已经保存的控制实验。
所有项 `paper_protocol_reproduced=false`、`total_compute_matched=false`；不存在SOTA排名。

## 资源与验证边界

首次真实清单运行状态为 `inventory_valid_scientific_gaps_open`，missing_assets为空，
输入累计2,758,059 bytes（包括重复小报告读取），单文件16MiB、累计96MiB硬读取上限。
输出只到stdout，无归档/副本/新结果目录；开始时磁盘约669MiB空闲，没有删除资产。
测试仅小型常量字典/JSON，覆盖乱序连接、错根/臂、重复/缺失根、部分端点、指标污染、
未知成本不归零、编译额外成本、NaN/布尔伪数、非法二值、路径逃逸/大小上限、来源SHA污染。
这些是算术和元数据验证，不是新的模型、物理、泛化或论文性能证据。

验证结果：新增29项测试全部通过；与既有 reported registry 联合85项全部通过
（0.19秒）。真实清单再次运行成功，五份physical文件SHA均匹配上游比较绑定。
`git diff --check` 通过。除这三个新增文件外未写入其他项目文件，没有另派代理。
