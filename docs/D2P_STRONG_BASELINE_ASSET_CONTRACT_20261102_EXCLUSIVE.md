# 强基线对照资产合同：GRASP / Hi-LeWM / LeWM / PlanWAM

日期：2026-11-02。本文仅整理已封存的一手作者报告值与本地协议边界。
不运行 GPU、环境、物理或神经推理，不读取新 outcome，不修改 gate，不建立
跨论文排行榜。

## 已封存作者报告资产

现有离线 primary-source archive 的封存报告为：

- GRASP：112 cells，`2602.00475v1`，Tables 1/3/4，PDF pp. 7–8。
  Push-T 表的 success/time 是 learned-world-model 内指标；不能改写为本地
  物理成功率。
- Hi-LeWM：106 cells，`2607.12547v2`，Tables 3/4/5，PDF pp. 13–14。
  `d=25/50/75` 是环境步 goal offset，不是本地预测 horizon；高层组件与
  macro-action encoder 的训练成本不能省略。
- LeWM：8 cells，`2603.19312v3`，Table 10，PDF p. 26。
  CEM 的预算不能借给 SGD、RMSProp 或 Adam；Table 10 的 `±` 保持原文未指定
  统计含义。
- PlanWAM：12 cells，`2610.11382v1`，versioned HTML Tables 1/2/3。
  NAVSIM/HUGSIM 驾驶指标不是 PushT manipulation success，不能横排。

既有封存导出 `results/p2_reported_baseline_table_20261027_v3/export/report.json`
的 SHA256 为
`52e687811e6d6f728bb3eda60836ea05eff11f36da2bb4af975d1dd19ef380f1`；
其状态为 238 selected cells、236 numeric、2 missing、`paper_reported`，
且 `cross_paper_ranking_allowed=false`。本次不重新导出、不读取新结果。

## 各自原协议下的报告值

以下独立列出，**不是可横比的成绩表**；完整表及缺失值见
[作者报告值交接](/home/gbk/code/LeWorldModel/docs/D2P_REPORTED_BASELINE_NUMBERS_HANDOFF_20261027.md)。

- GRASP Table 1：Push-T、H40、500 trials，GRASP 成功率 59.0%，
  成功试验中位时间 8.5 秒；这是 learned-world-model 内成功。
  原文：`https://arxiv.org/html/2602.00475v1#S4.T1`。
- Hi-LeWM Table 4：PushT，staged，目标 offset 25 环境步、固定
  `(high_h, high_replan, low_h, low_r)=(2,5,2,1)`，90.7±6.2%；
  ± 为跨 seed SD。原文：`https://arxiv.org/html/2607.12547v2#A3.T4`。
- LeWM Table 10：Push-T，LeWM+CEM 为 96.0±2.83%，LeWM+Adam 为
  84±7.12%；原表 `±` 的统计含义未确认。原文：
  `https://arxiv.org/html/2603.19312v3#A7.T10`。
- PlanWAM Tables 1/2：NAVSIM-v2 navtest EPDMS 90.9，NAVSIM-v1
  navtest PDMS 93.8，均是驾驶得分而非 PushT 成功率。原文：
  `https://arxiv.org/html/2610.11382v1#S4.T1`、
  `https://arxiv.org/html/2610.11382v1#S4.T2`。

## 本地协议边界

四类资产统一标记为：

```text
evidence_kind = paper_reported
paper_value_usage = context_only
local_comparison_status = blocked_protocol_mismatch_or_unknown
```

进入本地对照前，至少需要绑定：

| 资产 | 本地必须补齐的协议 |
| --- | --- |
| GRASP | checkpoint/world-model、root/goal 分布、观测与动作、反馈/horizon/成功分母、encoder/backward/predict/墙钟完整资源 |
| Hi-LeWM | 层级 goal offset/schedule、上下层 checkpoint lineage、macro encoder/bank 训练、高低层 CEM、反馈与 split |
| LeWM | v3 checkpoint/task、各 solver 独立迭代与停止、action block/repeat、反馈/replan、`±` 的统计单位 |
| PlanWAM | NAVSIM/HUGSIM task/split、多相机历史/ego 观测、trajectory/waypoint、驾驶指标与场景加权、LoRA/scorer 成本 |

缺失字段保持 `unknown`，不当作匹配；报告值不转写为
`locally_measured`，也不计算 root-paired CI、相对增益或胜率。

## 新增独占产物

- 合同实现：[strong_baseline_asset_contract_20261102_exclusive.py](/home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/strong_baseline_asset_contract_20261102_exclusive.py)
- 纯元数据测试：[test_strong_baseline_asset_contract_20261102_exclusive.py](/home/gbk/code/LeWorldModel/le-wm-main/tests/test_strong_baseline_asset_contract_20261102_exclusive.py)

合同拒绝跨论文 ranking、author-value outcome 字段和无证据升级；它只验证
资产分类与边界，不验证论文数值真实性，不读取 outcome，不授权运行。

## 验证

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_strong_baseline_asset_contract_20261102_exclusive.py
```

结果：**8 passed in 0.22s**。仅为标准库元数据合同测试，不是模型效果、官方复现、
预算公平或科学准入证据。

## 明确禁止

不得把四篇论文的数值合并成排行榜；不得把 GRASP 的 learned-model success、
PlanWAM 的驾驶分数或 Hi-LeWM/LeWM 的异协议表格当成本地 PushT/Reacher 结果；
不得用同名环境、同一个 `H`、相同候选数或相同 LeWM 名称推断等协议。
