# P3 budget saved 配对任务/资源分析

独占新stem，仅读封存输入，不改MAIN文档/producer/chooser/配额/阈值。不运行神经、renderer、physics或网络。

## 严格入口与统计边界

`analyze`先验证外部handoff SHA、冻结handoff三件套SHA，再完整调用handoff.verify：其重建调用原saved-v2 physical.consume和原数组/源闭包校验。HOST采用MAIN实际完整strict回执，不重复host.consume。不能只用摘要、manifest或模型Q认证物理。

完整24child/48slot逐槽输出同child/source/context/goal/start/selected command身份及付费signal+compiler实际attempt/return资源。每个固定/自适应方法24槽；5付费拒绝保留已付资源但任务结果missing，2未到达保留资源/任务missing，不填0、不取有利子集。统计同时列分母、known/missing、known-only mean与缺任一项即null的full-denominator mean。双侧结果缺失则paired delta=null；全部24pair均保留。

按两个known root聚类，root内多个boundary/重复轨迹相关，绝非24独立根；根均权配对差仅描述各根已知配对，不补缺失、无p值/泛化或独立成功声明。任务指标分栏：native7距离/负距离reward、joint(agent+object)XY、objectXY、wrapped angle、原native angle项、velocity及原生jointXY+angle success。保留冻结native角度规则（非另改wrapped阈值）。native7小不等于物理成功，objectXY也不等于joint成功。原生早停按实际returned steps，不padding；改善是initial-minus-final误差，reward另作final-minus-initial收益。

实际资源按原轴独立计数，所有付费signal（含拒绝）+compiler核对完整HOST总账；不把API/sample/J/VJP/SVD/replay折合FLOPs。不将nested轴相加。总host wall按原profile scope列出；没有独占方法wall或FLOPs则null。历史物理复用非新费用/fresh执行。相同cap不等于相同actual；391 vs 634predict不得称等算力。P1/P3/C3/eqcompute/freshness等科学门保持false，明确列C3缺口。

## 精确命令

在 `/home/gbk/code/LeWorldModel/le-wm-main`，测试：

```bash
OPENBLAS_NUM_THREADS=1 /home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m pytest -q /home/gbk/code/LeWorldModel/le-wm-main/tests/agent_p3_budget_paired_analysis_20261028_v1_test.py
```

实际saved分析（独占NEW结果，重复拒绝；16MiB cap、200MiB余量）：

```bash
OPENBLAS_NUM_THREADS=1 /home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_budget_paired_analysis_20261028_v1 analyze --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_budget_physical_handoff_20261028_v1_prepare_MAIN01.json --prepare-sha256 5c39919c632fd8b7feb8b22ad310900e424d5bcb226f1afd99a6a5b89b01fc9a --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_budget_paired_analysis_20261028_v1_saved01.json
```

输出给真实analysis SHA后严格verify（重新全闭包及算术，不重跑实验）：

```bash
OPENBLAS_NUM_THREADS=1 /home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_budget_paired_analysis_20261028_v1 verify --analysis /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_budget_paired_analysis_20261028_v1_saved01.json --analysis-sha256 ACTUAL_ANALYSIS_SHA
```

RAM/真实saved算术测试不是科学证据。实际saved报告4条native_stopped中2条在第5步终止、2条在第5步前终止：分列terminal与少执行步数，不能认为4条都少执行。实际analyze/verify终态另以交付消息及实际产物为准，不回写封存文档。
