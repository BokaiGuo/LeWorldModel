# Saved-only full164 / full80 history factorial summary

独占source/tests/doc，不改封存runtime。worker只跑synthetic RAM/临时文件测试，不执行实际生产consume/analyze。无torch/NN/physics imports或调用。旧goal API仍占用blocked目标，不覆盖；本次完成仅指实现/测试/冻结，不指P1/P5研究通过。

## Authority与独立重算

入口必须显式给factorial实际目录、report SHA、strict receipt路径和其外部SHA；不能预填尚未执行结果。固定runtime source SHA `2b012657c44f2c469a79abec590b14f384a20b444f251772a6d7624247554b7d`。
先调用封存runtime完整saved-only `consume(output, report_sha)`，要求其返回dict与外部strict文件整份相等，拒绝partial；因此不是仅信任status或receipt摘要。继承并重核prepare/model/source/全部native IO/artifact/journal/caps/预测seal后解label。summary再自行加载原full3 NPY、校验SHA/shape/dtype/EOF，独立实现而非调用runtime评分函数：全部164×4 common-label MSE、三MSE效应、feature效应L2、全部80四cell响应norm/cosine/plus-minus MSE及response效应L2，逐项与原report exact相等，不用allclose。

共同label始终为H1实际20特征。H2 label batch接缝仅单列bitwise_equal/maxabs，不交换label评分。四cell AA/AP/PA/PP沿封存语义，交换完整5/10/15视觉/proprio embedding块，不偷称单15帧或纯horizon。额外原生agent position/velocity权限披露，不称RGB-only或跨模型同输入公平比较。

原56/24成员只来自固定零集合分析 `results/agent_p5_jepa_callback_zero_sets_20261029_v1_MAIN01.json`，外部SHA必须为 `35b4c01e4ce7ad846af740878b5399ef17247425f22cba44ceb330a55755c3be`，核H1 authority与全部80精确identity/order/membership及其bindings。不会依新cosine/幅度挑选或重新定义cohort；独立观测response的exact-zero必须与原成员一致。

## 固定汇总及分母

分层预先固定：全体、root0/root1、off/on、root×condition，共9个table。各层内部保留以下分母，即使某cohort在该层为空也输出0与null统计：

- branch层：all164；4个baseline；原56 pair的112个端点；原24 pair的48个端点。baseline不能塞入pair cohort，也不因两个h而重复。各层局部分母明确记录，名称中的164/56/24为全体cohort名称而不是局部数量。
- pair层：all80、原fixed56、原fixed24。所有失败/zero/null保留，没有显著性、独立样本假设或有效样本筛选。

每branch四MSE和三类效应保留原始标量；每pair保留四cell cosine、predicted/observed response L2、plus-minus MSE、response效应L2。summary描述量固定available/null/mean/min/median/max，无挑阈值或分位选择。
cosine分类仅严格 `<0`、`>0`、`==0` 和null，无epsilon。四cell之间全部12有向转换，各给完整4×4=16 bins，包含null↔zero等；总数严格等于该层cohort pair数，不只报告改善或不良方向。

```
visual = ((PA-AA)+(PP-AP))/2
proprio = ((AP-AA)+(PP-PA))/2
interaction = PP-PA-AP+AA
```

MSE效应是标量度量空间代数，不等价于predictor feature的加性/非加性；feature与pair-response效应L2另列。label seam每branch全量记录，按上述branch分层给bitwise_equal/different计数与maxabs分布。没有从效果倒推patch/h/根/overlay选择，没有新readout/拟合/Q/阈值。

## 限制与工程

这是已见前期outcome后提出的探索性分析，不是预注册确认、物理可达counterfactual、充分原因或控制收益。known roots且共享baseline，不当164独立样本；P1/P5 admission保持false。无training/neural/physics生产。全saved任务wall600s（含重建strict、数组重算、最终SHA）；输出32MiB、floor200MiB；输入metadata32MiB、NPY单个4MiB。只允许NEW同stem results文件，canonical JSON exactly1 LF、fsync，禁止覆盖。负正零null均保留。

## MAIN CLI

目录 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -m pytest -q tests/test_agent_p5_history_factorial_summary_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p5_history_factorial_summary_20261029_v1 analyze --factorial-output results/agent_p5_history_factorial_cuda_20261029_v1_cuda_MAIN01 --report-sha ACTUAL_REPORT_SHA --strict-receipt results/agent_p5_history_factorial_cuda_20261029_v1_consume_MAIN01.json --strict-sha ACTUAL_STRICT_SHA --zero-analysis results/agent_p5_jepa_callback_zero_sets_20261029_v1_MAIN01.json --zero-sha 35b4c01e4ce7ad846af740878b5399ef17247425f22cba44ceb330a55755c3be --output results/agent_p5_history_factorial_summary_20261029_v1_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p5_history_factorial_summary_20261029_v1 verify --analysis results/agent_p5_history_factorial_summary_20261029_v1_MAIN01.json --analysis-sha ACTUAL_SUMMARY_SHA
```

verify再次完整重建runtime strict和全部summary，再比较整份saved分析；不会新增神经调用。成功status为`all164_80_saved_factorial_summary_verified`，verify为`saved_factorial_summary_recomputed`。未知/缺项/partial前置直接拒绝，不补零或删分母。
ABI：`branches[164]`带root/condition/cohort/pair_index/sign及4cell MSE/三效应/label seam；`pairs[80]`带完整schedule/cohort/四cell response；`tables[9]`含branch与pair各cohort全部描述/转换；`bindings`含原模型/source/原始数组/newreport/strict和本三文件SHA；`arguments`支持精确复算。
未决前置只有MAIN实际factorial report与strict的明确SHA、完整目录保持原样和MAIN审读授权。worker不会读取未来实际分数或执行生产consume。tests用明确synthetic fixtures与替代strict入口验证接线，不将fixture模型信息当实际receipt。

工程验证：本stem18 tests passed in4.66s，包含全164/80合成analyze→writer→verify、scalar/vector效应、四类精确cosine、全部12×16转移、原成员/顺序/错误分母拒绝、NPY EOF和外部strict不整份匹配拒绝；没有生产分析结果。
