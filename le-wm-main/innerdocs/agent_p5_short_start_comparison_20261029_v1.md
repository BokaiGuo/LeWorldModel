# short-start / oldH1 / oldH2 saved common-label comparison v1

本stem独占source/tests/doc；不修改已冻结full-roll或任何旧资产。只做saved数值重评分，无torch/physics/fullstrict调用，无训练/拟合/新根/生产重启。结果属于已有known-root探索性诊断，不能宣称改善、机制通过、C2或P1/P5科学准入。

## 输入和完整证据门

ROOT=`/home/gbk/code/LeWorldModel/le-wm-main`。固定旧H1/H2三组prepare/report/strict完整SHA，代码FIXED为唯一旧来源；H1 report=`e7ac4ef9eb7491f619de25b04c55f5e56560e2833a840dd5305f3db6207b3a5d`、strict=`4f6bbed041f077818627e68157b0cfab99ddba2e38100827db10f46c4aa1b759`，H2 report=`e5a561215bfe7b6f1c1c224e7661211e3f237fb16458a6a409f5a1da0fc08579`、strict=`783b65f67899791b5697bfe0819ccfb3231d57212b9e3e85aefc02419459c0ea`。旧prepare及runtime source hash同样固定，不是从文件名推身份。

新short-start通过四个必填参数显式提供output/reportSHA/receipt/receiptSHA，不猜未来SHA，不自动glob其他run，不退化成旧资产比较。MAIN已提供本次实际report SHA `1b014fc5c2e9868ee5e08e0dc05b602b5e9708a342c9c842698183f3430297c4` 和完整strict SHA `3a20eb780028ddd40a88d11c59b8ada9dcbc0c7147bf1df9aa9acde332aa26fd`；下方CLI绑定这组实际材料，worker未读取其实际分数或执行分析。

先验新report status=`all164_short_start_causal_roll_recorded`，外部receipt必须为`all164_short_start_causal_roll_saved_verified`且精确绑定report SHA、164/80、完整计数。partial/缺文件/错SHA/缺receipt直接拒绝，不输出“旧数据推得的新改善”。然后核各run实际prepare SHA、源pin、加载state/strict schema不变且三run相同、全目录artifact SHA/size和journal规范hashchain、prepare全部文件bindings。依赖原完整strict，不调用或冒称重新运行其native输入/labels/fullstrict语义检查。

## 真正对齐与明确不对齐

全164及原80对、两root×off/on×原两h×10坐标固定，zero/null全保留。逐枝核index/root/condition/task/selection、原15动作=原20动作后15、short原20动作、原normalizer和packet字节、原真实PNG/overlay身份。短启动的label refs必须精确指向旧H2 encoder的batch3和batch2，不仅shape相同。

真实源码显示：oldH1 frames5/10/15实际来源为warmup5/**plan0**/plan5；oldH2 frames0/5/10来源warmup0/warmup5/**warmup10**。plan0在goal bind后，warmup10在bind前；即使同global10也不能强求source record相等或冒称同历史。比较器核共享frame5与实际frame15/20精确来源，完整报告frame10两record与pixel SHA是否相同；允许且披露goaloverlay来源接缝，不以像素差筛枝。

固定可用phase仅H1@20、H2@15/20、short@5/10/15/20。旧H1@5/10/15、旧H2@5/10均null，不把旧full3早输出当因果short-start，不“补值”。预测数组从native calls及row refs逐字节关联，完整NPY检查shape、float32、finite、size、SHA和EOF。

所有phase统一采用每枝原H2 context encoder batch3 slots1/2为actual5/10、H2 label encoder batch2 slots0/1为actual15/20。新short保存的两batch必须与原H2全byte一致。旧H1先复算原自有batch1标签MSE以核旧标量，再以共同H2 batch2标签重评分；另列H1-vs-H2标签bitwise/maxabs/MSE接缝，不把label batch数值差错称物理变化。H2/short原saved标量也独立由NPY复算。跨时点batch3/batch2仍不同，不能据时间曲线唯一归因累积误差。

## 固定算术和资源轴

每枝所有可用phase：float64 C-order全256patch×384feature MSE。预固定candidate-reference差仅四项：short15−H2_15、H2_20−H1_20、short20−H1_20、short20−H2_20；这是数字差，不自动标“改善”。

每80对及每phase完整plus-minus response L2/dot/cosine、predicted/observed精确零标志、response-error L2、除2h的central difference L2；任一零范数cosine=null，无epsilon阈值。保留H1原标签零响应与共同H2标签零响应的并列记录，不按旧56/24或新符号筛选。不声称finite差分是无限小Jacobian。汇总只全体及四root×condition层的完整计数/均值/正负零null转移，无显著性/独立样本假设。

资源由每个完整module call的真实input shape和units元数据重算，核旧656/820与新656调度、native tensor descriptors唯一性、API/batch samples/frame samples及总IO，独立核原report counters。新short predictor656/1476和0encoder与旧H1/H2各自encoder/proprio_encoder/predictor axes分栏；新复制label bytes不伪算成predictor IO，继承成本来源不重复累加。wall/memory/model receipt原样单列，FLOPs=unknown/null；source/config/checkpoint文件hash是来源绑定而不是本分析重新加载模型或回证未保存的历史loaded bytes。metadata digest不当等预算证据。这里“validated returned”继承各自外部完整strict并交叉核完整shape/counter metadata，未再次运行原guard/fullstrict。

## 运行与输出

分析cap64MiB、floor200MiB、saved wall900s；每次按41枝group处理数组，避免缓存全164多协议full3；全bindings终态复核。输出只有新JSON，含arguments/protocol、164rows/80pairs、固定分层汇总、分轴资源、逐NPY/metadata/source的路径SHA证据。不生成未来模型/物理结果，不改门。verify确定性重做本saved比较全部算术，不执行上游fullstrict。

MAIN在ROOT执行：

```bash
M=experiments.spectral_reachability.agent_p5_short_start_comparison_20261029_v1
A=results/agent_p5_short_start_comparison_20261029_v1_MAIN01.json
.venv/bin/python -m "$M" analyze \
  --short-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_short_start_full_roll_20261029_v1_cuda_MAIN01 \
  --short-report-sha 1b014fc5c2e9868ee5e08e0dc05b602b5e9708a342c9c842698183f3430297c4 \
  --short-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_short_start_full_roll_20261029_v1_consume_MAIN01.json \
  --short-receipt-sha 3a20eb780028ddd40a88d11c59b8ada9dcbc0c7147bf1df9aa9acde332aa26fd \
  --output "$A"
.venv/bin/python -m "$M" verify --analysis "$A" --analysis-sha ACTUAL_ANALYSIS_SHA
.venv/bin/python -m pytest -q tests/test_agent_p5_short_start_comparison_20261029_v1.py
```

fixture使用synthetic完整shape NPY、metadata与明确虚构的外部receipt，只在本新模块将N降1/空pair以轻量测试完整saved接线；生产CLI强制164，无该缩分母选项。原80schedule与zero/符号算术独立测试。不将fixture/测试通过当作actual比较或科学通过。worker不运行actual analyze/verify，MAIN负责生产saved计算。已有旧goal未覆盖或假标完成。
