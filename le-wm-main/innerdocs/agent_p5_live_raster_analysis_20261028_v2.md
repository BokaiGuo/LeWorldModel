# P5 live raster 完整 saved 代数分析 v2

独占三件套 `agent_p5_live_raster_analysis_20261028_v2`，仅saved CPU数组/有理数计算，不执行神经、CUDA或物理。MAIN审读并在实际encoder结束后提供report/prepare/strict receipt外部SHA；本交付不预填任何生产分数或结论。prepare可在实际结果前封存分析合同，不读取未来结果。

输入限定已审`main_p5_live_raster_encoder_20261028_v2` source SHA `949a61354b9ecddd3b3a523fff398a92835b4773c455aa4cb2f1ae21231ac382`。调用其saved consume验证全十图/16分数/模型与事件闭包，并与外部SHA指定的完整strict receipt相等。然后分析器另读十个原始full256×384 float32 patch NPY，独立float64计算所有16个mean squared distances，逐canonical相等核对report/receipt。不会调用producer的scores/encode/runtime函数；consume只作上游authority验证。actual shape/路径/SHA/bytes缺任何项拒绝，不截token、不填0、不补图。

## 固定全量与时间点边界

词顺序固定`xy,yx,x_minus_x,zero_zero`；terminal/goal颜色各`visible,white`；共4×2×2=16分数，顺序word-major/terminal-major/goal-minor。不能事后挑词、颜色、最好条件或独立根。

**新raster为原四词frame15，旧native coverage/reward为frame20。** 来源为live renderer prepare.targets（15）与冻结encoder→v5 authority，v5源码54–56行明确抽取原record.frame20。因此旧coverage原样另栏，不能冒充当前frame15标签；native_vs_raster_rank_agreement固定null，不报告跨时间“同状态”一致率。它也不是本次新native观测或任务成功。

颜色干预只支持敏感性描述，不建立C2、因果机制、组合收益或科学准入。old renderer失败完整引用，old_renderer_release=false，historical_full_public_equivalence=null。四词来自单known构造root，不是四独立样本；无显著性、训练/拟合/阈值调整、p值或置信区间。

## 精确代数合同

对每词，令S00=visible terminal/visible goal，S01=visible terminal/white goal，S10=white terminal/visible goal，S11=white/white；越小表示表征距离越小，不叫任务成功。

```text
mu = (S00+S01+S10+S11)/4
T  = (S10+S11-S00-S01)/2       # terminal颜色边际主效应
G  = (S01+S11-S00-S10)/2       # goal颜色边际主效应
X  = S11-S10-S01+S00           # difference-in-differences交互
S(t,g) = mu + (2t-1)T/2 + (2g-1)G/2 + (2t-1)(2g-1)X/4
```

另列terminal在两种固定goal下的simple effects、goal在两种固定terminal下的simple effects、S11−S00。每词以及固定四词等权平均均完整报告，不按效果选择聚合。正效应仅表示cost上升。交互为颜色组合的评分非加性，不等于神经控制机制。

“精确”指将已保存binary64分数用`Fraction.from_float`转成精确有理数，主效应/交互/重构用有理数运算；不是声称原float32 encoder或float64 MSE无舍入。每值输出numerator/denominator十进制字符串及便于阅读的binary64_display；后者不能替代精确分数，显示溢出时为null，不填0。逐格重构严格相等，不用epsilon/allclose。

对任意有序词对(a,b)，各条件列D=S_a−S_b，D<0偏好a，D>0偏好b，D=0为tie。完整12有序词对×4条件=48条；所有6条件对×12有序词对=72条反转比较，同时覆盖a_to_b/b_to_a，含goal-only、terminal-only与双颜色改变，不只展示预期方向。严格反转当且仅当D_old×D_new<0；tie不是严格反转。以6无序词对×6条件对=36作描述性census，明确不是36独立样本；不重复计双向为增大n。

## ABI与CLI

prepare schema `<stem>.prepare`，status `analysis_protocol_only_no_actual_results`，actual_report=null，bindings包含自身三文件与已审encoder source，完整protocol事前固定。实际analyze report schema `<stem>.report`，status `all16_saved_raster_algebra_recorded`；authority含analysis prepare/encoder prepare/report/strict receipt全部绝对路径与外部SHA，analysis内完整16scores、4词factorial、固定平均factorial、48rankings、72reversals、36census、旧coverage另栏。另保留五state颜色feature差异原列，不能事后选择其一。错误拒绝且不产生伪成功输出；保留调用方上游全部negative/partial不改。输出2MiB cap、200MiBfloor、排他创建。

consume只接受analysis报告外部SHA，按authority重做整条saved分析、完整canonical相等，返回`saved_algebra_verified`；不神经fallback。若上游source改变，必须新版本，不能把未知source当当前已审source。

```bash
# le-wm-main根目录；PREP/SHA等占位符须替换为真实文件与sha256sum结果。
.venv/bin/python -B -m experiments.spectral_reachability.agent_p5_live_raster_analysis_20261028_v2 prepare --output results/agent_p5_live_raster_analysis_20261028_v2_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p5_live_raster_analysis_20261028_v2 verify --prepare ANALYSIS_PREP --prepare-sha ANALYSIS_PREP_SHA
.venv/bin/python -B -m experiments.spectral_reachability.agent_p5_live_raster_analysis_20261028_v2 analyze --prepare ANALYSIS_PREP --prepare-sha ANALYSIS_PREP_SHA --encoder-prepare ENCODER_PREP --encoder-prepare-sha ENCODER_PREP_SHA --report ACTUAL_ENCODER_REPORT --report-sha ACTUAL_REPORT_SHA --receipt ACTUAL_STRICT_RECEIPT --receipt-sha ACTUAL_RECEIPT_SHA --output results/agent_p5_live_raster_analysis_20261028_v2_saved_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p5_live_raster_analysis_20261028_v2 consume --report ANALYSIS_REPORT --report-sha ANALYSIS_REPORT_SHA --output results/agent_p5_live_raster_analysis_20261028_v2_consume_MAIN01.json
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p5_live_raster_analysis_20261028_v2.py
```

RAM tests用人为交叉/平局/单ULP和大数舍入矩阵，只检查代数与拒绝逻辑，不是生产结果。实际encoder report SHA尚待MAIN提供；worker不运行actual分析，不假造receipt，不修改旧三件套或MAIN encoder。

## v2 authority修订（指标不变）

只接MAIN encoder v2与其saved receipt：上游明确调用新saved consumer v2，以原ctor一次randint完整解释baseline→anchor的NumPy state变化，不删除RNG或改renderer。分析器原16score、所有factorial与全部双向排序定义保持原样；不能把v1 encoder/receipt标签改成v2。prepare允许实际report未形成时预封合同，不读取未来结果。实际report/strict receipt SHA仍必须由MAIN提供后才能analyze。保留并单列完整上游baseline_to_current_anchor与source_comparisons的hash绑定；不回证旧renderer，不把frame20 coverage当frame15标签。

实际analysis report的`raster_time_diagnostics`逐项保留完整`baseline_to_current_anchor`5项和`source_comparisons`65项，并提供`artifact_base`（真实renderer report.parent）用于解析原收据相对路径；不只转述“RNG已通过”，不删除exactdiff/构造draw证明。目标/terminal因子正负方向固定指cost增减，非coverage或完整PushT任务成功。

验证范围：34 RAM tests含原26代数/全分母/单ULP/缺项检查，另含真实shape的十个synthetic NPY内存接线、独立16scores复算、analyze→consume、分数/receipt/NPY/SHA/endpoint frame/v1 status/缺时间证明/缺diff拒绝。mock只替代上游authority，不能将这些tests当实际encoder/physics验证。worker未读取实际CUDA结果、未运行actual analyze、未生成生产prepare或receipt。
