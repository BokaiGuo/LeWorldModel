# Early-history factorial 独立saved标量核验器 v1

只独占本stem source/tests/doc；不导入producer、其math/helper或torch/physics，不重跑producer consume/fullstrict，不改任何封存材料。MAIN实际HOSTCUDA report SHA `4fc86a7fdc69fbe95003e417ee38e3a4d430360a8eb5bd0c89f73ee845926d0f`；MAIN完整持久strict SHA `d9461f7a21fa13454eca1f4371488431cf833a08ee758990c2dcb999b342600e`为外部信任起点。worker只实现和synthetic RAM/tempfile测试，不运行生产analyze/verify。

## 边界与固定分母

显式必填report路径/SHA与receipt路径/SHA。先核两文件真实SHA、完整schema/status、report关联、164/80/四cell/656API/1968frames、RR-all-native-IO旗标、全returned/validated、model前中后不变及科学门关闭；缺/partial/wrong receipt拒绝，且此时不打开任何数组。receipt必须为`agent_p5_early_history_factorial_20261029_v1.consume`、status=`all164_early_history_four_NEW_cells_and80_pairs_verified`。

上游native输入身份/causal构造/模型loaded bytes/历史source/label-before-seal chronology由这份完整strict认证；本层**不重新认证**其输入/物理/journal时序。不是第二份fullstrict，也不把reported flags当本层重新执行所得事实。报告`verification_boundary`明确此限。资源计数与耗时为继承值，新增NN/encoder/AD/physics/training均0，FLOPs=null。

独立重建固定schedule公式：四层顺序(0off,0on,1off,1on)，每层41枝，h=.01/.001，10坐标；minus=41*s+1+20*j+2*k，plus=minus+1，共80。全164行和全部4cell AA/AR/RA/RR保留，不按效果/zero/层选择。生产CLI强制164，没有缩分母参数。

每branch按call index4*i+j核full3输出指针与row一致、T3/returned/units；每cell完整解visual(1,3,256,384)、None、proprio(1,3,256,16)，检查全3槽float32 finite/文件SHA/bytes/inventory/单NPY EOF，禁止多cell别名。计分只取visual最后slot2。共同标签完整batch2(2,256,384)独立解码，只取slot1；不错误用actual15、proprio或full3早输出评分。完整label来源语义继承外部strict。all_slot_evidence保全部164×4个full3视觉/proprio对及164个label descriptor，不生成不存在的新预测。

每cell C-order float64 MSE；visual=((RA-AA)+(RR-AR))/2，proprio=((AR-AA)+(RR-RA))/2，interaction=RR-RA-AR+AA，joint=RR-AA；独立重算feature-effect L2并与report逐枝score exact相等。未复用任何producer arithmetic。

80对完整predicted/observed L2、dot、cosine、response-error、/2h central差分及response-vector factorial L2；任一范数exact0→null，零点积且非零范数cosine=0，不设epsilon。全部pair dict与producer report exact比较。再独立重算全体+四层的164/80/41/20分母、均值、效应、positive/negative/zero/null计数及全部六无向cell-pair的有向符号转移矩阵，与report全部summary exact相等。保24零或56非零的实际计数，不硬编码结果/过滤其他数值。

## 输出与MAIN命令

输出自有JSON包含arguments、rows/pairs/summaries、all_slot_evidence、已用文件/source/tests/doc bindings与verification_boundary，status=`all164_fourcell_and80_pair_independent_scalars_verified`。verify重做本层全部saved算术并要求整个dict一致，仍不调用fullstrict。cap32MiB/floor200MiB/wall900s；每41枝group释放array cache（约170MiB final features+label，不缓存656full3），文件只读，输出排他创建、fsync。全部实际输入SHA末尾重核。P1/P5继续false，无科学修复/收益/新鲜根声明。

ROOT=`/home/gbk/code/LeWorldModel/le-wm-main`，MAIN执行（若实际output目录名不同，显式替换路径但SHA仍须一致，不glob猜测）：

```bash
M=experiments.spectral_reachability.agent_p5_factorial_independent_scalars_20261029_v1
A=results/agent_p5_factorial_independent_scalars_20261029_v1_MAIN01.json
.venv/bin/python -m "$M" analyze \
  --report results/agent_p5_early_history_factorial_20261029_v1_cuda_MAIN01/report.json \
  --report-sha 4fc86a7fdc69fbe95003e417ee38e3a4d430360a8eb5bd0c89f73ee845926d0f \
  --receipt results/agent_p5_early_history_factorial_20261029_v1_consume_MAIN01.json \
  --receipt-sha d9461f7a21fa13454eca1f4371488431cf833a08ee758990c2dcb999b342600e \
  --output "$A"
.venv/bin/python -m "$M" verify --analysis "$A" --analysis-sha ACTUAL_ANALYSIS_SHA
.venv/bin/python -m pytest -q tests/test_agent_p5_factorial_independent_scalars_20261029_v1.py
```

轻量fixture使用nominal+完整signed pair共3枝/1对的synthetic真实shape NPY，只改本新模块N/schedule；另有原164/80/fourstrata全量纯RAM测试。测试独立手算常数MSE/效应、±/zero/null、所有full3槽finite、wrongcell/alias/missingreceipt/labels/proprio/篡改拒绝、真实排他writer→analyze→verify。没有导入producer以制造expected results；测试不是实际科学证据。尚待MAIN审读和真实saved执行，不预填效果。
