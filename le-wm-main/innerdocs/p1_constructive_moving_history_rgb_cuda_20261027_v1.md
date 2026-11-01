# Moving-history physical RGB evaluation adapter v1

独占新stem `p1_constructive_moving_history_rgb_cuda_20261027_v1`；只新增source/tests/本doc。
MAIN审查、prepare和运行CUDA。实现者仅运行RAM/mock测试，不prepare、不执行actual consume、
模型或物理。不修改Huygens physical/source、moving encoder/planner或其他seal。

## 已确认接口与固定输入

Huygens schema由MAIN转达并对照已冻结源码/doc实现，不猜字段：
`p1_constructive_moving_history_physical_20261027_v1.report`，
`status=moving_history_physical_recorded`。
rows固定root-major4行，只有root_slot，无source_slot；关联键是
`(root_slot,arm,selection_digest)`。journal branch号通过branch_started的完整plan匹配取得，
不是用任务slot冒充branch号。四臂固定：

```text
slot0 line_search_GN          prefix5, terminal10
slot0 full10_conditional_D2P prefix5, terminal10
slot1 line_search_GN          prefix5, terminal10
slot1 full10_conditional_D2P prefix5, terminal10
```

checkpoint是phase=plan、step=0/5/10的receipt：
png/file_sha256/pixel_sha256/state/snapshot/observation_state；PNG路径相对physical output。
moving guard没有旧版png/file SHA字段，因此使用新数组join，不调用旧generic assemble或改tag凑接线。
必须具备真实step5和10，任何partial/native提前停止、丢帧或无receipt都拒绝，不补帧、不改分母。
native成功标签不用于选择图像；完整10步终态成功也保留编码资格。

物理报告：`results/p1_constructive_moving_history_physical_20261027_v1_main01/report.json`
SHA `a5322dc4f483e82b5ed9d7b3996cf8a63845261d65774f90a4ae341773016aaf`。
物理prepare SHA `226749e2bf8acdcd7a4a4a08be7336445daad0b6ad12185b24cbc0fb4560b1ff`。
planner报告：`results/moving_history_planners_cuda_20261027_v1_cuda_main01/report.json`
SHA `cd8432c06288577d25d9d9b92c7eb3673de592499ee64cae93238156f3f19aa0`。
以上是MAIN提供的实际外部绑定，不代表实现者完成独立saved审计或实际执行。

两个新task路径：
`results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_00.json.gz`，
SHA `a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b`；
`task_01.json.gz`，SHA `101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74`。
直接调用新moving encoder `load_task(path,sha)`，绝不调用zero-only旧load_export。
原192 projected CLS、identity mean-Q、goal、非零raw动作归一化history、model/normalizer合同不变。

源码pins：

- moving physical `44d90599eab881c773d5efa09177dfe6b395fa3f4eef41070340681fea5ec96e`
- moving encoder `246b9194fd31c874d7f0da67c4dec17365f93294e6ed460fe46e2509673e1b63`
- moving planner `f8f1d5507e2474fbbbfa56cae98f7bfa99f5929829946a2d7760c2c30ccc03c1`
- generic RGB v2 `e1448739bbae827a2e3b1f62bf50f0f27aa06d47aef60163483fa289e6eb432e`

## 身份链与不读future选根

build由MAIN执行：固定planner report外部SHA→own moving planner consume/verify→完整四个
selected plans与全部原artifact SHA，再读取两个实际task（新load_task）。
physical report外部SHA→execution_binding→外部prepare SHA→manifest SHA→
显式compiler handoff JSON pointers→consume plan，按slot/arm/selection关联，不能猜report行号。
复用moving physical.selection的saved-only指针核验，绝不调用其material/verify/load_source/branch。
原sourcepack、source prepare及其他evaluator-only依赖仅hash bytes，不解析证书或witness。
所有父physical/model/task/source/manifest/handoff与新三文件形成闭包，最终再次核hash。

MAIN实际handoff `results/moving_history_planner_handoff_main01.json` 是consume计划投影：
`projection.omitted_plan_fields=["prediction"]` 且source_report_sha256必须对应指定原planner。
适配器逐字段比较所有保留内容（包括commands、logits、task/blind、scalar cost和selection seals）；
不会要求handoff含大prediction，也不会在删字段后的对象上重算原plan selection_sha。
完整prediction来自原封存planner own consume，其原plan digest与所有artifacts由own consume验证。
如果handoff保留prediction，则要求完整原plan精确相等。省略更多字段或投影来源不匹配就拒绝。

新assemble核完整四臂、原journal首项prepare、末项status/counts/errors和chain tail，
全操作嵌套attempt/completed计账。必须是8reset/80public/88render/8goalrender/16setter/
816physics/4goalbind完整分母。每臂warmup0..10、plan0..10全22个fresh receipt及guard对应；
记录先于guard且state/obs/snapshot一致。moving raw10 bits、三历史PNG与task像素hash一致；
root+goal identity、goal attachment及所有10条own command/state/goal与receipt一致。
PNG文件及像素SHA都核验。state只做既定身份/协议匹配，不用于计划、根、arm或终点筛选。

## 新编码与指标

复用封存底层PNG decoder、Store、Ledger、模型guard和NPY写入；不重跑任何planner。
run先own verify，再独占output及binding。真实HOST CUDA、MATH SDPA、matmul/cudnn TF32 off、
inference_mode、冻结参数、同checkpoint前后tensor SHA、每模块GPU同步，无CPU神经fallback。
固定8次encoder+8次projector（batch1），0predict/actionencoder/AD/physics/训练。
即使图像相同也不去重。输出每张原生float32[192] NPY，8行分母预置，逐次durable
image_attempt/module_attempt/module_return/image_saved；失败停止，不retry或替代未开始图像。

actual_Q = mean((observed_z-goal)^2)，root_Q同原moving root latent，
actual_Q_change及actual_vs_root_mse均保留。prefix与terminal预测取自同commands的原planner
terminal context（含严格H2历史），只用task core数组合同校验，无新模型调用。
记录predicted_Q、predicted_vs_real_mse、actual_minus_predicted_Q和改进方向符号是否反转。
不是将真实state拟合到latent，也不把缺预测当零；原selected seal缺预测则拒绝。
实际编码结果只衡量这些固定样本的模型/视觉误差，不预设方法或理论负结论。

cap2MiB、另保留200MiB、final128KiB；只存8NPY/ledger/binding/report，不复制PNG/权重。
prepare<256KiB，独占写，无大闭包复制。errors/partial完整，模型没加载时unchanged=null；
最终模型/来源、artifact listing/hash与report分别尽力保存，first_error不被secondary覆盖。
新report schema为本stem `.report`，完成状态 `moving_selected_rgb_encoded_not_admitted`。
原planning/8次原始history-goal编码/真实物理成本另记，不退款、不宣称全算力匹配。
P1/P2 admission=false，沿用recipe的两新moving任务不是新独立样本。

## MAIN最小命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`，output均须不存在：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_moving_history_rgb_cuda_20261027_v1.py

.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_moving_history_rgb_cuda_20261027_v1 prepare --output results/p1_constructive_moving_history_rgb_cuda_20261027_v1_prepare01.json
```

设置 `MOVING_RGB_PREPARE_SHA` 为真实prepare回执值后：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_moving_history_rgb_cuda_20261027_v1 verify --prepared results/p1_constructive_moving_history_rgb_cuda_20261027_v1_prepare01.json --prepare-sha "$MOVING_RGB_PREPARE_SHA"

.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_moving_history_rgb_cuda_20261027_v1 run --prepared results/p1_constructive_moving_history_rgb_cuda_20261027_v1_prepare01.json --prepare-sha "$MOVING_RGB_PREPARE_SHA" --output results/p1_constructive_moving_history_rgb_cuda_20261027_v1_cuda_main01
```

## 本侧验证界限

仅RAM数组fixture：复用Huygens ArrayEnv/MemoryStore（非模拟器）生成合法receipt，建立新moving
task和原生H2 seal、投影handoff。测试缺帧/guard/命令/目标/history/资源/身份故障拒绝、
新load_task明确调用和旧loader禁用、8图无去重计量、float32 NPY与meanQ/prediction差值、
partial不推进、prepare磁盘门及run验证先行。不会读取actual报告执行审计，也没有实际prepare/result。
实现者不运行神经或物理；真实结果由MAIN运行记录决定，mock通过不等于科学准入。

本侧38项RAM tests通过。应MAIN后续明确要求，另完成实际saved输入的只读build smoke：
8图顺序精确、164项closure绑定、序列化prepare为140464 bytes；只在内存中构建，
未创建prepare/result、未加载模型、0新神经/物理。这个smoke是执行接口接线验证，
不是独立科学审计，也不是新RGB编码结果。
