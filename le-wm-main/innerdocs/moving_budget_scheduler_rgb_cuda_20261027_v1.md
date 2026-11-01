# Scheduler moving RGB：固定八分支实际图像编码

仅新增本stem source/tests/innerdoc。MAIN独占实际prepare、HOST CUDA与科学结论；worker只做RAM/mock测试，不运行神经/物理。不修改physical、scheduler、旧RGB白名单或任何封存结果。采用experiment-code的预声明合同、资源日志及故障测试；不执行训练、调参、拟合Q或结果筛选。

## 输入与封闭authority

固定原moving两根，顺序root0、root1；每根固定四策略：
`U_uniform,D_geometry,S_swapped,N_no_signal_uniform`。8分支各取真实plan阶段prefix5/terminal10，两图不得去重或替代。不是新鲜样本/独立确认，不因native成功与否丢弃策略。

MAIN给定实际physical：

- report：`results/moving_budget_scheduler_physical_20261027_v1_main01/report.json`
- report SHA：`c9169b685e4e7f7d79ea2101aa81ad55b95593f96d6158ed8ff0815de136bd70`
- prepare SHA：`bddd73792da6ecabcc669e75d9c466940f763571990498f3363375336c7d6344`
- physical source SHA：`60ad315b96c2c9eda21cc79fe4b07710eafd6d16027d8b0ff46641bb794e3f1a`

本实现固定最后一项；report/prepare SHA必须由CLI显式传入，不根据扫描目录自动选结果。保存完整arguments和bindings。原sourcepack c2d8d573…、原task00/01 exports及SHA沿用冻结moving loader；真实非零raw10 history仍前5条[.1,0]、后5条[0,.1]，不接zero-only旧schema，不二次除100。

ownconsume只允许冻结`moving_history_budget_scheduler_cuda_20261027_v1`，其source/tests/doc认证由冻结physical ownconsume执行。独立重建8个physical plans，逐项比较原prepared plans，包括root/task/blind/task_contract、commands、compiler report/selection、own_plan及physical selection digest。沿用原source.digest的普通JSON空格canonical，不混用RGB紧凑canonical；whole_run_report_sha256在model_plan原selection seal之外，但在own_plan_sha256中保留。禁止任意module动态导入或放宽旧白名单。

完整物理journal链、首尾prepare/counts、8branch starts/closes、所有22帧/branch、freshguard、snapshot/state/observation、实际warmup raw返回、goal attachment与后续goal、10commands和5/10 checkpoint全部验证。核全16reset/32set_state/16goal_render/8goal_bind/160public/1632physics/176explicit_render/8construct/8close，attempted/completed均完整。physical的ctor可观测性边界不由RGB改写。闭包只hash source/certificate依赖，不从证书挑控制或看未来候选。缺任一真实checkpoint/branch或partial物理直接拒绝，不能补图/降低分母。

## 固定神经与存储预算

16原PNG → **16encoder + 16projector**；每图原projected CLS float32[192]。不重编码原root/goal；predict、action_encoder、AD、训练、物理均0，guard禁止这些接口。原模型CUDA-only、eval/frozen/inference，SDP=MATH、matmul/cudnn TF32均false，无CPU神经fallback，模型tensor SHA运行前后相同且等于原checkpoint预期；来源在最终阶段再次verify。原source/goal编码、scheduler和物理成本分别继承，不退款、不称总算力匹配。

16MiB整个新output硬cap，200MiB floor，512KiB final reserve；启动需至少216MiB空闲。prepare<256KiB（只小manifest/bindings，不复制大闭包），也保留200MiB。写PNG、NPY、journal、binding、final report都计入cap；每次文件fsync及目录fsync。原PNG单图最多1MiB，实际若累计输出超cap，保留partial并失败，不放宽cap。

**300秒只约束encode_images循环**，不包括prepare/verify、import、模型加载、final model/source校验。循环每图前与保存后检查；同步CUDA调用不可抢占，若单调用超时是在返回后拒绝，不是全进程300秒硬deadline。

每图7条正常events：image_attempt、raw_image_saved、encoder attempt/return、projector attempt/return、image_saved，共112条。原PNG逐byte副本在neural dispatch前落盘；模块attempt/completed独立计数，失败不重试，16rows全部预置，保留not_started/failed_partial与首错/secondary finalization errors。module completion含同步；不声称可精确细分encoder内部已返回但projector失败的所有内部工作。

正常目录恰35个普通非symlink文件：binding.json、events.jsonl、16PNG、16NPY及report.json（34artifact+report）。NPY在np.load之前先核字节hash、NPY1.0 header的float32[192]/C-order和精确header+768字节长度；再以allow_pickle=False读取，核有限值与读取结束位置，拒绝尾随数据/第二数组。consume核exact目录census，不接受多余文件或遗漏文件。

## 指标与Poincare comparison接口

每行保留provenance（root_slot/arm/selection_digest/own_plan/原task/goal/modelcompiler身份）、原PNG file SHA/pixel SHA与native192 NPY。

`actual_Q = mean((actual192-goal192)^2)`；`root_Q`使用批准原root latent和同goal；同时保存actual_Q_change、actual_vs_root_mse。terminal只与同root/arm/selection的prefix做`rawpixel_vs_prefix5_mse`（uint8原单位、float64算术）及actual_Q_change_from_prefix5。prefix对自身这两差为0。物理pose/native指标与latentQ不混单位。

不重新预测、不猜latent：predicted_Q、predicted_vs_real_mse、Q_improvement_sign_reversed均null，prediction_status=unknown_not_recomputed。未知不是0或无改善。

comparison在RGB尚未实际跑时必须保持**missing**，不能从物理state/其它策略或旧RGB填补。实际后接新adapter：
`consume(output, report_sha)`，CLI明确`consume --report-sha`（绝不用prepare-sha代report SHA）。成功schema为`moving_budget_scheduler_rgb_cuda_20261027_v1.consume`，status=`scheduler_rgb_verified_not_admitted`，errors=[]，16rows，固定8/16分母；row含root_slot/arm/role/selection_digest/task_id/root_sha256/compiler_report_sha256/metrics/rawpixel/latent。该ownconsumer只读saved数据，重核完整prepared closure、35文件、112events和指标；不调用神经/物理。可由MAIN转交Poincare；worker没有直接thread消息工具，不假称已发送。

producer成功status=`moving_budget_scheduler_rgb_encoded_not_admitted`，否则partial_or_failed。P1/P2/P3始终false；仅两knownroots描述性评价，不作准入/普适/理论因果/计算优势结论。

## MAIN命令（worker未执行）

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。output均必须全新。

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_moving_budget_scheduler_rgb_cuda_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.moving_budget_scheduler_rgb_cuda_20261027_v1 prepare --physical-report /home/gbk/code/LeWorldModel/le-wm-main/results/moving_budget_scheduler_physical_20261027_v1_main01/report.json --physical-sha c9169b685e4e7f7d79ea2101aa81ad55b95593f96d6158ed8ff0815de136bd70 --physical-prepare-sha bddd73792da6ecabcc669e75d9c466940f763571990498f3363375336c7d6344 --physical-source-sha 60ad315b96c2c9eda21cc79fe4b07710eafd6d16027d8b0ff46641bb794e3f1a --output /home/gbk/code/LeWorldModel/le-wm-main/results/moving_budget_scheduler_rgb_cuda_20261027_v1_prepare01.json
.venv/bin/python -B -m experiments.spectral_reachability.moving_budget_scheduler_rgb_cuda_20261027_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/moving_budget_scheduler_rgb_cuda_20261027_v1_prepare01.json --prepare-sha RGB_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.moving_budget_scheduler_rgb_cuda_20261027_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/moving_budget_scheduler_rgb_cuda_20261027_v1_prepare01.json --prepare-sha RGB_PREPARE_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/moving_budget_scheduler_rgb_cuda_20261027_v1_cuda_main01
.venv/bin/python -B -m experiments.spectral_reachability.moving_budget_scheduler_rgb_cuda_20261027_v1 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/moving_budget_scheduler_rgb_cuda_20261027_v1_cuda_main01 --report-sha RGB_ACTUAL_REPORT_SHA
```

RGB_PREPARE_SHA与RGB_ACTUAL_REPORT_SHA分别由MAIN实际新文件hash填写，不是已有结果。测试使用RAM PNG/NPY/fake arrays、mock模块entry及failure，无tmp文件/actual build/prepare/模型/物理。goal API仍存在用户已撤销旧review的blocked未完成目标，创建新goal被API拒绝；未虚假complete旧goal，也未把P1标complete。
