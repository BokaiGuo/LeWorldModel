# CEM8 baseline长表导出：消费严格分析，不重复分析器

独占 source/tests/innerdoc 同名前缀；旧八anchor物理adapter、CEM/teacher producer、Hegel分析器、封存results及共享文档均不修改。不创建后代，不执行CUDA/physics/神经/训练。按experiment-code技能明确来源、计量单位、失败测试；用户边界优先，不生成额外研究实验或图。

## 消费边界

唯一指标权威是 Hegel 的 `agent_CEM8_physical_analysis_20261028_v1.analysis`，status 必须为 `saved_fixed8_analysis_not_admitted`。该分析器已经负责两个完整physical.consume、全8原anchor/全24 variants来源验收、native指标独立复算及终止语义。本模块不导入/调用分析器、physical.consume、producer.consume或任何模型/env；源码仅标准库。测试借用Hegel RAM fixture生成其真实格式，不将fixture当科学证据。

输入是实际保存analysis绝对路径和MAIN独立复算SHA、实际analyzer source SHA、实际CEM与factorial v2 compiler report路径/SHA。拒绝原始producer报告冒充分析、裸physical结果、observed4、缺SHA、重复JSON key、NaN、symlink/路径别名、超32MiB输入。检查analysis.input_sha256中的**整个**source/task/artifact闭包和对应analyzer source；两个compiler报告必须已在该闭包内，且SHA等于analysis完整consume返回中的compiler_report_sha256。读取前后重验闭包与analysis SHA，不生成替代分析或future hash。

本模块不以外部保存analysis的自报status替代MAIN实际审核：MAIN必须先完成Hegel真实分析及其保存SHA验收，再运行导出。SHA绑定证明精确消费哪个产物，不单独证明科学正确。无法取得实际analysis时只交付实现，不能伪造真实导出或声称完成实证分析。

## 长表与可复算性

固定顺序：2 root_slot × off/on × line_search_GN/full10_conditional_D2P 原prefix arm，选teacher24中的ordinal `3*i+1`（teacher_full），不是另两variants。每anchor两method (`cem`,`spectral_teacher_full`) × 两endpoint_kind (`endpoint`,`horizon10`)，共32行，不因失败或earlystop删行。8 anchors仅2个已知根，不是8个独立根，不fresh。

每行保留task/原selection/anchor SHA、所消费physical/compiler报告SHA、完整selected-plan canonical SHA、model_Q、状态、错误、missing checkpoints与实际suffix_returned_steps；suffix_attempted_steps存在就照录，缺失=null。load后每行再附analysis/analyzer source SHA，CSV可追溯原JSON。指标逐字段投影Hegel的joint_xy_error/native7_distance/native7_reward/agent_xy/object_xy/velocity/wrapped_angle/joint_xy_wrapped_angle，以及原生native_success与wrapped_joint_success；不另算指标、不修改阈值。state7末两维按原native定义为agent速度，不是block速度。model_Q不是实际距离或成功率。

`endpoint`是Hegel认可的实际早停或完整终点，`horizon10`只允许实际suffix5；早停没有H10，null不变0/false。identity_unknown必须保持未评分；engineering failure保留错误、分母与observed_endpoint_metrics诊断，不能当成功endpoint。完整负结果不筛除。native多圈角度等语义保留Hegel原结果，不在此改为其他success定义。

JSON主产物含32 rows和16唯一resource_rows；CSV是相同32 rows的投影，model_budget_axes为JSON单元格。CSV的null写字面`null`、false写`false`，嵌套错误/metrics按JSON保留，不用空字符串混淆。每method的budget因endpoint/H10重复出现，两行相同resource_row_id；`budget_repeated_across_endpoint_kinds=true`明确**不能按32行求资源总和**。应按16个唯一resource_rows读取。

## 分轴预算，不称等总算力

每branch从实际compiler报告的module ledger分别数predict API/action_encoder API/pred_proj API的attempted与completed；predictor_forward_samples按每次实际predictor输入 `[B,3,192]` 的B累加，不把sequence3误当3个样本。batch1的65API/65samples与批量3API/12samples是不同情况；AE/proj不是额外predict，不与API/samples相加。CEM原生row.resources须与ledger对应。

teacher_full只取自身plan_index的13条实际predictor账；其J/SVD按该case真实returned helper operations统计；数学VJP directions仅在明确原合同24J/4608directions时记每J192，**不是engine backward API调用次数**。CEM的J/VJP/SVD/AD-forward零来自实际producer合同的显式声明，不因未找到hook推零，缺字段即null。teacher AD-forward归属没有独立receipt计数则null，predict账仍包含它。backward engine/JVP/每branch encoder/projector/modelphysics/wall/FLOPs没有独立可分配记录时null，完整报告contract的明确全局零和其他已知值保存在resource_groups，不强行摊到branch。

resource_groups分别保留CEM8报告resources/原contract和wall（报告无值则null）、factorial_all24的实际ledger/312predict_API/原contract/真实wall。teacher8明确subset_of factorial_all24、add_to_parent=false，不再把104加到312；inherited_shared_once保留完整原shared_resource_axes声明与overlap说明，不分摊、不抵扣、不重复退费，总历史费用未知。两个真实物理consume的全局counts各保留一次，spectral24仍是24全部variant计量，不冒充teacher8单独物理成本。既有共享编码/原prefix/teacher上下文不是免费，只是不在此捏造全历史总账。

仅同轴计数；predict/API/samples/AD/J/VJP/SVD/encoder/physics/wall不混加，不要求各轴相等，不宣称equal_total_compute、equal_FLOPs、freshness、P1或独立确认。模块不生成胜负结论或按outcome改配置。

## MAIN调用

先保存并审核Hegel真实analysis；本worker不代跑其分析器。以下三个变量由MAIN填**实际已有绝对路径及复算SHA**，不能填未来hash，也不能用本模块fixture代替。两compiler报告SHA已实际存在：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_CEM8_baseline_table_20261028_v1 \
  --analysis "${CEM8_HEGEL_ANALYSIS:?MAIN actual saved Hegel analysis absolute path}" \
  --analysis-sha "${CEM8_HEGEL_ANALYSIS_SHA:?MAIN actual analysis SHA}" \
  --analyzer-source-sha "${CEM8_HEGEL_SOURCE_SHA:?MAIN actual frozen analyzer source SHA}" \
  --cem-compiler-report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_moving_CEM8_20261028_v1_cuda_MAIN01/report.json \
  --cem-compiler-report-sha 660751b5784d25cce652d0ecb06ffb4cf3810a24ed135755606c97c3f957da0a \
  --teacher-compiler-report /home/gbk/code/LeWorldModel/le-wm-main/results/capability_suffix_factorial_cuda_MAIN_20261027_v2_MAIN01/report.json \
  --teacher-compiler-report-sha fc620c9fdcbb5046c1d076c79cd0703da999275ec921de799f3747c0204d69ab \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_CEM8_baseline_table_20261028_v1_MAIN01.json
```

要CSV，用同一输入参数改 `--format csv --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_CEM8_baseline_table_20261028_v1_MAIN01.csv`。每次都完整复验输入，不缓存“通过”。stdout返回实际产物SHA/32行数。JSON和CSV分别独占xb写，32MiB cap、200MiB free floor、fsync文件和父目录；拒绝覆盖任何现有输出、输入或写入两个物理producer目录。输出basename必须同新stem前缀。写失败保留partial，不删除、不回滚；不承诺JSON/CSV两次调用的跨文件事务。CLI不存在run/prepare/GPU授权入口。

Python接口：`load(analysis_path, analysis_sha, analyzer_source_sha, cem_path, cem_sha, teacher_path, teacher_sha)`只读返回完整JSON结构；`project(analysis,cem,teacher)`是RAM seam而非生产绕过验证入口；`csv_bytes(result)`做格式投影；`write_new(path,body,result)`写NEW同前缀文件。

## 已实际测试与未运行项

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_CEM8_baseline_table_20261028_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_CEM8_baseline_table_20261028_v1 --help
sha256sum experiments/spectral_reachability/agent_CEM8_baseline_table_20261028_v1.py tests/test_agent_CEM8_baseline_table_20261028_v1.py innerdocs/agent_CEM8_baseline_table_20261028_v1.md
```

首轮27 RAM tests passed：全32行/16budget、API与batch samples分离、null/false CSV、earlystop/H10、engineering failure分母、task/selection/variant/winner/资源/来源/闭包变异、重复JSON/NaN/实际SHA、exclusive写/floor/help。最终复测数与三文件freeze SHA见交付消息。保存compiler元数据可额外只读核对预算形状；它不是新神经/物理实验，也不能替代尚未提供的严格analysis。没有运行CUDA/physics、没有重新跑Hegel分析器；未取得实际analysis路径/SHA前，不声称真实长表已生成。
