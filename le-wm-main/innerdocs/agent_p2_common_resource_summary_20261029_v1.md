# P2 common-resource 六槽 saved-only 汇总 v1

独占新三件套，不改冻结runtime或其结果；不运行NN/CUDA/physics、不联网、不读物理outcome择参数。固定原2 known off roots×joint/composition/GRASP，共6槽，不删失败分母、不padding、不输出winner；P1/P2/fresh=false、equalFLOPs/equal_total_compute=false。

## 实际 authority / 输入 ABI

`build_summary(runtime_output, actual_report_sha, strict_receipt, actual_strict_sha)`：外部参数必须为MAIN实际终态文件/真实SHA，**没有未来SHA默认值**。先核冻结runtime三SHA（source `ea763d96eee1040652a3ea02badcb5e8e37b5781530da721f73d2a63d9cb147d`），实际调用其完整`consume`重建全部六槽strict；外部persisted receipt整个JSON精确等于该return，不只检查status。该ABI只接原runtime.consume完整return，若MAIN另加wrapper envelope须显式另审适配，不能把envelope冒充return。

这是**导入原strict作认证**，不是独立神经forward/梯度验证。重复验证实际源闭包、模型before/after、原数组与hashchain、optimizer/partial等属于原strict能力；本模块独立做身份join、saved float64 goal-Q与资源/missing分类，不伪称独立反传或新科学证据。summary.verify会重新完整consume，不是自证checksum。生产产物尚运行时不执行build/不轮询、不造receipt。

## 输出合同

逐槽root/task_id/task_file/SHA/完整source-slot摘要/model tensor hash/root history/goal/warm raw10/action-domain摘要；逐根三臂强等。原joint30球、float64 tanh→float32 raw15、原normalizer与identity192 mean-Q保留，实际export raw bits与last accepted/retained incumbent动作精确匹配。不同臂动作本来可以不同，不强求raw相同或改配置凑相同。

所有已strict发布计划的cp5/10/15原latent与同goal作**独立显式float64** mean squared Q，terminal与原reported_model_Q精确比较。缺计划则model=null，不能用incumbent或其它运行终点回填；谱臂保留完整incumbent及其Q64但明确not_export。GRASP保留最后完整unit与最后已接受commit（包括原journal pointer/动作字节摘要）、initialization不冒充完整更新；没有付费终态replay就不猜更新后的Q。

逐资源attempted/completed分栏：packet、实际predict API、predictor batch1 samples、AE/predictor/pred_proj、context tokens/projector rows、J、J输出mathematical VJP requested rows、J元素、AD primal实际packet与可用的requested steps、scalar backward、Adam/noise、SVD/ridge、encoder/physics等。原strict不存在的字段/null不补0；J数学请求方向≠engine实际backward，scalar backward也不能换算成完整J。已知0来自原strict，不由缺失推断。所有单位独立，永不求和成总算力。

1902 API与samples各列cap/实际/at_cap；resource_cap_stop与at_cap不同，预留6 replay造成拒绝不等于已经消耗1902。450秒同步搜索窗口列deadline_reached、overrun、stopped_on_wall（cooperative非抢占）。保留partial/incomplete/pending/实际error/缺replay/拒绝pointer；None照列六分母。

CPU SVD/ridge实际callback与capture事件计数并栏，**CPU_capture_seconds=null**（原生产没有分开计时，不能从event间隔扣减/冒充独立CPU耗时）。搜索scope包括CPU/capture/search IO；stage原timings完整保留initial verify/host setup/loader/post-search persistence/final identity/total，不能均摊成每臂；per-arm persistence=null。原继承四task encoder16/projector16 cohort仅列一次，off分摊仍unknown；没有重复退费。原始完整账本/数组均由bindings可追溯。

## MAIN CLI（worker不实际消费运行中新结果）

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。先将MAIN实际完整runtime strict return保存为JSON，外部SHA均取实际值。以下占位符不是虚构SHA，不存在文件时必须拒绝。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_common_resource_summary_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_common_resource_summary_20261029_v1 build --runtime-output ACTUAL_RUNTIME_OUTPUT_DIR --report-sha ACTUAL_REPORT_SHA --strict-receipt ACTUAL_FULL_STRICT_RETURN_JSON --strict-sha ACTUAL_STRICT_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_common_resource_summary_20261029_v1_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_common_resource_summary_20261029_v1 verify --summary /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_common_resource_summary_20261029_v1_MAIN01.json --summary-sha ACTUAL_SUMMARY_SHA
```

仅exclusive本stem输出≤16MiB、disk floor200MiB；不修改输入。读原report/prepare≤8MiB、完整receipt≤64MiB；journal用原有有界流读取。所有源/输入/receipt/report/原数组再次hash确认稳定。原runtime fatal未形成全六strict则拒绝成功summary，不能以本工具补签；真实问题由MAIN诊断后新版本处理。

## 交付验证与状态

以上RAM测试命令实际通过：27 passed in 10.51s。包含六槽partial账本、身份/Q/资源篡改、完整strict-return重建、CLI build→verify→tamper及exclusive输出测试；fixture不冒充真实模型或梯度证据。

MAIN已通知实际运行454.0318秒结束，report SHA `d781b493ecd624db846e3225ed0177bd3fe250d6ba9213d925b2a57956dd2242`，完整strict receipt为 `results/agent_p2_common_resource_grasp_cuda_20261029_v1_consume_MAIN01.json`、SHA `d3ee465f2fa6d31345f369f9514165c4817c51a29c3d00ffdd2d681e5829cc1f`。此处仅记录MAIN交接值；worker未运行该批真实summary消费，CLI仍要求实际runtime目录及外部SHA，不把通知当模块已验收。

RAM fixtures明确不是模型、梯度或资源实际运行证据；使用experiment-code技能仅落实结构化保存/回归，不采用其训练、调参、收益下限或图生成建议。生产summary待MAIN执行，不宣称独立样本总体效应。
