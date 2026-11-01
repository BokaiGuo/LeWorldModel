# P2 three-program：12个真实RGB检查点与同时间三层join

本模块只做已封存四条raw15计划的后验测量。固定2 known roots × 2 arms × cp5/10/15，完整12分母；不是新root、前瞻科学放行或P2机制证据。没有physics、预测、goal重编码、训练、拟合、择词或重规划。

## 权威与时间

prepare必须接收实际physical source/report及完整持久化consume回执SHA；调用原physical.consume重建并逐字典精确比较整个回执，随后重建原physical.prepare和原planner.prepare。不能使用status字段或另一种wrapper envelope替代。输入SHA均来自CLI，不预设尚未发生的结果。

每槽绑定原selected plan/execution seal、task/blind/source/model/normalizer、raw15位、对应edge witness/input/output context、实际checkpoint step、canonical off overlay0 PNG文件及像素SHA、同一步native7/response/reward/done。cp5和10的模型Q从各自edge.output_context.latent_history[-1]计算；只cp15核对原predicted_Q。不以终点Q冒充早期预测。实际Q和模型Q都沿用原goal192、float64 identity mean squared error；不把latent Q和native距离合并。

三层输出：model_Q、actual_encoded_Q、prediction_vs_encoded_MSE，原native_metrics及response并列；model_minus_actual_Q仅是预测误差诊断，不是任务收益。完整原图、预处理、encoder257×192所有tokens、CLS/projector输入输出和latent NPY保存。没有使用native标签选择输入/动作，编码调用只传真实RGB。

原native_success和原reward/angle定义由原native.metrics(saved vectors)复算，不创coverage阈值。native7最后两项是agent velocity；不把agent惯性差异视作物体能力。完整物理strict已认证其原生状态/目标/历史，不借用旧endpoint。

干净native_stopped允许；尚未到达checkpoint时source/native_metrics/实际Q为null，保留该槽及对应模型预测，禁止以提前终态替换。engineering partial或缺strict拒绝prepare；不伪造12图。没有任何去重：完全相同像素也各调用一次。

## CUDA与资源

仅MAIN授权run才import torch/load模型；固定原LeWM checkpoint和原224预处理/native192 ABI，CUDA:0，inference_mode、autocast关闭、TF32关闭、SDP MATH，结束恢复TF32。原权重before/after严格相等，完整输入闭包终态复核。无CPU fallback。

全12图时encoder12 API/12 samples、projector12；其余action_encoder/predict_API/predictor/pred_proj/AD及新physics均0。缺图按实际可用数调用，不补预算。Trace before-dispatch拒绝额外调用；独立原native guard限制CUDA batch1与禁用predict/AD，整个chronology由saved consumer重建。尝试失败不重试、不退款，保留partial report/数组。

继承planner actual_resources、物理counts及模型walltime独立列出，不和本模块成本相加，不将嵌套encoder/projector当额外独立样本。总FLOPs未知null，不宣称equal_total_compute。墙钟600s、输出64MiB含8MiB终态预留、disk floor200MiB；metadata单文件128MiB，完整bindings不省略。native_seal先落盘，再saved分析。仅成功consumer输出三层join；RAM测试不代表CUDA或科学结果。

## MAIN准确命令

工作目录 /home/gbk/code/LeWorldModel/le-wm-main。三个输出路径必须不存在；保留所有失败，不覆盖。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_three_program_rgb_20261029_v1.py

OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_three_program_rgb_20261029_v1 prepare \
  --physical-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_three_program_physical_20261029_v1_MAIN01 \
  --physical-report-sha 49b682d0ca326590c88ef69d74eecfdd9d5c19f21667e790a7c1458aac98e064 \
  --physical-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_three_program_physical_20261029_v1_consume_MAIN01.json \
  --physical-receipt-sha 95c967c95c8ff00e0c2d4ab5c801afbab708095479c96986319dc103933208ac \
  --physical-source-sha b0e7ae15d4810963a64916b1da16e34ae1017b189156a93b635d73520e2666e5 \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_three_program_rgb_20261029_v1_prepare_MAIN01.json

# RGB_PREPARE_SHA由上一命令实际返回；不填写未来SHA。
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_three_program_rgb_20261029_v1 verify \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_three_program_rgb_20261029_v1_prepare_MAIN01.json --prepare-sha "$RGB_PREPARE_SHA"

OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_three_program_rgb_20261029_v1 run \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_three_program_rgb_20261029_v1_prepare_MAIN01.json --prepare-sha "$RGB_PREPARE_SHA" \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_three_program_rgb_20261029_v1_cuda_MAIN01 --authorize-host-cuda

# RGB_REPORT_SHA取actual run返回。
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_three_program_rgb_20261029_v1 consume \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_three_program_rgb_20261029_v1_cuda_MAIN01 --report-sha "$RGB_REPORT_SHA" \
  --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_three_program_rgb_20261029_v1_consume_MAIN01.json
```

## 测试与限制

RAM使用明确标为double的NumPy callback/原Trace与guard验证完整12次调用、张量保存、hook恢复、禁止预测/AD/197token、缺帧分母及action/time/model/source/事件/数组篡改。另仅读取实际physical report/完整receipt与PNG，检查真实schema和逐步Q接线；saved build→verify回归也不调用模型或环境。测试不是新科学证据。worker未执行生产prepare、CUDA或physics；MAIN审读后执行。旧source和results不改；P1/P2/scientific gate始终false。

