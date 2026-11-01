# GRASP own RGB companion：原两根 × cp5/10/15

仅新增本 stem source/tests/innerdoc，不改冻结 planner、physical、consumer 或结果。worker 只做实现、RAM 回归和获准的已封存输入只读 ABI 检查，不运行生产 prepare/consume、NN、CUDA 或 physics。MAIN 审读后运行 HOST CUDA。这里是**动作封存后的后验诊断，不冒称事前预注册方法**；不是官方 GRASP 复现、等预算比较、P1/P2 科学准入或独立根确认。

## 六槽与真实 authority

固定 root0/off 的 cp5/10/15，随后 root1/off 的 cp5/10/15；完整分母六槽、两计划、两 known roots、零 fresh。即便 native success/termination 提前发生，也只按真实 step 是否存在保留 source/null；不补帧、不改 checkpoint、不编码更早 terminal 代替、不按 success 挑槽。包括 warmup 内 early stop：若未进入原 root，三个 actual 槽全部 null，原 model 预测仍保留。完整两计划最多6 encoder、6 projector；predict/API/action_encoder/predictor/pred_proj/AD/physics/optimizer 均零，无训练、Q 拟合或新目标编码。

物理输入来自 `agent_p2_history_grasp_physical_20261029_v2` **自己的** schema / strict：

- source SHA `4eddd161df17f506916db3bcb7a4ca39c20b091cc0539ef4db6276b055267411`。
- actual report `results/agent_p2_history_grasp_physical_20261029_v2_MAIN01/report.json`，SHA `d71588499aae674790ba95107650116b12aae07b5c17c718d432f003451534e6`。
- actual strict `results/agent_p2_history_grasp_physical_20261029_v2_consume_MAIN01.json`，SHA `17373c3d9878d2bfab9eebfa6ac16c11e8ebf2736d8b77c4d206b5a45cf4957e`。

以上都以外部 CLI 参数传入，不把它们做成默认 authority。`build/verify` 先验 source/report/receipt SHA，再完整调用实际 `physical_v2.consume`，以 canonical JSON 字节比较整个 receipt，然后重建其 prepare（继而完整重建 repaired GRASP v2 consumer）。不把 GRASP 伪装成 wallcapped planner/receipt/edges。

模型预测直接读取物理 prepare 指向的原 `agent_p2_history_joint30_grasp_cuda_20261029_v1_cuda_MAIN01` **root0.json / root1.json 的 whole[0:3]**，不是只重复末态，也不假定 model_plan 存在 contexts。原 report 的两行 result_file 映射、artifact SHA 和 physical prepare bindings 必须一致。逐根核 root/task/commands/logits/末态完整 context/terminal_Q 对实际 model_plan；逐边核 full3 layout、latent shift、原归一化 action history。source slots 使用 model_input/root_context、blind_pointer、warm_raw10 的真实 ABI。

逐 checkpoint 以实际 plan digest/raw15、task/goal/condition、same-step trajectory/response/state/snapshot/PNG 连结。history10 为原 slot warm_raw10 加固定 raw15 到该步后的最后十步。source PNG 必须是原真实 off frame，原 bytes/pixel hash 不变。native state/reward/成功轴仅评价，不是 encoder/planner 输入，也不做 outcome selector。

## 编码、严格消费与计账

复用已冻结 `agent_p2_contextual_physical_rgb_encode_20261028_v1` 的原始低层 encoder/Trace/guard/像素读取，以及冻结原224 preprocessing 与完整 encoder/projector。模型原 policy/config/权重和 state_dict hash 全闭合、运行前后相等。只调用 `model.encode({'pixels': x})['emb']`，batch1，native hidden `[1,257,192]`，原 CLS projector 输出192。禁止改成197 tokens、换 projector 或读取物理 state 作特征。

每槽保存原 pixels NPY、preprocessed NPY、encoder/projector 完整 IO NPY、latent NPY、逐 intent/returned/guard 事件与 hash census。运行与消费都独立复算 **原 pixels → 冻结 preprocessing**；消费再检查 preprocessing→encoder→CLS→projector→latent 的完整数值/shape 接线。精确事件序列拒未知事件、缺返回、跨槽混用与未引用文件。缺失槽维持原 null，无去重/重试/替代。partial incident 保存未成功状态和错误，不当作 strict 成功。

CUDA only；TF32/autocast off、MATH SDP；wall600s、输出64MiB/最终8MiB reserve、磁盘 floor200MiB、metadata128MiB、GPU peak allocated3GiB/reserved4GiB。expected_calls 根据六槽实际 PNG 存在数事前固定，实际 attempted/completed 从 native hooks 逐次计账，绝不声称是 equal FLOPs（unknown total_FLOPs=null）。

own `.consume.rows[6]` 每行输出 model_Q、real_Q、model_minus_real_Q、prediction_vs_encoded_MSE、source、原 native_metrics/response；全部 Q 为原 goal192 下 float64 identity192 **mean** squared distance。缺失 real_Q/error 为 null，model_Q 不删除。六项是同时间 model-vs-real 对照，不生成不存在的 joint/composition 两臂 pairs。保留物理 counts、原 planner resources/wall 原样，不把继承资源冒作新编码调用。

## MAIN CLI（worker 不运行）

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
MOD=experiments.spectral_reachability.agent_p2_history_grasp_rgb_20261029_v1
.venv/bin/python -m "$MOD" prepare \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_history_grasp_rgb_20261029_v1_prepare_MAIN01.json \
  --physical-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_history_grasp_physical_20261029_v2_MAIN01 \
  --physical-report-sha d71588499aae674790ba95107650116b12aae07b5c17c718d432f003451534e6 \
  --physical-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_history_grasp_physical_20261029_v2_consume_MAIN01.json \
  --physical-receipt-sha 17373c3d9878d2bfab9eebfa6ac16c11e8ebf2736d8b77c4d206b5a45cf4957e \
  --physical-source-sha 4eddd161df17f506916db3bcb7a4ca39c20b091cc0539ef4db6276b055267411

.venv/bin/python -m "$MOD" verify \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_history_grasp_rgb_20261029_v1_prepare_MAIN01.json \
  --prepare-sha "${GRASP_RGB_PREPARE_SHA:?actual prepare SHA}"

.venv/bin/python -m "$MOD" run --authorize-host-cuda \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_history_grasp_rgb_20261029_v1_prepare_MAIN01.json \
  --prepare-sha "${GRASP_RGB_PREPARE_SHA:?actual prepare SHA}" \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_history_grasp_rgb_20261029_v1_cuda_MAIN01

.venv/bin/python -m "$MOD" consume \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_history_grasp_rgb_20261029_v1_cuda_MAIN01 \
  --report-sha "${GRASP_RGB_REPORT_SHA:?actual RGB report SHA}" \
  --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_history_grasp_rgb_20261029_v1_consume_MAIN01.json
```

## 测试（非真实编码证据）

`.venv/bin/python -m pytest -q tests/test_agent_p2_history_grasp_rgb_20261029_v1.py`

首轮49 passed /4.83s；增加 source/report/receipt 拒绝及真实磁盘消费者的 partial/全 missing roster 后，**55 passed /5.34s**。包括六槽完整 native hook/guard 回放、禁止 predict/AD/action_encoder、257 hidden、原始 whole 三边/动作/context/目标/task/receipt 变异、pixels→preprocessing 独立复算变异、未知/重排/缺失事件拒绝、partial 无重试、全部六槽/早停缺项/零编码的真实 Store→NPY/JSON→full consumer 回归。另只读核了 MAIN 封存 report/receipt/prepare/root result 和原 PNG 的实际 ABI，未执行 full production consume/prepare 或任何真实模型。
