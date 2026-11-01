# Nonlinear suffix actual midpoint history3 intervention

## 固定范围与分析口径

仅已执行 nonlinear suffix v2 四槽 root0/off、root0/on、root1/off、root1/on，保留原 rawfloat32[10,2]、goal192、identity Q、模型/normalizer；0新物理、0训练、0AD，不换动作、不筛根/库/结果。独立 prepare/report/consume schema；不冒用 terminal-witness 或旧P2 authority。

prepare 硬绑定 MAIN 事前 protocol `results/main_nonlinear_suffix_midpoint_effect_20261028_v1_protocol_MAIN01.json` SHA `d421793a24ff9c38789f9baa419ff183ecbb36d2bd0082884f313ef763461a18` 及其 source/tests/doc hashes。主要 actual endpoint MSE；次要 predicted goalQ 的绝对误差；两者原 imagined 与 actual-conditioned 均保留，包括变差值。目标是同一个 actual cp10 latent 和原 goal192，不作无条件旧baseline排序修复 claim，更不作新物理控制收益 claim。

本次 intervention 是 **重编码并替换整段真实 history3**，不是 held-old2/last-only：macro5 下顺序为真实 warmup5、plan0、plan5，不是相邻 raw step3/4/5；frame path、phase、step、condition、PNG file/pixel SHA、native7 state、完整 snapshot（含原有时间字段）精确绑定。三观察身份必须不同，允许像素相等，不复制/生成帧。

prepare 每行保留 `original_root_context`（原始3 latent和2 action token）及原 final context 来源绑定。新 actual_full3 前2语义对应 original_root_context.latent_history 最后2（warmup5/plan0）；批布局重编码可发生数值漂移，本版不强加 equality threshold、不改 shape 求相等，也不把全history干预说成只修改z5。MAIN可直接逐值比较这两组数组；真正 held-old2 对照须另独立实验。本版不将 native7/jointXY 与模型Q混为同一目标。

## 真正执行的 ABI 与费用

每可用行 `model.encode({'pixels': x})` 一次：pixels `[1,3,3,224,224]` → flattened `[3,3,224,224]` → hidden `[3,257,192]`（绑定 config224/patch14，拒197）→ CLS/projector `[3,192]` → returned emb `[1,3,192]`。原 p0 transform 与独立 NumPy/PIL 预处理在 run、saved consume 两处逐值/字节核对原PNG。没有重新编码 goal。

incoming2 是 warmup6..10 和原 plan1..5；outgoing3 再附原 plan6..10。归一化使用原 float32 减均值/除std一次，无额外 tanh/clip/rescale。warmup6..10 token 须与原 root_context.action_history[1] 精确相同。predict输入只有实际latent3与动作token，不允许state/reward/outcome。模型原 action_encoder `[1,3,10]→[1,3,192]`，public predict→predictor→pred_proj(flat `[3,192]`)→native `[1,3,192]`；只取最后latent为5步后终态，三个输出不是 shifted history 或三个 child futures。

全4费用：encode_API4、encoder4/projector4（各12图）；predict_API4/action_encoder4/predictor4/pred_proj4（各12 context/flat rows）。同时分列 API/axis0 samples/temporal rows/实际attempt-return/wall；继承原allmode、suffix优化、已发生的物理、cp10编码费用，不冒称相同FLOPs或免费历史。消费自身0物理不代表原执行从未物理运行。genuine native stop缺历史保留 null 四分母；工程遗漏failclosed，不padding、不重试。

使用冻结 physical.consume→suffix full168 consume/verify/build 及 terminal.consume 的真实闭包，完整 receipt canonical重算，不接受status-only/relabel/partial。绑定所有继承输入、PNG、rawbits、task/blind/native goal/context/Q、模型tensor hash和实际installed class source（含suffix v2补齐builtin模块）；own triple、预测ABI helper与protocol闭包也绑定。模型前后 frozen/hash/source检查；CUDA only、MATH SDP、matmul/cudnn TF32 off、autocast off、inference mode，守卫before/after持久。线程profile拒AD，不替换torch globals；predict计量仅实例Hooks，退出恢复。

立即保存所有原精度pixels/preprocessed/hidden/CLS/projector/full3/normalized raw packets/incoming/outgoing/AE/predictor/pred_proj/public predict inputs和outputs；全量event顺序、native seal、完整文件census与SHA/数组shape/dtype/hash严格消费，不过滤unknown events/modules。输出64MiB（final reserve8MiB包含在内）、floor200MiB、wall600s，exclusive路径，失败保留partial。未执行或失败不作为科学负结果。

## MAIN 与 Nash 的公共 saved 接口

先 `consume(output, report_sha)` 完整核验。consume.rows 提供 `conditioned_final192`、imagined/conditioned endpoint MSE与goalQ绝对误差；**不内嵌 actual3context**。通过 `read(output/report.json)` 定位 report.rows[i].actual_full3 与 report.rows[i].feedback.inputs（context/raw_packets/action_mean/action_std/outgoing/incoming）的数组引用。用本模块 `load_saved_array(absolute_output, reference)` 读原数组：内部冻结 rgb.load_array(ref, flat-bound bytes reader) 校验dtype/shape/hash，不导入模型。先整份consume，不能只验证单数组便认整个来源。

report.prepared+prepare_sha256 绑定 prepare.rows[i].original_root_context、frames、raw10、goal192、model_task_binding和row_binding；prepare.physical_authority.producer_prepared/producer_prepare_sha256 指向原suffix full168 compiled prepare。原candidate/result应沿该闭包与原report/receipt读取，不能从终态或其它row拼凑。下游如新增recompile/winner replay须单独事前计量，不包含在本版4predict预算。本版尚未实际运行，不能给未来output SHA。

## MAIN CLI（inner根目录；全路径，无worker生产执行）

```bash
M=experiments.spectral_reachability.agent_nonlinear_suffix_midpoint_rgb_20261028_v1
MID_PREPARED="$PWD/results/agent_nonlinear_suffix_midpoint_rgb_20261028_v1_prepare_MAIN01.json"
MID_OUTPUT="$PWD/results/agent_nonlinear_suffix_midpoint_rgb_20261028_v1_cuda_MAIN01"
.venv/bin/python -B -m "$M" prepare --output "$MID_PREPARED" \
  --terminal-output "$PWD/results/agent_nonlinear_suffix_terminal_rgb_20261028_v1_cuda_MAIN01" \
  --terminal-report-sha 78bb8cfcb051623e64017da0061dea4ad00ea90daaf89bfbf774b20a50727ba5 \
  --terminal-receipt "$PWD/results/agent_nonlinear_suffix_terminal_rgb_20261028_v1_consume_MAIN01.json" \
  --terminal-receipt-sha 18d281427d34c1afce2054308208e7a9561df24a5d117cc851a5d143f17939e4
# MID_PREPARE_SHA 取上述实际stdout，不虚构未来SHA
.venv/bin/python -B -m "$M" verify --prepared "$MID_PREPARED" --prepare-sha "$MID_PREPARE_SHA"
.venv/bin/python -B -m "$M" run --prepared "$MID_PREPARED" --prepare-sha "$MID_PREPARE_SHA" \
  --output "$MID_OUTPUT" --authorize-host-cuda
# MID_REPORT_SHA 取实际report.json SHA
.venv/bin/python -B -m "$M" consume --output "$MID_OUTPUT" --report-sha "$MID_REPORT_SHA" \
  --receipt "$PWD/results/agent_nonlinear_suffix_midpoint_rgb_20261028_v1_consume_MAIN01.json"
```

## RAM回归与限制

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_agent_nonlinear_suffix_midpoint_rgb_20261028_v1.py \
  tests/test_agent_nonlinear_suffix_terminal_rgb_20261028_v1.py
```

仅RAM native-hook/数组/metadata及临时saved artifact fixtures、protocol小文件只读回归、独立冷进程禁止foreign runtime import，无真实neural/physics/production prepare/consume。首轮47passed；扩展闭包测试首轮60passed/1failed是RAM accepted fixture遗漏真实consume资源字段，已补完整字段，不改生产门。最终实际测试结果随冻结交付，不把测试当研究证据。MAIN审读/真实prepare/verify/CUDA/consume、可用磁盘/GPU仍未在worker执行；P1/freshness/scientific_admission始终false，不涉及4 reserved roots，不证明P1或机制成立。
