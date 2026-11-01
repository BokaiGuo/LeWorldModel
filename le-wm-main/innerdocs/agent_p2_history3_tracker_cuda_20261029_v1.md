# known2 history3 tracker HOST CUDA smoke

独占 runner：`experiments/spectral_reachability/agent_p2_history3_tracker_cuda_20261029_v1.py`；RAM 回归：`tests/test_agent_p2_history3_tracker_cuda_20261029_v1.py`。遵守 MAIN 事前 slice protocol，不是三 proposal 实验、不产生 raw action、不证明 scaler、P1/P2、freshness 或性能。

固定 known root0/off、root1/off。真实 parity prepare/report/完整 strict receipt 三 SHA 固定；prepare 重新调用原 parity consumer 并逐字段比较整个 receipt，不接受 status-only。读取原 observed0/5/10 的 Hi/P2 逐位相等 encoder/projector 输出，拼接 `[1,3,192]`，再与原 route/history 双核；原 normalized action history `[1,2,10]`、原 goal 编码 `[1,192]`。逐个原 RGB SHA、task/source/causal context 和全部原数组均封存。没有新 encoder、目标重建或 witness warmstart。

固定 H3/N8/I2/topk2/var1/seed0/segments=[1,2]。全时域零初始化、unclipped normalized Gaussian、identity192 terminal float64 meanMSE。复用封存 core 的安装 CEM 方程和 callbacks，不 patch globals。每根搜索 6 API/48 samples，actual returned mean 的 whole/segmented replay 各3 API/3 samples，总每根12/54、两根24/108；encoder/high/AD/physics 均0，FLOPs unknown。保存全部 candidates、实际 mean/std、history、模块输入输出、whole/segmented checkpoints、paid scores 及 local_check，无数组去重。

真实 CUDA 由 MAIN 授权。原可信 NPY loader 425 项完整 shape/dtype/value SHA、eval/冻结属性、construction 全字典与原 parity construction 精确相等；每根前后及全作业前后均核模型。新 core 内部显式 facade，未改原 loader。无 pickle/foreign 动态模型导入、CPU 神经 fallback、训练/refit/raw inverse。

资源：root 文件64MiB（常规数组预留8MiB footer）、全作业256MiB（预留32MiB footer）、floor200MiB；core 数组32MiB/root，root integration300s；全作业900s包含 verify/原 strict、加载、哈希、两根 dispatch/保存/local_check。TorchDispatchMode 在实际 tensor op 前后检查 CUDA allocated peak3GiB/reserved peak4GiB，模式自身源码纳入闭包、不改算子。SIGALRM/合作式检查不能抢占正在执行的 CUDA kernel。`whole_wall_seconds` 在最终 report 写入前采样，最终排他 fsync 收尾时间不在该数值内；收尾仍受容量/floor约束，不声称严格内核抢占或完整端到端硬实时保证。

每个文件排他创建、fsync(file+parent)、SHA/bytes/census/写尝试账完整消费。两根输入在模型加载前落盘。普通 core 异常返回的 partial 全保留、实际 debit 不退款；core 在返回 trace 之前抛出则 debit 标 unknown 而非0；未启动根明确0/not_started。core 本身在内存返回 trace 后才保存，进程崩溃/强杀之前未返回的 trace 不能承诺逐事件持久化；不伪造完整性。任何 partial 不可获成功 receipt。运行文件不覆盖，失败目录保留。

消费重新 verify 整个 prepare 和外部 strict、全部 NPY/header/EOF/finite/SHA、原输入三元组、root归属、完整 core local_check、模型/源码前后/计账/census。core 的 CUDA reduction/RNG 检查是运行时 witness，不冒称独立 GPU 数值重算或独立梯度。RAM backend 永不升级 HOST marker。

## MAIN 精确命令

下列生产 prepare/verify/consume 会执行原上游 full strict，仅 MAIN 执行。worker 仅运行 RAM 回归和窄 saved ABI 检查，不运行这些命令。占用的新文件名若存在须使用新 MAIN 后缀，不能覆盖失败。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q tests/test_agent_p2_history3_tracker_cuda_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_history3_tracker_cuda_20261029_v1 prepare \
  --parity-manifest results/agent_p2_hilewm_computation_parity_20261029_v1_prepare_MAIN01.json \
  --parity-manifest-sha256 796c80ca14574b29b9920b0ef53b02cc6c3b3b47842708951e5bc0b17aadf4e1 \
  --parity-report results/agent_p2_hilewm_computation_parity_20261029_v1_cuda_MAIN01/report.json \
  --parity-report-sha256 12fe43cc4a796fffc18b98fcc867283f8534c22110d794729652ac112aedead8 \
  --parity-receipt results/agent_p2_hilewm_computation_parity_20261029_v1_consume_MAIN01.json \
  --parity-receipt-sha256 3555696c02329062084d791c0c40958084d7352e6f7680938bade0068afe9480 \
  --output results/agent_p2_history3_tracker_cuda_20261029_v1_prepare_MAIN01.json
# TRACKER_MH 填上一步实际返回 SHA；不预填未来 hash。
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_history3_tracker_cuda_20261029_v1 verify \
  --manifest results/agent_p2_history3_tracker_cuda_20261029_v1_prepare_MAIN01.json --manifest-sha256 "$TRACKER_MH"
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_history3_tracker_cuda_20261029_v1 run \
  --manifest results/agent_p2_history3_tracker_cuda_20261029_v1_prepare_MAIN01.json --manifest-sha256 "$TRACKER_MH" \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_history3_tracker_cuda_20261029_v1_cuda_MAIN01 --main-authorized
# TRACKER_RH 填实际 run 返回 report SHA。
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_history3_tracker_cuda_20261029_v1 consume \
  --report results/agent_p2_history3_tracker_cuda_20261029_v1_cuda_MAIN01/report.json --report-sha256 "$TRACKER_RH" \
  --output results/agent_p2_history3_tracker_cuda_20261029_v1_consume_MAIN01.json
```

剩余前置不是新资产：MAIN 审阅后实际 prepare、HOST CUDA 控制路径及完整 consume 尚待运行；RAM 成功不证明 GPU memory-mode 或真实安装 CEM 已接通。不扩三 proposal，不迁 raw，原 scaler 仍未知。
