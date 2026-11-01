# 两根 H15 同域 GN：HOST 交接

固定使用原 H15 两根、模型、完整 context、目标和 normalizer；四次当前点 GN，每次两方向四次球内差分探针，探针不参与选择。每根 17 次候选评分、16 次差分评分、2 次最终 whole/segmented 重放和 4 次当前点 AD；预算为 117 predictor samples、768 VJP directions 和 768 grad/raw-backward 回调，均逐实际入口计数。四次 CPU numpy 线性求解没有接入 ledger 的 solve 轴，零计数不能理解为零 CPU 求解成本。与 H15/CEM 不等总计算，已知根并非盲态或独立验证。

失败不重试、不补根，不交付部分赢家。saved consumer 检查来源/模型/精度、原生轨迹和 AD 的 basis、raw、逐行 grad、链式 Jacobian、账本、差分与优化轨迹；不重新运行模型，也不能证明不存在仪器之外调用。HOST 的 wall/disk/GPU 限额沿用已有 cooperative guard（3600 秒、1536 MiB、单文件 128 MiB、空闲地板 512 MiB、allocated 4 GiB/reserved 6 GiB），无法抢占单次卡住的 CUDA 调用；进程或存储崩溃也可能没有最终 receipt。

CPU 测试 12 项通过，含两根 fixture 的完整 slot 与 saved 审计；fixture 为替身，不是真模型。HOST 的 CUDA ABI、方向差分阈值、内存/位级重放、完整 HOST consumer 尚未真实运行。故以下是可执行候选交接，非成功结果；GPU 使用必须先与用户协调。无物理执行或训练。方向差分固定参数和两根详见 prepare，失败不得事后放宽。

有效 prepare 为 `p2_h15_gn_matched_host_20261029_v1_prepare_CODEX02.json`，SHA256 `12bf348ce9eb3c0b22d2231a0ef66339dcf230af4f606403cc66d716f76df87f`。此前 CODEX01 在源码增加绑定后失效，保留但不得使用。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_h15_gn_matched_host_20261029_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_gn_matched_host_20261029_v1_prepare_CODEX02.json --sha256 12bf348ce9eb3c0b22d2231a0ef66339dcf230af4f606403cc66d716f76df87f
# 只有在用户明确协调并批准主机 CUDA 后运行；输出目录必须不存在：
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_h15_gn_matched_host_20261029_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_gn_matched_host_20261029_v1_prepare_CODEX02.json --sha256 12bf348ce9eb3c0b22d2231a0ef66339dcf230af4f606403cc66d716f76df87f --output /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_gn_matched_host_20261029_v1_cuda_MAIN01 --authorization MAIN_APPROVED_P2_H15_MATCHED_GN_HOST_CUDA
# REPORT_SHA 必须取自上一步实际输出，只有完整完成才可消费：
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_h15_gn_matched_host_20261029_v1 consume --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_gn_matched_host_20261029_v1_prepare_CODEX02.json --sha256 12bf348ce9eb3c0b22d2231a0ef66339dcf230af4f606403cc66d716f76df87f --output /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_gn_matched_host_20261029_v1_cuda_MAIN01 --report-sha256 "$REPORT_SHA"
```
