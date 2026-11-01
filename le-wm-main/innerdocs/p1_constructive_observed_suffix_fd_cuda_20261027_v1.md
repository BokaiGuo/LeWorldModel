# Observed-prefix H1 direct FD（MAIN-only CUDA）

仅检查已付费反馈编译的原导数路径，不优化动作、不拟合新 Q、不看 terminal outcomes 选方向或点位。外部固定来源：`results/p1_constructive_observed_suffix_cuda_20261027_v1_cuda_main01/report.json`，SHA `ac8a65900b94639f94b3797e90f80e73fa0500e13a055629750dc4dc418aaf79`；parent prepare SHA `5b9f970a72e3cc3ef45b31d5d35416af6cdc779189f55fd3fd8ad5670a9a3e88`。

## 接线与数学

固定四臂 census 中 D2P branch1/3（source_slot0/1），不是挑表现最差分支。逐项核对原 `suffix_problem_frozen`、`zero_nominal_saved`、`fresh_J_saved`、result 重复记录、实际 prefix NPY 与原批准 task。只读取 prefix5 NPY，不读取 terminal10 NPY、物理 endpoint、certificate/sourcepack，不调用旧 producer material/verify/consume 或任何优化器。报告/结果 JSON 的封存字节包括其他候选信息，但本诊断不使用其 cost 或 winner 来生成输入。

原 `TorchBackend.terminal` 是 `transition(observed_emb3, observed_actionhistory2, tanh(u).reshape(5,2))`，`jacobian` 在 zero10 向量用 functional Jacobian，故存储矩阵是192×10 **logit J**，不是还需除 std 的矩阵；零点 tanh 链等于1，不能再乘一次 normalizer/chain。

固定3方向：packet time-major flat0 basis、flat1 basis；固定 sign vector `[1,-1,1,1,-1,-1,1,-1,-1,1]/sqrt(10)`。epsilon `[.001,.003,.01]`，minus/plus。每次 float64 的 `±eps*v` 只 cast 一次 float32 logits5×2，然后 NumPy float32 tanh 一次生成 raw5×2；不投影、不 clip、不修改上下文。

NativeH1 只继承 NativeCuda 的冻结模型/module guard、float32 normalization、单 packet `_step`/serialization，不调用 fullH2 rollout。每次从同一个完整 observed3-state/2-action context 重建 CUDA tensor，接一个 token，保留全部新的3/2内容。原 parent taskID 保留，observed_problem_id 另列；不把修改 context 的 H1 问题伪称原 H2 任务。

先每根各新算一次 zero_nominal，必须 terminal 与完整下一 context SHA 精确匹配原 zero candidate。即便不匹配，新返回值也先 durable 保存，然后停止、保留 partial、不重试。每个新扰动的全部 context、请求、attempt/completion、终端192值均写 fsync+Z_SYNC_FLUSH gzip journal。比较全192中心 FD 与 savedJ@v，另报 float32 effective direction 对应 Jv、abs L2/Linf/RMSE、relative/symmetric relative、cos、midpoint drift、相邻 eps FD 差；零分母为 null，无科学 passing 阈值、无自动科学 verdict。

## 独立账本与存储

每点 `1 anchor + 3 directions × 3 eps × 2 signs = 19 H1`；两点38 predict +38 actionencoder，18中心差分。0新AD/J/VJP、encoder/projector、SVD/solve/selection、physics/training。原反馈已付20predict/actionencoder+4J/768方向，AD forward含于20；更早编译/编码/物理成本仍独立继承，不称总 pipeline 零成本、等预算或 H2 诊断。

每次模块 pre-entry 计 nonrefundable attempt，38硬 cap；真实同步后记 completion，必须 actionencoder→predict→单packet。禁止 encoder/projector 和 AD 入口；完整 frozen model 前后 SHA、原配置/权重/源闭包/实际 prefix 与 task 导出 SHA 双向复验。首错不被 cleanup/模型复验/写盘错误覆盖。模型未加载时不声称模型已复核。

输出上限2MiB（含 partial、journal、analysis、report），预留128KiB封尾和最终报告，执行前要求200MiB+2MiB空闲；prepare<256KiB。mock 的完整 syncflush journal+分析在内存压缩/读回验证；这不是实际 CUDA 容量或导数正确性证据。真实超限保留失败记录，不能通过删 J 或失败数据降低账本。

## 命令（仅 MAIN prepare / launch）

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_observed_suffix_fd_cuda_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_observed_suffix_fd_cuda_20261027_v1 prepare --output results/p1_constructive_observed_suffix_fd_cuda_20261027_v1_prepare01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_observed_suffix_fd_cuda_20261027_v1 verify --prepared results/p1_constructive_observed_suffix_fd_cuda_20261027_v1_prepare01.json --prepare-sha <MAIN_SHA256>
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_observed_suffix_fd_cuda_20261027_v1 run --prepared results/p1_constructive_observed_suffix_fd_cuda_20261027_v1_prepare01.json --prepare-sha <MAIN_SHA256> --output results/p1_constructive_observed_suffix_fd_cuda_20261027_v1_cuda_main01
```

实现方仅 source/mock/saved-array RAM build，不产生 prepare、actual CUDA 或物理报告。实际执行必须 MAIN 审核后独占；结果即使数值一致也仅支持这两个已知 observed H1 零点、三个固定方向的局部导数诊断，不补独立根/P1准入，也不解释唯一物理失败原因。
