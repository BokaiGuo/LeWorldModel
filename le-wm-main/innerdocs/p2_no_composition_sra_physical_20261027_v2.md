# Joint-H2 no-composition SRA v2 physical consumer

独立 consumer；仅消费主线程已经完成的四个 calibration root：
`results/p2_no_composition_sra_20261027_v2_prepared01/report.json`。
不修改规划器、旧 consumer、白名单、门槛或任何旧 sealed 字节。
实现与测试遵循 experiment-code 的日志/验证要求；本任务不训练、不绘制效果图，
不读取未来 heldout，不启动 GPU/physics。

## Evidence contract

prepare 首先要求真实 report 已存在；调用规划器原有只读 selected_actions，
验证 planning sources/input seals、40 候选谱 bank、动作、原 goal0/Q 分数、
稳定 first-tie winner 和单 edge 独立 H2 replay。这里的 min 是检查已冻结选择，
不是重新规划/重选；无模型 forward。谱 bank 的数组复核需要每次 inputs 调用额外
8 SVD/4 bounded solves，单列 verification_work，不计作零成本，
不混入已完成的 planning ledger。跨 NumPy/LAPACK 的退化基不一致将明确拒绝；
不改变 bank、不松动复现比较。

consumer 再按 root_sha256 建唯一映射、拒绝重复/缺 root，不按数组顺序配对。
已核对原执行前 source_seal_1/sources/004.source 第386行含 selected_actions；
归档/manifest/当前源码 SHA256 全为
`6e1402ac8abe3aabfffefd24aa016c341e6eb4f40d3f60e233152460ca70e45a`。
consumer 固定该源码 SHA 及此次 report SHA
`522afa5a68b23a6ff829c5c6d820d858953dccc0f778548e747ba594ff73937a`，
禁止事后补 exporter/重封存追认旧报告；若原归档不存在 exporter 则明确拒绝。
检查完整 3-state/2-action predictor context、RGB root/goal 身份、目标向量/Q/source
哈希、精确 arm、实际 dispatch/encoder ledger、budget binding 和 compilation operation
receipt。prepared 绑定报告、四份 bank、全部上游输入和 installed evaluator/环境源；
源封存清单及 args 绑定 physical protocol。输出独占且不可覆盖。

## Accounting and claim boundary

| 每 root | 数量 |
| --- | ---: |
| 完整 H2 candidates | 40 |
| candidate predict | 80 |
| replay predict | 4 |
| nominal + Jacobian forward predict | 2 + 2 |
| 总 predict / action encoder | 88 / 88 |
| Jacobian / SVD | 1 / 2 |
| scalar-output VJP directions | 192 |
| predict processed-context-token samples | 264 |

四 root 共 352 predict、4 Jacobian、8 SVD、768 数学 VJP directions。
VJP directions 不是实测 backward API/kernel 次数。相对 CEM/random/MPPI 的 84
predict/root，SRA 多 4 compile predict/root，还存在 Jacobian/SVD overhead。
仅 candidate 40H2/80predict 加 replay4 一致，不能称等总预算；
FLOPs、上游建 bank/RGB 编码、共同 wall-time cap、backward 内部开销仍未测量。
P1/P2_admission、fair_total_budget_comparison、prediction_budget_matched 均保持 false。

## Main-thread workflow

先 review 新 module/tests/本文件，再由授权者执行 model-free prepare：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.p2_no_composition_sra_physical_20261027_v2 prepare --planning results/p2_no_composition_sra_20261027_v2_prepared01 --output results/p2_no_composition_sra_physical_20261027_v2_prepared01
```

prepare 不构造 env、不 reset/render/step、不分配 CUDA、不加载模型。
测试使用真实已保存报告进行只读验证，以及明确 fake Env/metadata fixtures；
fixtures 不是物理效果证据。若封存目录已存在，必须选择全新独占输出，不能覆盖。

只有 main 显式运行下述命令（本实现线程不运行）：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.p2_no_composition_sra_physical_20261027_v2 run --prepared results/p2_no_composition_sra_physical_20261027_v2_prepared01 --output results/p2_no_composition_sra_physical_20261027_v2_cuda_main_v1 --encode-terminals
```

复用 full20 physical 的 action binding/durable guard/journal：
重建原 root_seed 的4根，RGB history/goal reset 校验，原 official evaluator/pose；
每次 env.step 前 fsync attempt，返回后记账，异常保留分支及失败，缺分支显式补齐，
无 retry/replacement。先落 physical_report 再允许最多4个完整10-command终点 CUDA
编码，冻结 eval 模型并禁止 predict；guard/termination 非固定H2终点不得编码。
40 selected-path commands 为计划上限，root construction/reset/warmup 不在该计数内，
不声称 simulator 总步数只有40。完整四根 physical 与4编码尚需 main 实际执行，
即使全门通过也只是 calibration integrity，不打开 P2。
