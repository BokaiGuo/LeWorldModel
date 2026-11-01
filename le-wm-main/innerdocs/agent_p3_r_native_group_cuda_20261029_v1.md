# 原生 R：known2 × H1/H3 四组诊断

独占三件套，遵守 MAIN 事前 `main_p3_r_native_group_diagnostic_protocol_20261029_v1.md`（SHA `b8c738587bc51cd0db1ce7b5883924a82ea1167d572e20389ca6929522faf505`）。复用已冻结 tree/components/bank 与真实 NativeTree，不改旧库、不伪装 schema、不猴补 globals。不是旧 equal-cap tree：旧 replay cap2 仍不能派发 R18；本 runner 使用 MAIN 新固定的独立完整 integration cap，不倒灌旧实验。

## 固定输入与执行 ABI

`build()`完整复用冻结 tree.build 的原 task/root/model/Q/normalizer/source 权威，绑定其冻结三 SHA、自身三件套和 MAIN 协议。四槽顺序 root0H1/root0H3/root1H1/root1H3。root 是原 warmup 结束 control0 的已保存 history，不是 actual cp5；`root_context_role=original_saved_warmup_control0_NOT_actual_cp5`、`actual_cp5_start=false`。不得从 outcome 改根、换 bank、改参数或重试。将来 actual-cp5 resolver 属于另 own-schema，不能事后 relabel 本版。

每槽独立 shared Ledger。先仅调用该 H 的原 NativeSignal，保存 root/local parent 双身份 request/response/mapping、完整 nominal/AD/J/192VJP/native IO；不另付不需要的另一 H。已付 signal 全 transcript 排他 fsync 保存成 `signal_s{slot}_0000.json`，实际文件 SHA 与原数组指针、外层付费 entry 一起进入 spectral bank authority。J 已含原 normalizer，不能二次缩放。

再调用冻结 `tree.evaluate_R`：一次 float64 SVD，完整原 spectral8（H1原 last8、H3明确30维推广）加独立 zero。每项原 raw edge5H 真 replay、postcontext H1 nominal/J、原 bounded_ridge、float64 tanh→一次 f32 raw suffix5、真实 suffix replay。保存全部 full3/between2 路径和全部 eight+zero 数组，不用 root J 替 child、不复用控制预测。资格严格 immediate>zero 且 reach<zero；全部 eligible indices 保留，reach winner 只是诊断，不是 tree beam/计划发布。无 eligible 记录 `no_witness`，仍完整九项；suffix 从不拼成可执行 edge。

公开 `group(task,h,factory,sink,...,capsule=...)`、`execute` 与 `validate_group/validate_rows`。HOST factory 在 run 内唯一绑定实际 `old.NativeTree`；RAM 只有显式 permit，用于非神经 fixtures，生产 consume 不提供 RAM 开关。report.rows[i].result 包含原 parent_signal、完整 signal_mappings、R_groups[0].control/rows/eligible_indices、native_calls 全 transcript 与 resource_receipt。所有 model_plan_available/plan_release/physical_release/fullpolicy/scientific flags 保持 false。

## 静态真实费用与失败

每槽 cap 精确由 `signal_cost(H) + bank_cost + kernel.minimum(H)` 的27轴投影编译。无未知轴补零；27轴中结构未调用操作为确定零，额外未知 scalar_backward/FLOPs 保持 null。

| 每完整槽 | H1 | H3 |
| --- | ---: | ---: |
| inclusive predict API / samples | 38 | 60 |
| context tokens / pred_proj rows | 114 | 180 |
| J / signal_compile | 10 | 10 |
| VJP / grad / raw callback（各） | 1920 | 1920 |
| array SVD / solve | 10 / 9 | 10 / 9 |
| program_compile / metric / replay | 10 / 18 / 18 | 10 / 18 / 18 |
| AD primal predict（独立子轴） | 10 | 12 |
| J 元素（独立子轴） | 19200 | 23040 |

完整四槽196 inclusive predict、40J、72 replay、112 native invocations。AE/predictor/pred_proj 等对应轴独立列出，不将嵌套成本重复相加或宣称 FLOPs 公平。完整 group 前原子预检其 bank+九项，零 final verification reserve；本诊断 final-plan replay0，没有计划发布或物理执行。

原生异常停止后续组，仍保留固定四槽和所有已尝试费用/数组。R 进行中失败保留 control+8 rows，已完成/失败/未请求分别记录；未完成不补数据、不退款、不改成 no_witness。report partial 原样持久化，run 非零，不自动重试。完整 consume 要求四组完成；内部 RAM prefix verifier 可以验证已知失败前缀，不将其签完整结果。

MAIN 协议工程限额：每组512MiB（实际 journal全字节+source capsule+final report row），总2GiB含256MiB收尾reserve，floor200MiB，wall1800s含 verify/加载/hash，allocated3GiB/reserved4GiB。排他输出，持续检查与 fsync。MATH-only、matmul/cudnn TF32=false、CUDA/CPU autocast=false，模型权重/buffer/installed classes/source 首尾核验，forward实例 hook 完整恢复；无 CPU 神经 fallback、encoder、训练或物理。

## Saved strict

own consume 重建完整 prepare 与原 source 权威，认证实际 report/prepare SHA、全量 artifact SHA/census、source capsules、binding、完整 hashchain、precision/model/class/source beforeafter、工程限额。逐 native调用核原 full IO/AD/raw/normalize/history/path，逐事件核全量资源。再以保存的 native 返回值重放 inert numerical ledger/bank/SVD/ridge/资格，对完整 group 精确比较；用原实测 clock 还原 time-bearing authority hash，不使用 neural parity 容差或重跑模型。未知 event/孤立 call/group、遗漏费用/候选、错 root/horizon、改 normalizer/J/raw/资格均拒绝。

## MAIN 精确 CLI

从 `/home/gbk/code/LeWorldModel/le-wm-main` 执行，以下 SHA 只允许填实际上一条命令返回：

```bash
.venv/bin/python -m pytest -q tests/test_agent_p3_r_native_group_cuda_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p3_r_native_group_cuda_20261029_v1 --help
.venv/bin/python -m experiments.spectral_reachability.agent_p3_r_native_group_cuda_20261029_v1 prepare --output results/agent_p3_r_native_group_cuda_20261029_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p3_r_native_group_cuda_20261029_v1 verify --prepared results/agent_p3_r_native_group_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p3_r_native_group_cuda_20261029_v1 run --prepared results/agent_p3_r_native_group_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_p3_r_native_group_cuda_20261029_v1_cuda_MAIN01 --authorization MAIN_APPROVED_HOST_CUDA
.venv/bin/python -m experiments.spectral_reachability.agent_p3_r_native_group_cuda_20261029_v1 consume --prepared results/agent_p3_r_native_group_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_p3_r_native_group_cuda_20261029_v1_cuda_MAIN01 --report-sha ACTUAL_REPORT_SHA --receipt results/agent_p3_r_native_group_cuda_20261029_v1_consume_MAIN01.json
```

worker 仅执行 RAM/NumPy buffers、保存文件 ABI、冷导入与 CLI tests；未 production prepare/run/consume，未神经/物理/fullstrict。实际输出与模型数值未知，待 MAIN 审读冻结件后执行。本版本不签原 R 等价、P1/P3/C3、equal-cap、full tree 或控制收益。后续 actual-history 物理 slice 必须另绑定真实 history 起点并报告全部 no_witness/refusal，不从本诊断直接放行。
