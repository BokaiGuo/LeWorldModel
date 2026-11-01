# P3 accounting binding 独立增量审计（2026-10-27）

本版本只补充已保存事件的 protocol/source/cohort 身份绑定，以及 parent/frontier/candidate/selected-action 绑定。没有修改上一版源码、测试、报告或 seal。它不是独立神经 replay，也不是 P1/P3 放行证据。

## 权威输入

所有 results 路径相对于 `le-wm-main/`。不读取或混用其他目录中旧 corrective artifact。

| 输入 | 路径 | SHA256 |
| --- | --- | --- |
| 主线程权威修正报告 | `results/P3_ADAPTIVE_RUNTIME_ACCOUNTING_AUDIT_20261027_main_v2/report.json` | `2f82dcb6a0b9ff91f2c7ab86813b6660d37be2cd869850413a3c6e5ff088d840` |
| 原始 CUDA 报告 | `results/P3_ADAPTIVE_RUNTIME_20261027_cuda_v1/P3_ADAPTIVE_RUNTIME_report_20261027.json` | `0b581b19caa5ccbd83bef13692a06b611550b229049f769c1105eb91b4c3bd96` |
| 原始 protocol envelope | `results/P3_ADAPTIVE_RUNTIME_20261027_cuda_v1/P3_ADAPTIVE_RUNTIME_protocol_20261027.json` | `56320a62371b90b8ff6a551d25288852b39a3ab3959a32e19d541830698ad5dd` |
| source compilation bank | `results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json` | `d5f544689a9d2b5c6807b2454c1a98b9db6d88bb374badc457bdcc6defa87c8a` |

重算的 protocol_id 为 `a18656bd92f5db33bb95227d8d1e643fcd30096856151641c25648a6d355e5c0`。必须使用原 runner 的 `sha256(json.dumps(protocol, sort_keys=True).encode())`，不是 compact separators、不是 envelope 文件字节哈希、不是二次 SHA。额外禁用非有限 JSON 数值不改变这些真实输入的序列化。

## 具体新增检查

1. protocol_id 与 envelope、原报告、paired-plan source_seal 以及各 root contract 的 protocol_id 一致。sources 必须恰好非空 52 项，逐项为绝对源码路径和 SHA，运行时读取当前源码验字节；4 个唯一 roots 的顺序必须与 paired-plan、compilation protocol 及 compilation rows 一致。8 个 root/method 行必须完整且无重复，task/model/method/schedule/caps 保持冻结。
2. 读取各 root compilation 的有序 16 项 `prefix_0` 至 `prefix_15` bank。严格复刻原 CUDA runner：每项取 `program_live_commands[0]`，重复为 5 个原始二维动作；不把 compilation 的后续 suffix 错作此 bank 的 primitive。
3. 从空 parent path 开始，按冻结算法重建每层有序 frontier。所有 parent、primitive 和 selected 路径都是非 bool 整数且在 `[0,16)`，parent 必须是下一项实际 retained frontier；每 kernel 的事件必须按候选 0 至 15 顺序。检查事件成功/完成标记、数量、kernel/layer 状态、预算停止和保留宽度。
4. 按保存的 costs 与 `(cost, depth, primitive_indices)` 排序，核对 frontier 剪枝、全局 best 的路径/深度/成本；按保存的 signals 复算 scheduler 决策与 causal stop。selected actions 必须有限、位于 `[-1,1]`，且与该路径 raw bank 串接逐值完全一致。
5. 复用主线程 `corrected_resources` 与 `audit` 重建资源和 paired receipts；对重建报告的全部字段与 main_v2 比较，不另创修正口径。其 receipt root/protocol/task/model/method 与 planning/evaluation phase 绑定仍被校验。新报告记录 main_v2 文件 SHA、原始输入与 52 项源码 SHA，并校验 main_v2 seal。

失败处理是整份快照拒绝通过，不是过滤失败后缩小分母；本版本只适用于这批完整成功的固定存量记录，不把失败快照“修复”为通过。

## 可运行命令与已验证数字

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p3_accounting_binding_audit_20261027.py
.venv/bin/python -B -m experiments.spectral_reachability.p3_accounting_binding_audit_20261027
```

新增测试 65 个通过。连同 `test_p3_adaptive_runtime_accounting_audit_20261027.py`、`test_p3_adaptive_runtime_20261027.py`、`test_d2p_p3_paired_accounting.py`、`test_d2p_planning_compute_ledger.py` 合计 260 passed。负例覆盖空/51/53 sources、身份或顺序错配、parent/candidate/best 索引 999、合法但非 frontier parent、bool 索引、候选倒序、失败事件、receipt 错绑、越界/非有限/错误的盒内动作、权威报告 SHA 漂移、原输入 SHA 漂移及拒绝覆写输出。

真实保存数据复算结果：52 sources、4 roots、8 root/method 行、72 kernels、1152 planning events、19 verification predict、1171 总 predict。两处 adaptive horizon 从声明 3 更正为实际 2，其他值保持；4 对均在冻结预算内，2 对资源向量相等。没有新增 predict 或物理 step。

产物为 `results/P3_ACCOUNTING_BINDING_AUDIT_20261027/report.json` 与 `source_seal/`。目录必须不存在；脚本拒绝覆盖，所以首次产出后不要直接重跑写入命令。新 seal 存档本脚本、测试、本文及复用的直接审计依赖；report 保存 seal manifest SHA 与验证结果。它是本次内容审计的快照，不证明原 CUDA runner 执行过这些字节。

## 证据边界和剩余缺口

本审计独立增加的是保存记录的身份、索引、顺序、动作串接和 causal schedule 一致性，不是独立重算模型输出。沿用的 costs、SVD signals、best context、`full_replay_exact` 与 predictor 事件来自原始运行记录；没有重新检查神经 transition、终态编码、SVD 数值、真实 GPU kernel 时间线或物理结果。资源推导复用 corrected_resources/audit，明确标记 `original_count_reconstruction_reused=true`、`independent_neural_replay=false`。

P1 仍未通过；P3 admission、efficacy claim、compute advantage 均为 false。当前 Q 的 agent 移动而物体不动的能力假象不在此补丁中修复；不能据此宣称 task efficacy、成本优势或研究闭环。神经验证若另行进行，仍须主机 CUDA；本任务没有 CPU 神经降级，也没有抢占 GPU。
