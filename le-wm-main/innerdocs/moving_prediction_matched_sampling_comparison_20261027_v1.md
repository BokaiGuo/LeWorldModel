# Moving prediction-matched sampling：saved 物理比较 v1

新三文件独占实现。固定两根 × `CEM` / `kernel_faithful_mppi_nominal` 共四组，读取 Jason 的 NEW `moving_prediction_matched_sampling_cuda_20261027_v1` 和 Pauli 的 NEW `moving_prediction_matched_sampling_physical_20261027_v1`。不重做当前 overlay comparison；不产生新的动作/候选、模型或物理调用。不改任何 sealed 文件或共享总账。

experiment-code 技能用于先固定合同、真实输入和失败留存；其通用训练、hill-climbing、绘图和“不得零准确率”等规则不用于本任务。零/负结果必须保留。两个 required references 已从本机 `/home/gbk/.codex/skills/experiment-code/references/` 完整读取（技能文本中的 `.claude` 路径不存在）。本会话没有跨 agent 消息工具；以下接口依据两方现有源码，供 MAIN 转交，不声称已经直接发送。

## 接口与实际预算

planner schema/status 固定为 `.report` / `matched_sampling_plans_sealed_not_admitted`，own consume 返回 `.consume` 的四条 `rows`，每条带 source_slot/arm/task/blind 身份、commands、selection_sha256、whole_run_report_sha256、selected_index、model_terminal_cost。比较器只允许该固定 producer 的 saved-only `consume`；在 import 前验证 MAIN 指定的 source/tests/doc SHA 与白名单绝对路径。绝不调用 optimize/run 或任何神经/环境执行入口。

重要命名边界：保留 arm 的既有字符串用于精确 join，但 **本 NEW 变体是 nominal-guided sampling，派发72个已评分 sample 的稳定 argmin，不是最后 updated nominal**。不混入旧72predict final-nominal MPPI，也不称官方/importance-corrected MPPI。

| 资源轴 | 每组 | 四组总计 | 口径 |
|---|---:|---:|---|
| fullH2 candidates | 72 | 288 | 3×24；非另外增加 nominal/replay |
| predict API dispatch | 144 | 576 | 每候选两次H1 |
| predictor forward samples | 144 | 576 | 真实 batch1，不是3个 context rows |
| action_encoder dispatch | 144 | 576 | 与预测重叠，不加成“总WM调用” |
| pred_proj dispatch | 144 | 576 | 与 API 重叠 |
| AE / pred_proj context rows | 各432 | 各1728 | 与 batch samples 不同 |
| AD / J / math VJP / SVD / solve | 0 | 0 | 新采样算法无这些额外轴 |
| 新 encoder / image projector | 0 | 0 | 原8+8历史编码单列且全次只继承一次 |
| 额外 nominal / replay / padding / retry | 0 | 0 | 不空跑补齐 |

逐组拒绝143/145互抵、把432 rows当144 samples、或仅用whole-run aggregate掩盖某组超支。检查 actual planner report、prepare contract 与 own-consume receipt一致；producer consume还按其封存合同复算所有候选/RNG/分布更新和实际 dispatch ledger，确保72候选以外没有未记账额外评分。此处不复制一个新的 candidate auditor；RAM tests 注入 saved consumer 回执，只验证本接线，不证明真实CUDA计数。

两种采样方法只说 **prediction API calls与forward samples匹配**。各资源轴分别输出，不相加成 FLOPs；不要求它们与GN/D2P或完整pipeline总计算相等。保留原normalizer、native192 identity Q、packet球、MATH/TF32off、冻结模型；无训练/adapter/Qfit/新阈值。

## 物理合同与错误分母

新 physical report schema为该stem加 `.report`；状态仅接受 `matched_sampling_physical_recorded_not_admitted` / `partial_or_failed`。plain prepare内四plans，report四rows以 `(root_slot,arm)` join，不能依赖数组顺序。验证 prephysical selection manifest、源/模型任务身份、compiler own plan SHA、physical selection_digest、prepare bindings、execution_binding、journal文件SHA与完整hashchain。

目标从 **原 moving sourcepack + construction journal** 取得，不从实际终点或latent重新拟合：最后 construction native_return.state 为goal7，replay0 step10为initial7，按原float64拼接重算root SHA。固定sourcepack authority：`c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6`。只读 evaluator metadata，不提取 witness actions。

物理动作只能来自这四个已冻结 plan；核对 warmup前五 `[.1,0]` 后五 `[0,.1]` 的float32 bytes、selected full10 dispatch raw、returned command、returned original goal。physical kernel自己的 `latest_action == command` 检查没有另存每个plan实际action数组，因此本比较器不冒称独立重建该数组：独立可核范围是saved dispatch/returned command及封存producer的原检查。

每张 fresh RGB224 PNG核文件SHA、decoded pixel SHA、native7 state/snapshot/observation_state；0/5/10 checkpoint必须与journal完整receipt一致且有accepted/render-preserved guard。返回状态必须与checkpoint一致。没有额外render/encoder。所有四分母保留，失败不替补，合法早停不补10步；缺step10时terminal10=null，不用早停末态填充。原errors/首错/secondary、实际attempt/return、last returned state与checkpoint可用性分列。

资源上限是 **40 selected +40 warmup=80 public steps**、8 reset、16 set_state、8 reset内goal_render、4离线goal_bind、816内部physics、88显式render、constructor/close各4。轴重叠，不求和。constructor中途失败内部成本unknown不能填0。

新 Ops 在journal intent写入成功后才进入callback并记attempt；report `actual_operation_receipts` 是实际entry/return账。比较器用此账重算actual，与journal正常返回的下界分别保存，**不把intent当成已执行**。失败后日志可能少于已返回callback：partial时保留actual与缺口，不缩小分母、不伪称完整日志；缺失transition label保持缺失。完整状态要求所有return与实际账对齐并核原census恒等式。

## 输出指标与条件比较

每根每臂分列 initial0 / prefix5 / terminal10：native7 distance及reward、原native success、wrapped angle/independent wrapped success、object XY、agent XY、velocity误差和wrapped joint距离。success条件没有速度约束；速度属于7维distance/reward。原native angle `min(abs(delta),2*pi-abs(delta))` 多圈可为负，原样报告并与wrapped结果分开，不能静默修env或将2π表示差算几何错位。

两采样方法按固定每根 MPPI−CEM 描述差，不从结果筛根。prefix distance5−distance0只是前缀变化，不足以证明相对raw-zero的严格非贪心。无新RGB编码输入时，`actual_latent_Q=null` / `missing_not_encoded`；已保存的模型terminal cost明确为预测，不冒称实际收益，不从PNG hash推断Q。

可选GN/D2P参考仅支持 original moving `p1_constructive_moving_history_physical_20261027_v1`，不借用overlay或不同horizon终点。必须同时提供其外部physical/prepare/binding/journal与原planner/prepare/code、handoff。saved own-consume验证原GN/D2P，随后完整plan→去prediction的handoff→physical raw commands绑定。`moving_history_planner_handoff.v1` 的SHA **不是**原CUDA report SHA：分别核handoff与 `projection.source_report_sha256`。

每个root × sampling arm × GN/D2P，只有sourcepack/root/goal/initial/blind、off条件、H2/full10、真实physical metric，以及task文件SHA/task_id/task_contract完全相等才给描述性数差。任一不一致，不计算差，列mismatched_fields；结构/链/文件损坏直接拒绝。GN每根78predict、D2P66predict及各自AD/VJP/SVD等预算从原report保留；不将GN两根156predict写成单根。GN/D2P predictor samples没有同等receipt时不凭API推断新样本账；参考比较一律compute unmatched，无等FLOPs结论。

## MAIN binding 与 CLI

本实现不实际build/prepare/assemble，不创建fake actual result。MAIN实际两producer完成并审读后填写小JSON：

```text
schema: moving_prediction_matched_sampling_comparison_20261027_v1.binding
files: 下列13键，每项恰好 {file: canonical absolute actual path, sha256: actual lowercase hex64}
reference: null，或下述完整GN/D2P reference文件映射
```

`files`精确键：

- `physical_report`, `physical_prepare`, `physical_binding`（output/execution_binding.json）, `physical_journal`, `physical_manifest`, `physical_source`
- `planner_report`, `planner_prepare`, `planner_source`, `planner_tests`, `planner_doc`
- `sourcepack`, `source_journal`（原moving源，不是旧zero-history源）

source/test/doc路径必须是各producer在本项目内的实际同stem路径，未冻结或未出现的SHA不得编造。physical prepare必须真实绑定上述源码/planner/source依赖，manifest的authority必须是NEW planner及其实际report SHA。输入fixture中注入consumer只用于RAM测试，不提供任意可执行plugin/模型回调CLI。

`reference`非null时精确十键：`physical_report`, `physical_prepare`, `physical_binding`, `physical_journal`, `planner_handoff`, `planner_report`, `planner_prepare`, `planner_source`, `planner_tests`, `planner_doc`；每项仍为 `{file,sha256}`。不提供时不阻塞本次四项sampling比较，GN/D2P pair为missing而非0。

在 `/home/gbk/code/LeWorldModel/le-wm-main`，实际参数仅MAIN填入：

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python -B -m experiments.spectral_reachability.moving_prediction_matched_sampling_comparison_20261027_v1 \
  --binding /absolute/NEW_sampling_comparison_binding.json \
  --binding-sha ACTUAL_BINDING_SHA \
  --output /absolute/NEW_sampling_comparison.json
```

只在验证与算术完成后独占写单个新JSON，cap2MiB、floor200MiB+2MiB reserve、`xb`/fsync；无覆盖/删除/下载。源码仅复用冻结overlay comparison的纯数学/parse/png/hashchain/source_targets/write helper（SHA `f4ecb7b318771becb0d411ef594b9303c0981b653ea3a9af6fc11e3be6fb1ac5`），不修改它、不运行它的material，不复做MAIN当前overlay分析。

测试命令（仅RAM PNG/metadata/callback，禁神经与模拟，大fixture不落盘）：

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_moving_prediction_matched_sampling_comparison_20261027_v1.py
```

mock覆盖：完整新sampling binding→四物理分支、GN/D2P旧handoff接线、非数组序join、逐组API/samples计数、禁止nominal/replay额外评分、原goal/raw history/标签/guard、损坏SHA、callback正常返回但日志失败的留存、missing10、native/wrapped/velocity语义、CLI。通过仅表示接口工程验证，不是actual计算/控制效果证据。

交付时33项RAM/mock测试通过。partial的数值消费仍要求外部文件SHA及可验证的完整终止hashchain；若最终journal本身未能写成，拒绝生成比较报告，不把工程损坏当负结果。原始失败产物不删除，MAIN可从producer原始证据另行恢复。

所有输出始终P1=false、P3=false、independent_confirmation=false、equal_total_FLOPs=false。两个known根复用不是独立确认；0/4、零改善或负改善也不能推出理论否定。
