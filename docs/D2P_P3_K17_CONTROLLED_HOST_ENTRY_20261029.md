# P3 K17 受控 HOST 入口

## 范围与状态

`p3_k17_host_bundle_runner_20261029.py` 现在提供 `prepare`、`run`、`consume`：不再使用永久拒绝壳。研究范围固定为两个已知根、六槽 K17、每槽原始 warm10 加 paid replay 的 first5；现有资源轴上限、900 秒墙钟、有界存储和失败停止规则不变。需外部持有的 prepare 文件 SHA 和显式确认字符串；缺失、内容变化、来源/源码/预算不符时拒绝。prepare 和 consume 不构造环境；run 才会创建 `swm/PushT-v1` 并执行物理。没有模型训练或 GPU 调用入口。

prepare 只读取已保存的 pinned replay 和 task 并重建来源，验证既有固定 PushT 源码路径及 SHA，固定 registration/class、render 参数、六槽与资源限额，写独占 canonical JSON。run 重核源码与预算、重新读取并对照来源摘要；现有 `validate_env` 在环境创建后验证源码绝对路径与 SHA、relative、action_scale、dt、control_hz 和 with_target，runner 另检查 class/module、env spec 和 renderer。来源/源码清单不是完整加载依赖闭包，也不具备加密签名或防范本机恶意篡改的硬件身份；使用者须在受控宿主机独立保管 prepare SHA 和运行结果 SHA。

环境校验挪到计量过的 constructor 返回之后、任何 reset 之前；若不符，原 outer 的 finally 使用计量过的 close，保存 failure。此前直接关闭未计量的问题已修复。构造过程中抛异常若无法返回环境对象，内部资源无法保证关闭；记录为失败，不能称实际资源完整。若 native store 初始化失败，尽力独占保存外层 failure；若目录创建、磁盘、fsync 或后续 manifest 写入失败，可能没有有效 completion，需要保留残留目录人工检查，不能将其当成功。

成功返回只意味着产生一个外部 SHA 锚定的 bundle。`saved_engineering_only` 必须经过 source budget consumer 和 guard/PNG consumer 且覆盖六槽；物理来源真实性、完整计算/依赖、科学准入与生产授权仍不成立。失败完成包仅校验其清单，不能将缺失过程视作已验证；部分运行不能补零。

## 主线程命令（本轮未执行 run）

从 `/home/gbk/code/LeWorldModel/le-wm-main`，分别选用不存在的独占路径；首条为 CPU 只读来源预备：

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python -B -m experiments.spectral_reachability.p3_k17_host_bundle_runner_20261029 prepare --output /absolute/path/p3_prepare.json
```

独立保管输出的 `prepare_sha256`，核对来源、预算和研究范围后，主线程明确选择执行：

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python -B -m experiments.spectral_reachability.p3_k17_host_bundle_runner_20261029 run --prepared /absolute/path/p3_prepare.json --prepare-sha256 PREPARE_SHA256 --bundle /absolute/path/p3_bundle --confirm P3_K17_KNOWN2_SIX_SLOT_FIRST5_PHYSICAL_ONLY
```

以上 `run` **会实际运行物理环境**，本对话未执行。独立保管输出的 `completion_sha256`，再做 saved consumer：

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python -B -m experiments.spectral_reachability.p3_k17_host_bundle_runner_20261029 consume --prepared /absolute/path/p3_prepare.json --prepare-sha256 PREPARE_SHA256 --bundle /absolute/path/p3_bundle --completion-sha256 COMPLETION_SHA256 --receipt /absolute/path/p3_consume.json
```

在 prepare 后如源码变更，必须重新 prepare 到新文件；不要改旧摘要或用同目录事后摘要替代外部锚。运行所在工作目录须为 `le-wm-main`，各输出父目录须存在且有足够空间。缺文件、初始剩余空间不足、source pin 或 native 源码不一致均阻止执行。

## 本轮验证与剩余风险

测试命令：`.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p3_k17_saved_native_guard_frames_20261029.py tests/test_p3_k17_host_bundle_runner_20261029.py tests/test_p3_k17_saved_source_budget_consumer_20261029.py tests/test_p3_k17_live_source_budget_seam_20261029.py tests/test_agent_p3_c3_k17_live_guard_20261029_v1.py tests/test_agent_p3_c3_k17_physical_gate_20261029_v1.py tests/test_agent_p3_c3_k17_native_meter_20261029_v1.py tests/test_agent_p3_c3_k17_candidate_join_20261029_v1.py`：72 passed。环境、实际调用、持久化集成仍用 fake fixture；部分 bundle 测试用 stub guard audit。测试不证明物理执行闭环。

仍需主线程核对真实依赖、外部身份锚和存储/墙钟承载，实际运行后审查 completion、独立 consumer 和失败材料。源码清单非全依赖闭包；计数仅覆盖已挂接操作，不证明全部 CPU/GPU FLOPs 或未记录回调；两根不是 fresh roots；8 个历史中间 on 帧未有原生配对。不可把 guarded/engineering 状态视作 P3/C3 科学通过或生产就绪。

## 主线程实际运行与保存消费

固定 prepare SHA `5ad994ebad867096e7de6cc3aada7b59bbe12881e1aab1eca6a1e22a0abc641b` 后，主线程在宿主机执行了两已知根六槽 first5 物理仿真；本流程不调用神经模型。completion SHA `41ed9feb5fdfaaa1edbb9edf6b2dc1f28df130826b5129dba82c3173bceb2818`，独立 saved consumer 的 receipt SHA `20caa3ba066ca86299c59192505aba5a13dce7a6037af207df87c8281e01a155`。记录覆盖六槽：306 次显式渲染、90 次 public step、924 次物理步，均 attempted/completed 相等且无记录失败；consumer 重算 6 个 guard 和 306 帧。状态仅 `saved_engineering_only`，`native_provenance_verified=false`、`complete_receipt_verified=false`、`scientific_pass=false`；不把保存事件一致性当作完整原生来源、等计算预算、独立新根或 C3 科学优势证明。
