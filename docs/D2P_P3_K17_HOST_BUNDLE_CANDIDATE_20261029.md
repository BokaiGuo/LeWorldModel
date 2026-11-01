# P3 K17 主机 runner 与 completion/manifest 候选

## 当前结果

新增 `le-wm-main/experiments/spectral_reachability/p3_k17_host_bundle_runner_20261029.py`，接入既有 `SourceBoundStore`、真实 K17 `outer.execute_material` 和独立 saved consumer。公开 `run_host` 和内部执行核心都调用硬编码拒绝门禁；任何传入的 authority 都不能解除它，拒绝发生在创建目录、导入 gym 或构造环境之前。没有命令行派发入口。本次未启动 GPU、物理或训练；没有真实原生日志或真实物理结果。

内部候选代码准备了固定原生 factory：`gym.make('swm/PushT-v1', render_mode='rgb_array', resolution=224, with_target=False, render_action=False)`；在构造后用现有 `validate_env`、渲染条件、env spec、unwrapped 类名和类源码 SHA 与外部 authority 对比，不符时关闭并拒绝。该证据不是不可伪造的实际环境身份证明，也不覆盖依赖闭包。执行核心校验外部 authority 摘要、固定六槽、既有 meter 各轴 cap 和 wall、已保存数据来源摘要、既有源码清单与 runner/consumer 本身的源码摘要；不接受预算扩大或按结果重排。authority 是待独立批准的合同输入，当前没有验证签名或批准过程，不能作为执行许可。

候选执行结束后，写有界 native report 和保存 consumer 输出，核对保存/实时操作计数与槽状态，再为 native 文件生成 `manifest.json`，最后独占、fsync 写 `completion.json`。外部持有 completion SHA 与 authority SHA 后可用 `consume_bundle` 重核文件清单、哈希、authority、固定来源/预算、环境证据和独立 consumer。状态区分 `failed`、`saved_incomplete` 和 `saved_engineering_only`；最后一种仅是事件覆盖的工程标识，仍不表示 native 真实性、生产或科学通过。当前 saved consumer 的 `complete_receipt_verified` 仍为 false，不被 runner 改写。

bundle 元数据每文件最多 2 MiB，写入时检查整个 bundle 总量不超过既有 128 MiB cap 和 200 MiB 空闲 floor，不改原有物理预算。底层 native store 的资源、wall 与失败保留机制仍适用；bundle 清单和 completion 写入不是原子事务，写入失败或断电可能留下不完整目录，不能宣称完成或保证失败记录成功。实际墙钟持续时间、文件系统行为及 native 所有资源并未独立认证。

## 检验

在 `le-wm-main`：`.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p3_k17_host_bundle_runner_20261029.py tests/test_p3_k17_saved_source_budget_consumer_20261029.py tests/test_p3_k17_live_source_budget_seam_20261029.py tests/test_agent_p3_c3_k17_native_meter_20261029_v1.py tests/test_agent_p3_c3_k17_candidate_join_20261029_v1.py`：40 passed。包括公开及内部入口的默认拒绝、假环境不匹配与关闭、fake 失败状态保存、saved consumer 失败发布、外部锚及文件篡改、预算拒绝，以及六槽假环境与假帧的完整接口链。测试中通过 monkeypatch 专门替换硬门禁；并非当前主机执行授权。假帧不是真实 PNG，six-slot 事件覆盖不是物理证明。

## 之后实际可执行的前提

1. 独立闭合并审核 native 安装/环境及完整执行源码和依赖绑定，包括真实环境版本、文件、加载代码与不可绕过的 meter/store 边界；补充实际环境证明，不能只相信自报类名和哈希。
2. 扩展并审核 saved consumer：核对 33 warm、plan0 与后续观测的 phase/step/overlay、PNG 解码与像素、原始 guard 完整重算、目标/状态及 replay 闭包；验证缺失、伪造、提前停止、不可恢复的 I/O/崩溃、空间与 wall 限额。现有计数与事件数量检查不充分。
3. 独立批准六槽、固定来源、物理预算、终止及存储合同；由可信渠道预先持有 authority SHA、源码 SHA 与环境 pin，并在完成后独立保管 completion SHA。当前 authority 数据结构没有可信签名/批准校验。
4. 审查完整的 failure/cleanup 和 manifest 发布协议，包括底层写故障可能无法落 failure、bundle 元数据空间不能保证足够、文件/进程并发篡改及 source-to-runtime TOCTOU。审查后才可实现真实 admission，进行有明确授权的主机执行。

当前任一前提未闭合，硬门禁继续拒绝。已知两根不是新鲜根，所有工程封存状态均不改变 P3/C3 的科学准入。
