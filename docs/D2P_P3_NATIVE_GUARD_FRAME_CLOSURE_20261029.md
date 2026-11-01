# P3 K17 已保存原生 guard 与帧消费接入

## 本次推进

在既有 bundle 独占代码中新增 `p3_k17_saved_native_guard_frames_20261029.py`。它读取外部 SHA 锚定 journal，重新读取现有 `outer.material()` 的 pinned replay/task/reference，并以外部来源摘要比较整个材料快照；对 guard 及物理审计直接依赖源码清单做前后摘要检查。该来源闭包继承原 upstream 的 pinned receipt 和 saved replay 读取，不是对所有训练或模型执行历史的重认证。外部来源摘要必须由独立可信渠道提供，进程自报或事后同目录取 hash 不构成可信身份。

实际复用 K17 `physical_gate.capture_check` 和 `replay_guard_bridge.check_live_only`，从 journal 的 reset 配方、10 个 warm 返回、33 帧、goal attach 和 plan0 帧重建 live guard 输入，验证 pinned 原始 task、已保存 whole/segmented paid replay token、动作归一化、warm10、root/goal 和原始 warm reference；重算 proof 与保存的 proof 精确匹配。帧按 warm0..10 与 plan0..5 的 off/on/off 顺序检查，并与对应 transition 的 state/snapshot、前后非变异不变量及 goal 绑定。对每帧校验文件 SHA、实际 PNG 解码、RGB224、原有 `moving.image_sha` 与 `moving.guard`、off/on/off 的 off 像素一致。已登记保存文件由 bundle manifest 外部 completion SHA 锚定。

新 consumer 接入 bundle 的写入和独立 `consume_bundle` 重算；只有基础六槽覆盖且六个 guard、306 帧被严格复核，才可标记 `saved_engineering_only`。该状态仍不是物理证明、生产授权或科学通过。原来 fake 六槽 bundle 测试将严格 consumer 替换为 stub，只验证 bundle 接线；严格 consumer 的独立测试使用真实 PNG 编码和真实 history/action guard，但来源、事件是 fixture，且针对合成图像将 `moving.guard` 替换为 stub。

## 实际接口缺陷及修复

核查发现原 K17 outer 给 `meter.where` 放入整数 `slot`，而沿用的原生 `place()` 调用 `store.emit('placement_recipe', **meter.where, slot=slot)`，其中后者 slot 是 reset 配方字典：这会在 Python 合并关键字时直接抛 `TypeError`，无法到达 journal。未修改原 outer 或其他子智能体文件；新增 `p3_k17_native_outer_scoped_20261029.py`，复用原 episode/meter 和原 outer 控制流，在 meter 上移除冲突的 slot 字段，在独占 store 代理中分别注入整数调度 `slot` 和字典 `reset_identity`。真实 `place()` 函数配合 fake reset 的接口测试证明事件保留两种身份；它不是物理运行。bundle 当前改用该适配 outer，且新接口与 strict consumer 均纳入外部源码摘要清单。

## 真实来源与测试边界

仅 CPU 只读调用原 `outer.material()`，返回固定槽 `(3,0,fixedH1)`、`(4,0,staticH3_progress)`、`(5,0,adaptive)`、`(9,1,fixedH1)`、`(10,1,staticH3_progress)`、`(11,1,adaptive)`，来源快照 SHA256 `ee75bb9cd863e700bf0e3d08587e353ea7c21600a0dbd9f11d38c0c4fe86f03a`；当次检查的 11 个 guard/原生源码文件重建前后未变。之后将 candidate-join 源码也纳入未来严格源码清单。此检查不是新的原生采集、不是完整依赖闭包，也未验证可信运行时环境。

执行 `.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p3_k17_saved_native_guard_frames_20261029.py tests/test_p3_k17_host_bundle_runner_20261029.py tests/test_p3_k17_saved_source_budget_consumer_20261029.py tests/test_p3_k17_live_source_budget_seam_20261029.py tests/test_agent_p3_c3_k17_live_guard_20261029_v1.py tests/test_agent_p3_c3_k17_physical_gate_20261029_v1.py tests/test_agent_p3_c3_k17_native_meter_20261029_v1.py tests/test_agent_p3_c3_k17_candidate_join_20261029_v1.py`：71 passed。对抗测试覆盖重链后的帧顺序、guard proof、reset 和 goal 篡改，以及文件和像素摘要分别伪造；这是 fixture 验证，不是实际物理结果。

## 仍缺项

尚无真实 K17 bundle 和外部实际 completion 锚。实际环境身份、已加载代码与完整依赖闭包、硬件/运行时和不可绕过的事件采集并未由当前源码摘要证明；后置验证不能证明回调来自真环境。真实原生指标/回报、所有操作嵌套栈、部分失败后捕获、时间与资源完整性、并发篡改和崩溃持久化仍需独立严审；当前 guard 的 8 个历史中间 on 帧没有原始配对来源。bundle 自报环境证据和用户提供 authority 尚无可信签名与先验批准校验。主机硬门禁没有解除：需要独立批准六槽来源、预算和实际物理范围，完成这些证据与故障协议审核后才能实现有条件的 admission；不会把测试通过或已知两根当作科学通过。
