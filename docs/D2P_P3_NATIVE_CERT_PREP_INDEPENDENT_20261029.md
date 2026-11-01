# P3 原生认证准备：只读来源与 fixture 边界

本次新增独立文件 `le-wm-main/experiments/spectral_reachability/p3_native_cert_prep_20261029.py` 及其测试，不改其他子智能体文件、现有阈值或现有结果。没有 GPU 运行、训练或物理执行。

## 已核查的真实来源

CPU 只读调用现有 `agent_p3_c3_k17_native_outer_20261029_v1.material()` 成功，返回 K17 固定顺序 `(3,0,fixedH1)`、`(4,0,staticH3_progress)`、`(5,0,adaptive)`、`(9,1,fixedH1)`、`(10,1,staticH3_progress)`、`(11,1,adaptive)`。该调用使用既有 pinned handoff、receipt、保存的 replay 和来源材料；其上游认证边界由现有实现继承，不是重新运行模型、物理或全量 strict。两根是已知根，不是新鲜根。本次未获得外部锚定的真实 candidate join 完成目录，故没有声称对真实 join 完整消费或原生采集认证。

现有 candidate join 的 `consume` 需要外部 manifest SHA，可核对保存文件、journal、六槽、PNG hash 和事后 lifecycle 投影，可选重建来源；其 `run_fixture` 明确只能用于 fixture。生命周期并非原生实时调度记录，不能证明环境身份、真实动作、实时 guard、资源和物理结果。现有测试的 source rebuild 使用 monkeypatch fixture，与上述真实来源读取分开。

## 新增的准备能力

纯内存 `audit_fixture(events, caps, expected_actions)` 检查六槽顺序、明确逐轴限额、guard 在动作之前、每次 intent 的五动作摘要和对应 return、预算超限、失败停止与未完成记录。费用按调用者给出的 `charge` 汇总；调用者提供的摘要、guard、事件和限额均未认证。它不是原生采集器、独立 saved consumer、实际计量器或预算授权器；无论 fixture 是否完整，`native_provenance_verified`、`budget_authorized`、`dispatch_allowed`、`scientific_pass` 均为 false。提前 native termination 按保守 fail-stop 处理，不宣称完整六槽；不会改变既有 scientific gate。

新合同不验证固定 budget 的外部批准、实际 attempted/completed 资源配对、墙钟/存储、原生动作字节来源、原生 guard 与事件来源真实性、原始重放和目标/环境绑定、保存日志的哈希锚及失败持久化。调用者提供五动作摘要不等于从付费 replay 提取并验证了真实动作；fixture 本身也没有防止删改事件或伪造计量。它不能取代已有 child budget authorizer，也不引入或修改其节点、配额和拒绝规则。

## 剩余原生采集门槛

需要事先批准的六槽完整物理预算、停止合同和资源轴；不可替换的原生 reset/placement、warm10、33 帧、goal 和计划帧采集与原来源逐项绑定；对原付费 replay 的动作字节认证；原生 guard、intent、实际 return 和失败先后持久化；可信实际资源、墙钟和有界存储计量；独立锚定的完整产物与严格消费者；异常、存储故障和伪造来源的端到端对抗测试。缺失任何一项不应派发，也不能以 fixture 满足。

## 检验

在 `le-wm-main` 执行 `.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p3_native_cert_prep_20261029.py tests/test_agent_p3_c3_k17_candidate_join_20261029_v1.py`：12 passed。测试覆盖完整 fixture、守卫拒绝、动作不匹配、额度超限、顺序错误、return 不匹配、布尔伪额度、待返回、失败和提前终止，以及既有 join fixture 的篡改及来源重建。通过仅说明所测代码的工程行为，不是原生证据或科学通过。
