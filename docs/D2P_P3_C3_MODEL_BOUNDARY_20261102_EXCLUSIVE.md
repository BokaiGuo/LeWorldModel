# P3/C3 独占模型调用接线合同

`p3_c3_model_boundary_20261102_exclusive.py` 是可接线接口，不启动模型、
不导入 CUDA、不执行环境，也不改现有执行器或 gate。调用方提供已有模型、
预注册配额点和臂、完整三轴预算、同步回调。每个候选通过
`adapter.candidate(root, index, evaluate)` 顺序进入冻结配额；
`evaluate(adapter)` 只能将需要计数的模型调用接到
`adapter.predict(context, action_context, phase="signal"|"search")`。

该方法在 JEPA 的 `action_encoder(action_context)` 和
`predict(context, encoded)` 两个实际 API 调用前分别计费；
predict 同时以 `context.shape[0]` 计入样本数。所有相关轴先验检查，
超额不派发相应调用。完成后调用同步回调，只有同步成功才记 returned；
异常保留 attempted、不退款、关闭后续候选。信号与搜索分别分账。
候选超出预先冻结的根配额或未按顺序请求则拒绝。

该接口**不是透明代理或完整运行时监控**：调用方若绕过适配器使用原模型，
此账本无法发现；原模型内部 predictor、输入编码、计划前处理、物理动作、
wall 与其他成本也不在三个轴内。`candidate_panel_complete` 只表示调用方
完成了规定数量的回调，不证明每个回调执行了真实模型调用，也不证明
合法候选、真实信号来源、durable journal 或实际端到端等预算。
正式执行前需接入并独立核验上述来源、完整计量与固定输入合同。

CPU mock 测试仅验证边界计数和拒绝逻辑，不是神经或物理实验；
`P3_admission=false`、`C3_admission=false`。
