# P3 live-chunk 执行器调用边界接线

独占桥接 `p3_c3_live_executor_bridge_20261102_exclusive.py` 调用既有
`codex_p3_live_chunk_adapter_20261101.run_chunk`，通过仅暴露
`action_encoder` 和 `predict` 的模型包装器，将实际调用导向
`p3_c3_model_boundary_20261102_exclusive.BoundaryAdapter`。调用方显式
提供已有模型、同步回调、三轴上限和已冻结 20/30 配额；每个根的候选
由 `run_candidate` 在配额顺序下进入。live-chunk 只使用 dynamic 分支，
其中的模型调用统一记为 signal；signal/reverse 是根配额臂，不是
live-chunk 的 fixed/dynamic 开关。

encoder 和 predict 分别在模型调用前检查并记录 attempted，predict 按
实际 context 批量维度计 predictor samples；模型返回且同步成功才记
returned。超额不执行相应模型调用；异常保留失败计数并阻止重试。
桥接保留原执行器自身的 meter；该 meter 的复合计数不应与新账本相加。

CPU mock 直接运行原执行器的数值和控制流程，模型和环境均为替身，
验证成功调用、预算不足、模型异常。没有执行真实模型、真实物理或 GPU。
这不是透明的全进程拦截：调用方仍可能在桥接外持有原模型；调用前已有
计划/编码开销、predict 内部模块、环境调用和 wall 不是新三轴账本。
原执行器在信号计算后会调用传入的 `env.step`，因此真实运行须单独授权、
计量物理调用并满足源身份、时序、持久化和实验协议。未建立等总预算、
模型/物理收益或独立确认，P3/C3 gate 均不改变。
