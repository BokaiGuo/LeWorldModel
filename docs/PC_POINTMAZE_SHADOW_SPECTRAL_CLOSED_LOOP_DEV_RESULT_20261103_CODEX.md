# Shadow RGB 谱初始化闭环：开发实现与非物理测试

## 输入核查

用户新建的 `le-wm-main/experiments/pc_spectral_shadow_context_diagnostic_20261103_codex.py` 为两个已暴露 shadow 决策上下文的离线固定输入诊断；本次仅读取，没有运行或修改。已有结果的基线方案在各自保存输入上复现，两个 Jacobian 的有限差分相对误差约为 0.000793、0.000534。那是离线模型诊断，既非闭环结果，也非独立确认；每个上下文的 20 次有限差分前向不属于新执行器。

核查 SHA-256：诊断脚本 `061bc435dc2be6f66ea4d600b26f59a8bed127f78f7c603a0c7b3d4e57e20562`；诊断结果 `ff410434453bf538ede4192e0bcfc69144ba013bc3e853df38fe73f8225ce543`；谱模块 `54be2507a20d699805d6b4f9816298424864cdea6e8c08ebb6cdb1836142d26a`；旧持久执行器 `99c94f7e4c118fd51fa0ff6f8a543fe2f36b68f8eed30ea70dcdd5b0e9324489`。

## 新实现

预先合同为 `docs/PC_POINTMAZE_SHADOW_SPECTRAL_CLOSED_LOOP_DEV_PROTOCOL_20261103_CODEX.md`。新文件 `le-wm-main/experiments/pc_shadow_spectral_closed_loop_dev_20261103_codex.py` 支持 baseline、spectral 和 paired；每臂每根独立 reset，前缀每帧检查已有 v4 shadow 哈希，后续仅将本臂连续 RGB 送入模型反馈，独立执行各自选出的动作。环境初始状态只传 bridge，不传规划回调。复用持久快照、事件和重置模块；新单桥派发前核对动作快照，attempt 和 returned 各自 fsync，返回 RGB 独占保存并 fsync；失败不重试。

每个成功决策两臂各 50 CEM 样本、4 批；spectral 额外 1 次可微响应前向、2 次反向向量、1 次 SVD、0 次有限差分前向，计数和结构墙钟单独记录；额外编码、持久写入和其他运行成本也不应被忽略。完整 paired 合同是 20 次决策、1000 CEM 样本、80 批，额外 10/20/10 次结构前向/反向向量/SVD，不是等总计算预算。模型调用 attempt/returned 事件、完整 CEM trace、结构数组、RGB、动作历史和随机噪声均有对应持久记录；异常在完成事件之前停止，不能把缺失返回视作未执行。

## 测试与限制

项目虚拟环境中执行新测试、现有谱模块测试及现有持久派发测试，共 **16 passed**。新测试使用 CPU 可微假响应和假 RGB bridge，检查谱候选进入被评分首代、结构计数、退化失败、双臂独立反馈、快照动作绑定、故障不重试和未显式授权时 CLI 拒绝。测试不会启动真实 bridge；CPU 假体不能证明真实 CUDA、渲染、驱动、模型梯度或未来物理闭环可以运行。开发合同与测试驱动方法分别固定比较边界并验证失败路径。

**本轮未运行真实 bridge、CUDA 模型或任何物理实验**，因此没有新的真实 RGB 反馈、物理结果、有效性或性能结论，也未做独立确认。未来实际运行需要另行授权、符合主机 GPU 协作锁合同，并检查输出和失败事件；不得覆盖旧产物。执行器为实验开发代码，真实端到端路径尚未实际验证。
