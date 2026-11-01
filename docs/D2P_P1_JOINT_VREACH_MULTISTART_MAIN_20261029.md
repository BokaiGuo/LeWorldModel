# P1 联合 prefix / 条件 Vreach 多起点探索

针对已知数据中大量边界外向更新，新增独立的两臂回调核心：原投影上坡和 outward-free 切向更新。四槽分别使用四个固定起点、每起点五个候选，总共 160 个候选；每个新 prefix 都重新计算条件 Jacobian、bounded-ridge Vreach、suffix 与完整动作重放。切向臂在每个非零半径去除向外径向分量，并非仅在边界切换；这是预先固定的探索性规则，不表示原投影实现有错误。

候选资格同时要求模型阶段上坡、相对原 winner-prefix 的 Vreach 改善、相对原 winner 终态改善；该额外条件不替代 P1 原有物理合取。相同的回调账本为 1104 predict、128 scalar reverse、160 Jacobian、30720 数学 VJP 方向和 160 ridge solve，但内层 native 资源尚未认证，不能称等 FLOPs 或与旧实验等总预算。

主线程审阅核心并独立复跑 6 项 fixture 测试通过。当前没有实际来源 full strict、CUDA 运行、独立持久消费或物理执行；两根均为已知根，所有模型见证与 P1 门仍未开启。下一步必须先认证实际 NativeSignal 的真实前向／VJP 账本、动作 chart 和条件上下文，再运行预固定面板。

原生 callback 与 H1 signal bridge 已接入，主线程联合复跑核心及桥接 14 项 fixture 通过；测试在 callback 层核对每个条件 H1 的两次预测、192 个 VJP 方向及失败事件前缀。它们仍是非生产夹具：实际四槽八臂宿主、winner/baseline 身份、标量梯度、持久失败记录和真实 CUDA 模型闭包尚未验证，不能把声明的 native 调用预算当作实测。
# 真实来源 authority 准备

主线程独立复跑 authority 准备器 8 项 fixture 回归，并对原四槽 mixedmode 保存来源执行完整 authority 重建和独立 verify，均成功。新文件 `le-wm-main/results/agent_p1_joint_vreach_authority_prepare_20261029_v1_MAIN01.json`，SHA-256 `997289d3ca4ae173f5ece36e7c6a54fb021a306d99b8f07222b747df3fd148bd`。这是已保存输入的来源证据；`production_ready=false`、`P1_admission=false`、`freshness=false`，没有运行新的模型或物理。四槽八臂宿主、持久失败消费和独立根验证仍未完成。

后续四槽八臂候选宿主在主机 RTX 5060 Ti 完成，report SHA `25eb16c1e2515f98d4e8ace4fafbf1a8d5cbc74d79762421ba248387d1f272c5`，独立 saved consume SHA `7b3e3f9f7940a38e8ee19bbc36a337a0317960f05aac5b5fe8de27f734805179`，状态 `verified_model_only`；壁钟约 451.9 秒，标量路径实测 784 次 predictor 和 128 次 reverse，另有 160 次条件 H1 signal（含 Jacobian/VJP），不能把标量计数当全部成本。八臂均 no_witness：160 个候选位置里阶段上坡 122、相对原赢家终态改善 14，但条件 Vreach 改善 0，故合取 0。两臂各自共享根和起点，位置不是独立根。该意外的全零 Vreach 正在做条件上下文、尺度、求解与 baseline 的代码级差分审计；不能据此排除代码或判定理论不可行。没有新物理、没有独立根，P1 gate 不变。

资源补充：每臂 signal 独立记账 40 次 predictor、3840 次 VJP，共八臂 320 次 predictor、30720 次 VJP；与标量路径合计 1104 次 predictor、30848 次 VJP/反向方向，未包含 SVD/求解的等效 FLOPs。零动作处 Jacobian 是 raw 动作导数；ridge 返回的参数经过 tanh 后才成为实际后缀。因此 `own_reach_value` 是线性代理，并不自动等于实际后缀模型代价。零点 tanh 导数为 1，此观察本身不证明实现错误；需分别重算代理、执行动作和评分排序。
