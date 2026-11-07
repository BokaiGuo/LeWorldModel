# 绑定动作见证：GPU 接口等价结果

发现旧 `ActionWitness` 虽为冻结 dataclass，仍接受可变的嵌套列表；调用方若在创建后修改列表，动作与指纹可随之变化。其指纹也只标识动作形状和数值，不绑定 root 或时钟。这个风险不证明已有研究结果错误：已核查的 state-assisted 局部实验使用不可变元组，且原有后验审计已核对该实验结果。

新增独立版本 `pc_action_witness_v2_20261107_codex.py`：创建时复制并固定 float32 原始动作，指纹绑定动作字节、形状、root 与时钟；对混合 root 或时钟的探针，在模型查询前拒绝。时钟字符串是显式绑定，不自行认证物理帧数；该版本也不替代最终派发动作指纹检查。为避免破坏历史来源哈希，旧模块及原实验结果没有修改，新版本尚未接入所有历史规划器。

按事前协议，在已暴露 PointMaze root 1880、首相机、官方 proprio 路径的本地兼容模型上，旧／新见证各做三个五环境步动作的 GPU 预测，得到相同目标代价：

`0.3250966966 / 0.3248792589 / 0.3253208399`。

共六次预测，原零动作代价与后果差值也与历史封存结果在协议容差内一致；模型前后指纹一致。动作见证、原后果描述及输入权限测试共 **28 项通过**。

这只验证单根、单相机、state-assisted 的接口等价和身份约束，不是官方完整复现、真实物理增益、后果描述优于 lookahead 或 RGB-only 闭环证据。PC0–PC5 不改判。

- 协议：`docs/PC_BOUND_WITNESS_GPU_PARITY_PROTOCOL_20261107_CODEX.md`
- 结果：`le-wm-main/results/pc_bound_witness_gpu_parity_20261107_codex.json`
- 实现与测试：`le-wm-main/experiments/pc_action_witness_v2_20261107_codex.py`、`le-wm-main/tests/test_pc_action_witness_v2_20261107_codex.py`
