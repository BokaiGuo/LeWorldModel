# P1 full CEM v2：预执行合同

本合同只约束未来采用 v2 名称、新 prepare 和新输出的运行，不修改、不
重新解释、不补造任何历史报告、成本或结果。当前准备和 CPU 测试不
授权、不执行 CUDA 或物理环境。

v2 复用 v1 的真实模型、RGB、环境、CEM 算法、根目标认证、逐步回执、
预算驱动和合作 GPU 锁。每次运行仅一个指定 root，按 0、5、10、15
提供反馈机会，原生成功可提前停止；已知根不构成独立确认。

实际 CEM：每节点四轮、每轮 16 个候选、四个精英，初始中心为零、
标准差 0.5，候选首项为中心，剪裁 raw 至 [-1,1]；用 float32 精英
均值和裁剪后的标准差更新，64 个分数中选全局最低（同分选较小索引），
再做一次 winner replay。各节点 horizon 为 3、2、1、1，node 合同中
action encode、predict 和 predict samples 上限依次为 195、130、
65、65；其他 node 账面维度为零。实际提前停止时未启动的节点不计费。

不继承旧全局 `node_caps`、多臂 `initializer_labels`、`common_suffix`
或 `extra_policy`。明确 `module_hooks_gate_each_node=false`，实际用量
和覆盖未经独立认证；controller 的 LeafAdapter 和 durable 账本不是
内部模块全覆盖 hook。节点账本复用四臂 schema 的 `native` 标签，
不表示执行 native 基线或四臂实验。shared/upstream 成本未知，不
声称等成本、公平性、完整成本、P1/E3 admission 或独立确认。

新 prepare 文件名必须以
`codex_p1_full_cem_host_v2_20261101_exclusive_root{0|1}_PREP_` 开头，
运行目录须使用同 root 的 `_RUN_` 前缀。prepare 必须在运行前独占
创建并以其 SHA 验证，源闭包包含 v2 runner、本合同和测试；运行时
校验当前来源是否与 prepare 完全一致。旧 prepare 不得供 v2 运行。
运行仍须独立授权字符串和显式 CUDA/物理标志并获得原合作锁；创建
prepare 或读取本合同本身不表示运行获授权。

本合同是可审查的事前规格而非运行记录；未进行真实运行前不可据
CPU 测试宣称 CUDA/物理执行或成本证实。实际使用前另需审查环境、
资源、模型和锁状态，以及结果的完整保存。
