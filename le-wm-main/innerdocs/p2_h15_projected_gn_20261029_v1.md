# H15 同域投影 GN：数值核心与待接线边界

固定对象：既有 H15 的两条已观察 off 根、各自原 root context、identity192 terminal MSE 目标、15×2 动作和半径 `0.75*sqrt(15)` 的 30 维 logit 球。raw 动作使用 numpy float64 tanh 后转 float32。根与结果已知，本协议不是盲态或独立验证。现有 H15 使用单次零点 Jacobian 和 40 候选谱库；旧 gradient MPC 是 H2，并非本 H15 同域方案。CEM 两根物理结果已存在，不得按其结果更改设置或筛根。

数值核心固定零点初始化、4 次阻尼 GN（damping 0.01）、每次固定步长序列 `1, 0.5, 0.25, 0.125` 并逐个投影球内，所有候选计分，严格改善才替换 incumbent、相等保留先前者。每根恰好 17 次终点评分、4 次当前 incumbent 的完整 192×30 logit Jacobian；失败不替补。只计抽象调用，并非 predictor samples、VJP 或 FLOPs 对齐 CEM/H15。4 次 Jacobian 至少对应 768 个标量方向，具体 predictor 和 AD 实际轴必须由原生 ledger 计数；不得用抽象次数冒充实际资源。此处固定的是候选方案，不是已封存、可执行的生产预注册。

已实现和测试的只是 `p2_h15_projected_gn_20261029_v1.py` 的 CPU 纯数值回调接口；测试为合成 fixture，不是模型结果。现有原生 signal 在零 raw 动作处取 Jacobian，不能用于任意 incumbent。要完成：新建文件实现任意 logit 的 native AD 和 tanh 链式规则、严格账本和异常关闭；绑定现有 HOST 的模型/任务/normalizer/精度和哈希，持久记录、独立 saved consumer、资源上限及完整物理接线；用 fixture 检查整个生命周期及动作字节。必须把预算与源码封存后才可单独协调 CUDA；物理执行也须单独授权。当前不提供冒充可运行的主机或物理命令，不训练、不启动 CUDA、不执行物理。
