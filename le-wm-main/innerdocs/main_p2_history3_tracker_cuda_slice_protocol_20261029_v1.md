# MAIN：history3 tracker 最小 HOST CUDA 接通协议

在任何本 tracker 生产输出产生前固定。本 slice 验证真实安装 CEM、共同 normalized history3 入口及 actual returned mean 的两次付费 replay，不是强基线实验，不签 P1/P2 或三 proposal 比较。未知原 raw scaler 保持拒绝，不训练、不拟合、不打开 outcome。

固定复用已完成 computation parity 的原两个 known root、原 observed0/5/10 编码 history3、原 between action history、原 goal 编码 target；两根均运行，不按效果选 root/目标，不用编译动作 warmstart。外层必须由真实 parity prepare/report/consume 与原 checkpoint/source 绑定证明输入来源，不能只接受调用者自称的 state 字典。若字段或语义无法核对则 prepare 明确失败，不猜值。

显式 spec：horizon=3，samples=8，iterations=2，topk=2，var_scale=1.0，seed=0，segments=[1,2]，initialization=full_horizon_zero_no_witness，action_domain=unclipped_normalized_gaussian，objective=identity192_terminal_meanMSE_float64，predict_api_cap=12，predict_sample_cap=54，array_bytes_cap=33554432，wall_seconds=300。样本/迭代是有界整合 smoke 的事前工作量，不是性能调参或论文配置，不在看到结果后增加预算。

每根计划12 predictor API、54 predictor samples，两个 root 分开计账；encoder/high/new AD/physics 均0，模型加载不计为 encoder forward。完整保存实际 CEM callbacks、返回 mean/std、whole/segmented mean replay 的全部数组、模型/source/输入 authority、partial 首因与完整分母。已有 core 的 RAM 测试不是 HOST 证据。只允许主机 CUDA，禁止 CPU fallback；唯一 GPU 由 MAIN 串行派发。

外层独占新 producer schema；输出每根最多64MiB、整作业256MiB（含32MiB失败收尾 reserve），磁盘 floor200MiB；整作业 wall900s，包含输入验证/加载/哈希/两根实际运行，CUDA allocated3GiB/reserved4GiB。必须在派发前封存源/输入/spec；文件排他写、hash/census、全量消费，不能以 RAM marker 改标 HOST。不能把本 normalized smoke 的数值放进 raw physical baseline 主结果表。
