# P3 quota 保存接线进度

本轮独占新文件，旧实验与 gate 不变。新增独占文件、逐行 fsync 的 hash-chain journal；非终态不能消费预留空间；短写/fsync 错误使 writer 失效。失败记录仍依赖文件系统可写，不能保证 I/O 故障时一定落盘。reader 使用外部 tail 检验顺序、截断、重复 JSON key、篡改。

独立消费者不调用神经模型；针对完整事件逐项重建顺序、两根候选身份、quota、逐候选四轴 attempt/return、轨迹合法性、model-Q 及逐根赢家。只检查整体语义，不重建模块数组与神经输入输出链，不认证设备、模型、来源、物理或科学通过；不能把此消费者作为生产 strict consumer。

来源工具对显式文件及已导入模块逐字节 SHA、大小、文件替换与读期间变化检查；动态导入与模型实际加载模块尚未封存，不是完整来源闭包。尚未生成新 prepare、没接设备 UUID 或实际模型权重前后 hash、没完成 producer/consumer 端到端接线，也没获取 GPU 合作锁；不得启动。曾经的 GPU 查询仅表示当时空闲，不是锁或启动许可。

已测试：15 项离线测试通过，覆盖事件截断/篡改、quota/成本/调用记录篡改、来源漂移、失败不重试、两根分别选赢家和 CPU device 拒绝。测试用 RAM 模拟 callback，不是 CUDA 或物理结果。

下一步应以独占文件固定 prepare 和输出路径，绑定全部 checkpoint/依赖与真实加载模块、设备 UUID 和模型 tensor hash，使用外部 pinned report 和 journal tail；增加模块输入/输出与真实调用一一对应的 saved 校验、失败终态，以及锁内加载/进程退出清理测试。完成后重新检查 GPU 与同一合作锁，才考虑无覆盖的 host CUDA。已知根、付费 shadow 候选、全加载与历史信号成本仍不代表等 FLOPs 或等 wall，也不意味着独立验证或 P3 通过。
