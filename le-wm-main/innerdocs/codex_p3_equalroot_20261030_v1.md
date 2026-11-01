# P3 每根等调用预算探索合同

固定两已知根、三策略及顺序 uniform/signal/reverse；各根每策略独立评估 10 个 H2 候选，每候选实际计量 action_encoder、predict、predictor、pred_proj 各 2 次及 metric 1 次。逐根逐策略各轴上限 20，metric 10；整批 60 候选。各轴嵌套，不相加当总计算量。每策略均实际调用，不缓存或虚耗调用；每根选择实际评估候选中 (cost,index) 最小值。

候选来源：每根 15 个 PCG64 ball 候选，种子为 SHA256 字符串 `p3-equalroot-v1-fixed-cold-bank:{root}` 的前 8 字节小端整数，不接受成本、物理结果或赢家作为输入。生成分布使用现有动作域 ball、两段各 10 维和 tanh 变换；候选流不因结果重抽。统一策略选 index 0–4、10–14；signal 对高权重根选 0–9、低权重根选 5–14；reverse 对调；权重相等三臂都用统一策略。索引窗口是人为负控干预，不是原跨根配额方法，也不证明权重能指导根内候选质量。公开旧结果、既知根与历史模型使其仅为执行前固定的探索，不是预注册或独立检验。

固定信号历史获得一次并共享；历史记录的 predict API 8、Jacobian 2、VJP 384、grad API 384、raw backward callbacks 384、model SVD 2 各自单列，存在嵌套；不假定任一臂的独立获取成本，加载、CUDA 上下文、FLOPs、每槽 wall、跨设备或全流程等成本均 unknown。不得由调用数相同声称等总算力。模型/设备、源码和原始任务身份通过现有 host 固定，源动态闭包 unknown；CUDA 初始化前获取固定 inode 合作锁，非合作进程不受约束。仅模型候选评估，禁止训练，失败留存不重试；物理须另行绑定全部六赢家与 reset/goal/warmup、动作 ACK 和 saved consumer 后才执行。科学 gate 不改变。
