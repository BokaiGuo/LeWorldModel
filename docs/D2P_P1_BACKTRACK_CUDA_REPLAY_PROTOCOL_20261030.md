# P1 回溯动作的固定 CUDA 重放

对已导出的 8 臂 × 两角色全部 16 条 raw15，先完整复验上游保存结果和清单，再固定新 runner 与所用辅助源码的运行前哈希，继承原任务 prepare 的模型类和投影器绑定。每条均从对应原 root context，使用原 normalizer 分两条独立控制流重放三个五步 packet：一次连续路径，一次每段构造新上下文的路径；全部 96 个 predictor、96 个 action encoder 调用都执行并计数。保存各 packet 后继、最终完整历史和 action tokens，比较两路径逐位一致，第一 packet 与原 prefix 的模型中间 context 相同，第三 packet 与回溯保存 endpoint 相同。失败保留清楚状态，不选择有利槽或放宽指标。

只运行冻结模型 HOST CUDA，核参数与缓冲区前后哈希和固定源码闭包前后哈希；不训练、不做物理执行。该重放是对动作与当前固定实现的新确认，不追认旧回溯运行缺失的辅助源码 before/after，也不证明物理可达、独立根或 P1 准入。实际物理 adapter 的 reset/目标/完整历史和控制尺度仍须单独认证。
