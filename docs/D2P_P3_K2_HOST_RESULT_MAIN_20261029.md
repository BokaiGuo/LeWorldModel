# P3 K2 主机 CUDA 探索结果

K2 是观察 K1 队列拒绝后提出的探索合同，不是独立前瞻确认。两已知根 × 三策略固定六槽在主机 CUDA 完成，报告耗时至写报告前 553.47 秒；顶层无错误。六槽均以 `resource_cap_before_node_queue` 结束，均无完整模型计划，也无物理动作释放。固定 144 请求位置每槽中，五槽实际返回 20 个候选，另一 adaptive 槽返回 16 个；每槽另有 4 个请求被整批拒绝，属于未分配位置的一部分，不另加分母。

保存消费对完整事件账本和固定数学重建通过，receipt SHA-256 `864b260787f180953591277ec502bb4090b4c90fa6df42b726d9a9c13d16b099`；基于该 receipt 的独立汇总再次完整重建通过，SHA-256 `9cdeadcc4a6f8d3dd81b76710e5c4609b680cd8dc52723f327df07a591896106`。产物分别为 `le-wm-main/results/agent_p3_c3_k2_host_cuda_20261029_v2_consume_MAIN01.json` 和 `le-wm-main/results/agent_p3_c3_k2_saved_handoff_20261029_v1_MAIN01.json`。

三个策略的实际资源不相等；各槽预测调用分别为 68、108、110、68、108、108，cap 均为 142。对拒绝队列逐轴核查：六槽唯一超限轴均为 `program_compile_calls`，五槽已尝试 20、另一 adaptive 槽已尝试 18；整队最低追加 4，该轴验证预留为 0，上限为 20。该 4 次未实际派发。代码的 `attempted + minimum + verification_reserve > cap` 与冻结合同一致；不能从预测调用剩余推断预算检查出错。不同资源轴仍有各自预留。

完整计划缺失是原固定 no-partial-plan 合同的结果，不是物理失败、模型能力失败或 P3 调度科学负结果。未运行真实物理，新根、等实际成本优势和 C3 机制均未证明，P1/P3 门不变；不以修改旧 cap 或事后择优代替新协议。
