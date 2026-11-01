# E3 cell11 探索性反馈运行交接

固定格 `r1.g0.h3.dDx` 原有主机 CUDA control5 receipt 仅是
五步初始控制，不是闭环完成。新 authority 通过固定 resolver、
checkpoint、control5 report/completion/receipt、H3、Dx 和 cp5
setup 重建；来源 SHA-256
`6525f072cbea5042e845f973fa8e43e19affa08f0e012887f777124d73b2190a`。
启动模块闭包 SHA-256
`ce019ebaa4056f6560961e9d6a90030dfc287dbdf1fef8da524806916759fd0b`，
独立启动 precheck 通过。该检查仅加载 checkpoint 检查依赖，
不执行神经前向或物理环境。

已完成固定单次主机 CUDA 执行；2026-10-31 00:46（上海时间）
实际子进程 PID 605322 已在 NVIDIA GeForce RTX 5060 Ti 上分配
约 330 MiB。作业句柄 22926，输出目录
`le-wm-main/results/e3_cell11_cooperative_native_20261031_main_RUN01`，
claim 和父日志位于同名前缀。相关启动和既有准备测试 16 项通过。
这里是已知根探索，不是独立任务或 P1 准入。

父进程持有项目的合作式 GPU 锁并创建一次性 claim；该锁不阻止
绕过锁的旧入口、其他主机或任意路径运行，不构成全局排他。
已有 claim、日志或输出不得覆盖、删除或自动重试。若子进程失败，
先检查父日志、原生 partial 和实际模块 census，保留失败证据。

完成后必须对 report 和 completion 计算实际 SHA，调用
`agent_e3_chunked_identity_host_20261029_v1.inspect` 及既有 continuous /
integrated 保存消费者，核对实际状态、边界、物理动作、帧和来源。
退出码零不等于闭环完成；原生提前停止不得补足。计量应分别报告
本次与既有 control5，缺失的 FLOPs/GPU 时间不能编造。完整 saved
strict、跨任务公平对照及独立泛化仍需另证。

## 保存态验收结果

已实际调用既有 chunked identity、continuous 和 integrated 消费链，
独立消费 report、completion 和 authority 的固定哈希。结果为
`completed20`：10 个 warmup 步、20 个控制步、32 帧、未原生早停；
第 10 和 15 步各完成一次真实观测的 encoder 与 projector，
帧匹配无遗漏、终止行核验通过、分块绑定完整。

精简验收结果保存在
`le-wm-main/results/e3_cell11_saved_receipt_20261031_main_RUN01.json`，
SHA-256 `81e7ab7df4e6cbd786ad314162fd48842743093604ba2cfb891910394aba2fd1`。
验收入口及拒绝不完整、早停和帧错配的 5 项测试通过。验收只读
既有产物，不进行神经前向或物理执行。该记录不是完整身份封存、
物理重放或科学通过；本轮也没有包含原 control5 的全部成本。
