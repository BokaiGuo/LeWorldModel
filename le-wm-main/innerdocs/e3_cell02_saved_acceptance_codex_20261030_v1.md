# E3 cell02 保存证据验收交接

此消费者只读取固定 cell02 输出、固定 authority、该入口的 claim
和经外部固定 SHA 的 cell02 manifest。只有调用者确认进程已终止、
提供独立固定的 report/completion SHA、manifest SHA，以及经审阅的
本消费者、native saved 检查器、feedback join 检查器三个源码 SHA，
才会构建结果。新检查器不在旧 authority 中；这三个 SHA 需要独立
审阅和固定，不能由程序自算后视为批准。claim 绑定 attempt，但其
PID、记录和调用者确认不是进程生命周期证明。

验收调用原 chunked host inspector：重验 cell02 authority/source、
完整 publication、载入和实例身份、native 物理记录、guard、保存的
before/after census、到达的反馈节点及数值 replay；再调用原 native
saved semantics 检查身体快照、root/cp5、dispatch/return、cleanup。
对于 `completed20`，额外调用原完整 feedback saved join，验证
控制 6–20 逐步 literal action、物理新 RGB 与编码输入、编码数组和
latent、节点 body 及保存 replay 选出的下一 raw5，且要求身份绑定的
10/15 两个 encoder 边界。完成后重读 report、completion、journal、
claim、manifest 和三份检查器源码以检测审计期间变化。

合法早停或 control5 初始成功仍按原合同保留，仅检查实际到达的节点，
`complete_feedback_chain_checked=false`，不补齐后续步；partial
publication 非零拒绝，不产出验收 receipt。任何拒绝不触发重试或
删除 claim。receipt 仅是保存证据检查，不是 full saved strict、
实际物理/神经独立重放、进程内存证明、同字节 import 证明或科学准入。
源码检查是时点哈希，额外 checker 的全部传递依赖、标准库和 native
依赖没有独立封存；审阅者应检查依赖与运行环境，不得扩大证据声明。

`audit` 和 `verify` 使用相同参数：`--report-sha256`、
`--completion-sha256`、`--manifest`、`--manifest-sha256`、
`--consumer-sha256`、`--native-sha256`、`--feedback-sha256`、
`--confirmation CELL02_TERMINAL_CONFIRMED_NO_RETRY`、`--saved`。
`verify` 额外需要外部固定的 `--saved-sha256` 并重建逐字节对比。
manifest 必须是专用目录内 `e3_cell02_manifest_*.json`，receipt
必须在同一专用目录内命名 `e3_cell02_acceptance_*.json`。`audit`
使用独占创建、文件和目录 fsync；写入或 fsync 中断可能留下不完整
receipt，应保留并人工判定，不能覆盖。审计前还须人工确认进程终止、
保持记录不变以及报告 SHA 的独立保全。

目前没有 cell02 生产结果，不能报告真实验收通过。测试使用模拟环境、
CPU fixture 和临时目录；即使对被篡改事件重新封口使 transport
通过，真正的反馈 join 仍应拒绝错误的 dispatch、RGB 身份、latent
和下一节点动作。fixture 不构成物理或 CUDA 执行证据。

回归：消费者及既有 saved/identity/native 检查测试共 **37 passed**；
新增真实 feedback join 接线测试后，消费者测试单独重跑 **16 passed**。
其中接线 fixture 替换 authority/host/native 结果，仅 feedback join 使用
真实保存校验器；不能将这条测试描述为真实 authority 端到端验收。
