# SRA 在线 bank、选择和保存审计

`codex_sra_online_run_bank_20261030.py` 把当前观察的在线 context 送入既有 joint-H2 `compile_cuda`，再送入真实 `run_bank`：按原生策略评估 40 个两包候选、执行 winner 的分段与完整独立 replay；只有搜索和复核全部完成才将 winner 的**第一包**（五步 raw 动作）交给既有 online selection journal。第二包和所有候选保存在审计文件中，不当作已经执行的物理动作。后续边界必须重新观察、重新编译和选择；不接受保存的旧 winner 作为输入。

调用者先准备独占来源 manifest JSON：`schema` 为 `codex_sra_online_sources_v1`，`files` 将代码中 `REQUIRED` 各文件名映射为 `{"path": 绝对路径, "sha256": 文件摘要}`，可加入更多依赖，`action_units` 写明 raw 动作单位。manifest 自身的摘要作为 `online_select` 的 `method_source.sha256`；路径和摘要同时传给 `planner`。目标文件固定为 JSON：`goal_rgb_sha256`、`goal_vectors`、`scoring.metric.projection`；其文件摘要必须等于 journal 身份的 `goal_sha256`。入口前后核对 manifest、清单文件和目标，目标向量及投影由该目标文件读取，不使用物理评价值选择。

生产接线的调用者必须提供同一个 frozen model 和 device 对应的原生 `RootAnchoredCudaReplay`；接口本身不加载模型或创建环境。`planner` 返回在线回调，可传给 `online_select`。audit 路径必须不存在，完整 audit 独占写入并 fsync 后，其摘要嵌入候选 ID，进而绑定在线 receipt 和 journal selection。完成并封口 journal 后，`consume_sra` 只读检查 journal、receipt、audit、来源、目标、候选及资源关联。若 audit 已写入而 selection 失败，孤立 audit 不代表选择成功；若失败发生在 audit 写入之前，journal 仍保存已进入的资源操作，但不保证留有完整候选细节。

资源区分：模型 `predict` 和 `action_encoder.forward` 逐入口在执行前记 attempt，返回后记 return；正常选择要求 compilation 4 次、候选与重放 84 次，共 88 次预测和 88 次动作编码、每次 batch 1，并把各次事件存入 audit 与 journal。Jacobian 1、192 个 VJP 方向、2 SVD、1 solve 是编译阶段的静态预留，不是逐算子实测；其阶段 return 不是硬件证明。`run_bank` 本身捕获普通候选异常并继续其他槽位，因此失败时不能认为已立即中断所有后续候选，但不会产生有效 winner，也不退款。编码之外的上游工作、wall、I/O、内存、GPU 内核及 FLOPs 未完整计量，未知不等于零。

证据边界：固定清单只覆盖列出的项目文件和调用者额外列出的文件，**不证明完整传递依赖闭包或内存模型与 checkpoint 字节相同**；目标文件中的 RGB 摘要和 latent 目标之间的真实性也需独立验证。源哈希、调用计数和 receipt 不是远程证明。fixture 模式明确写入 audit，不能作为真实神经执行或物理结果。当前测试用替身编译与替身模型/replay，但实际执行 `run_bank` 的候选、选择与双重 replay 逻辑；没有运行 GPU 或物理。此处不声明完整 D2P、独立确认或等预算公平。
