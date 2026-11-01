# 在线模型动作接口：范围与使用约束

`codex_online_model_action_20261030.py` 是不加载模型、不分发物理动作的回调接口，接入既有独立 paired durable journal。它不是 Full D2P 或 SRA 的真实运行实现。

对每个臂、每个观察边界，调用者提供该臂三帧字节、身份字段对应的本地来源文件及摘要、方法文件及摘要、在线 encoder 和 planner 回调以及独占 receipt 路径。帧摘要须匹配该臂 journal 内历史；encoder 收到帧与该臂两包动作，planner 收到观察请求、三状态 latent / 两动作历史上下文与计量器，不通过此接口接收评价器物理状态或奖励。planner 必须返回在线模式、请求及上下文摘要、完整候选清单及稳定最小 cost 对应的五步二维 raw 动作；不能直接输入已保存的旧 winner。

encoder 调用先持久记录 visual_encoder 预留，planner 必须通过计量器执行至少一次计量回调。每次计量在进入回调之前检查该臂显式上限并记录 attempt，正常返回才记录 return；异常留下 pending，停止该臂，不自动重试。receipt 独占创建并 fsync，再以摘要绑定 journal 的 selection；若写入完成后 selection 失败，孤立 receipt 不是成功证据。journal 自身写入失败可能进入 poisoned 状态，应保留不完整记录，不继续调用。

`consume_online` 先由原 journal consumer 重建并验证已封口日志，再核对 receipt 摘要、当前来源文件、候选、对应 observation/计量 attempt/return/selection 顺序及身份。它不重执行回调。来源文件必须在消费时仍可访问且字节不变；校验文件哈希并不证明文件内容就是所声称模型、任务或环境。

局限：计量数由回调主动申报，接口不能阻止回调绕过计量，也不提供硬件、模型执行或物理执行证明；帧只校验字节摘要，不校验解码后的 RGB 尺寸或感知真实性。来源闭包是调用者提供的文件清单，不证明依赖完整；动作和 latent 的物理语义需要真实 adapter 单独验证。预算 cap、attempt 和 return 均不证明等 FLOPs、等总成本或比较公平，未知 FLOPs 保持 null。真实 Full D2P / SRA 在线模型、动作归一化、资源计量和物理执行的接线及其独立验证仍未完成。测试替身、旧保存候选和这里的 journal 不构成物理结果或方法效果证据。
