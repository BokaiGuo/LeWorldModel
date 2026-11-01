# P2 H15 真实任务桥接及双根 fixture 生命周期

离线桥接重建 P2 来源清单、固定的 P1 joint prepare 和历史 P1 `task_from`，对两根的 task ID、压缩文件 SHA、完整上下文、goal、Q 和归一化做连接。P1 native ABI 的 root 是历史 producer 从 task 标识派生的命名空间占位标识，不是 P2 物理 root 内容 SHA；两者分别保存。native model 为历史预期 state-dict SHA，checkpoint 是文件 SHA，二者通过共同历史 prepare 和 task 绑定，不是对当前加载模型的证明。P1 producer 的其他派生绑定也不能解释为原始物理证明。

双根执行接口仅允许 RAM fixture，串行调用既有单槽 `run_fixture`；写入 start、seal 或 fail 哈希链并保存报告。成功槽调用既有完整 saved consumer；失败槽复核资源账和已存日志，外层异常仅保存异常声明，不冒充 native 审计。固定分母为二，失败后不启动下一根；独立消费以外部报告 SHA 开始，重新构造桥接并重放生命周期、槽报告及相应审计。日志及报告均独占创建。I/O 故障或进程崩溃可能留下无最终报告的部分记录，不能声称总能持久化。

此路径没有调用单槽 `validate_case` 的错误跨命名空间相等假设；桥接按字段来源逐项核对。HOST 入口无条件拒绝。尚缺：将经过认证的桥接接到真实 native HOST callback、实际模型 state-dict 与精度前后测量、源码及依赖完整闭合、调用预算及宿主资源计量、物理 adapter 和其独立证据。RAM fixture 的事件、零物理值和离线映射不构成 HOST 或物理证明；saved consumer 也不能检测没有被记录的调用。
