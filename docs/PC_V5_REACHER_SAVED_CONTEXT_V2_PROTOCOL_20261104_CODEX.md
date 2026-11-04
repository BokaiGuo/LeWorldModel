# V5 Reacher 已保存上下文：设备权限重试

实验内容与 [v1 协议](PC_V5_REACHER_SAVED_CONTEXT_PROTOCOL_20261104_CODEX.md)相同。v1 在输出 manifest 后、分配模型前因沙箱无 `/dev/nvidia*` 退出，未产生计划或物理派发；保留其 manifest。v2 仅使用新输出前缀并在可访问 GPU 的授权环境重试，额外封存本协议及包装器，其他方法、上下文和预算不变。
