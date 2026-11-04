# JEPA 原生物理规划边界重放结果

按[事前协议](PC_JEPA_NATIVE_BOUNDARY_REPLAY_PROTOCOL_20261104_CODEX.md)，在已有隔离依赖中运行原生 PushT，从固定 seed 2026102701 reset 后执行历史非零暖身的前 10 条命令。新记录的帧 0–10 与历史同 seed a 臂在五种文件上均逐字节一致：RGB、状态、proprio、声明的公共物理状态和 transition，共 55/55 项。执行计数为一次构造、一次 reset、十次环境 step、101 次 0.01 秒物理子步、十一次采集和一次关闭，均符合预定合同。

[原始结果](../le-wm-main/results/pc_jepa_native_boundary_replay_20261104_codex/result.json)保存全部命令、逐帧哈希、实际调用计数和日志绑定；[独立保存数据审计](../le-wm-main/results/pc_jepa_native_boundary_replay_20261104_codex/audit.json)还检查新旧日志的哈希链，并将历史帧文件与历史日志的原始收据逐项比对。审计没有再次运行物理环境。运行器及其输入记录在 `le-wm-main/experiments/pc_jepa_native_boundary_replay_20261104_codex.py` 和结果目录中，旧记录未被改写。

这个结果支持在**同一软件和环境配置下，通过 reset 与已执行动作重建该已暴露规划边界**；不是仅从七维状态重置，也不证明私有接触缓存可序列化。未来比较必须分别重放并在派发前核对各臂边界，或保持同一实时环境连续执行。未生成目标、未调用模型、未执行规划动作；因此没有新的控制收益或 PC gate 证据。该根不是独立任务或训练外来源认证。

实际使用的覆盖目录提供 Gym 0.23.1、Pymunk 6.8.0 和 scikit-image 0.25.2；主项目环境的不同版本不适合直接冒充这一原生运行时。模型计算应保持独立进程，不在 GPU 进程加载该物理依赖覆盖层。下一步需要另立事前协议，绑定合法目标与隔离规划输入，先将同一 v7 编译器输出封存，再将动作送入已核验的物理边界，并逐轮记录反馈与真实任务指标。
