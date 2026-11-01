# P1 新响应前缀中点重规划：物理回放交接

固定协议 `D2P_P1_RESPONSE_MIDPOINT_PHYSICAL_PROTOCOL_20261031.md`。
准备文件
`le-wm-main/results/p1_response_midpoint_physical_20261031_prepare_RUN01.json`
SHA-256 `3d70d2feb2a05ff8241c63a4c00cf3a5d4f5a73d6e5450f5a4f5f2bf37541cf4`，
13 条实际轨迹、3 条源删失。运行入口
`experiments/spectral_reachability/p1_response_midpoint_physical_20261031.py`；
在后缀第六步派发前，强制比对所选 RGB、状态及快照，错配即停止。
准备重建、模拟错配与源绑定测试 3 项通过。

单次物理运行已启动，输出目录
`le-wm-main/results/p1_response_midpoint_physical_20261031_RUN01`，
执行句柄 85749。启动阶段需要重建并检查来源，目录尚未出现
不表示运行失败；以实际进程和句柄为准。此过程不执行神经推理。
不得覆盖输出或仅因为轮询超时重启。完成后需要以实际报告哈希
运行入口的 `consume` 命令，并进行同一源轨迹的结果配对；若
中点检查失败或有其他工程错误先保留全部产物、查明原因。

已有先前主机 CUDA 计划是离线计算，本次回放不是实时在线求解。
没有独立根、完整总预算或 P1 准入证明。
