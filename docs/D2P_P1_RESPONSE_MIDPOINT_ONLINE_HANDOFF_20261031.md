# P1 新响应前缀：现场中点求解一致性运行

按 `D2P_P1_RESPONSE_MIDPOINT_ONLINE_PARITY_PROTOCOL_20261031.md`
已封存 13 可执行槽、3 历史删失的固定输入：
`le-wm-main/results/p1_response_midpoint_online_20261031_prepare_RUN01.json`
SHA-256 `86a81ddb5252ca181fd1f53e7b43a2cdae639256cf9a38c0b83217141a7269a0`。
入口 `experiments/spectral_reachability/p1_response_midpoint_online_20261031.py`。
准备重建与中点派发前错配拒绝测试共 3 项通过。

已启动主机 CUDA 与物理同进程执行，句柄 48246，输出目录
`le-wm-main/results/p1_response_midpoint_online_20261031_RUN01`。
每条在真实中点用现场 RGB 执行 Adam，生成的后五步必须与已封存
离线结果逐字节一致后才能派发；现场 RGB、状态和快照也先核对。
若结果不同，保留失败记录并区分设备／数值／接线问题，不以
封存动作替换现场结果。只读轮询超时不能作为重启理由。

此任务是已知根上在线计算与物理接线一致性，不是新根、等预算
或非贪心机制确认。结束后须对报告、逐行轨迹、现场规划事件、
动作派发顺序和模型哈希进行保存态验收，再对照离线回放；退出
码零不能代替完整验收。
