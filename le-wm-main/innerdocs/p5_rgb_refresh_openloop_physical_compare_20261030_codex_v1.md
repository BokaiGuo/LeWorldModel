# 固定已知面板：刷新一步与开环物理绝对误差

仅使用此前已开放 physical receipt、固定 panel 和已接受 journal；不读取未开放 holdout。固定 164 枝、四个 baseline，off/on 各 82，两个已知 root；80 对保留在输入身份，不把它们或同 root 的枝当独立样本。分别严格复核 MAIN02 开环和 MAIN01 刷新 accepted、源码、supervisor、journal 和首尾快照，再核对原始动作、六帧历史、图像 hash、checkpoint 归一化及同枝身份。刷新准备文件从原始公开资产重建验证；开环 journal 原合同验收重放。

时间固定 h=1…5 对齐同一枝 plan6…10 物理 state。每条刷新预测都只用当前及前一真实 RGB 和该步动作；开环则只在 plan5 编码后连续递推。两者 h=2…5 的真实观测权限不同，这是 teacher-forcing 诊断，不是公平闭环对比，也不是物理闭环控制运行。

两种模型输出均使用同一 checkpoint 的原 mean/std 反归一化。物理 state 映射为 `[agent_x, agent_y, block_x, block_y, sin(angle), cos(angle)]`；逐枝、逐 horizon 保存各轴物理值、两种预测、绝对误差及刷新减开环绝对误差。按 off/on 和 horizon 分别对 82 枝求六轴算术平均，不混合坐标或单位、不筛选、不填补。前四轴为模拟器位置坐标，末两轴为无量纲 sin/cos；负差值表示该已知面板的刷新平均绝对误差较小。

这是已知、非独立根上的描述性 saved-only 结果，不是 fresh confirmation、跨模型排名、因果证明或 P5 科学准入。此前训练使用物理状态标签；本对照不做训练、神经推理、CUDA 或物理生产。
