# JEPA PushT v7：输入隔离开发回归

## 结论与边界

已按独立协议完成一次同根、同模型的规划输入隔离工程回归。准备进程认证原始封存并导出白名单输入；单独的 bwrap 规划进程未挂载原轨迹与旧结果，五项指定路径读取探针均返回 `FileNotFoundError`。新旧运行的全部请求动作、投影位置、模型评分及目标 chart 逐元素一致。这支持本次实现对旧运行的数值等价，并修复该新运行对已知原轨迹路径的直接读取风险；不使旧运行本身重新满足输入隔离。

这是已暴露 episode 的模型内诊断，不是物理执行、RGB-only、独立 checkpoint 或来源外确认，也不能据此宣称 PC0–PC5 通过。原轨迹目标来自已暴露的同一 episode；checkpoint 训练成员及 normalizer 拟合成员独立性仍未知。挂载策略遮住了用户 home 和 `/tmp` 中的已知数据，但系统其他位置可能有副本；其不是对恶意代码或全部信息流的形式化证明。

## 实际结果

| 臂 | 模型内最终 MSE | token | 批次 | 轨迹样本 |
|---|---:|---:|---:|---:|
| v7 response | 0.06233594939112663 | 1920 | 60 | 737 |
| smooth CEM | 0.06012116000056267 | 1920 | 55 | 640 |

逐批请求动作、投影位置、分数的最大绝对差均为 0，目标 chart 的最大绝对差为 0；这只是此次同硬件配置上的精确回归，不是跨平台确定性承诺。两个臂的模型内分数不等价于真实 PushT 成功率或强基线确认。

输入包包括三帧历史 RGB、目标 RGB、三帧归一化 agent 位置和速度、两段已执行动作 packet，以及动作均值与标准差；非 RGB 的本体信息是模型合法输入，不能称为纯视觉。禁止路径探针包括原始视频、状态、相对动作、原始 manifest 和旧结果。模型参数收据、checkpoint 和源码校验均由规划 worker 检查；此处只对已声明的来源范围负责。

## 证据与复现

- 事前协议：`docs/PC_JEPA_V7_ISOLATED_INPUT_DEV_PROTOCOL_20261104_CODEX.md`。
- 准备器：`le-wm-main/experiments/pc_jepa_v7_isolated_prepare_20261104_codex.py`。
- 隔离启动器及 worker：`le-wm-main/experiments/pc_jepa_v7_isolated_launch_20261104_codex.py`、`pc_jepa_v7_isolated_worker_20261104_codex.py`。
- 结果：`le-wm-main/results/pc_jepa_v7_isolated_input_dev_20261104_codex/output/result.json`；SHA-256 `72a3ececbcabc7d787caf56442c9ba2d32b602ebcb77e4db67bd91d16a70fb35`。
- 逐批独立对比：同目录 `audit.json`，审计入口 `experiments/pc_jepa_v7_isolated_audit_20261104_codex.py`；旧运行哈希和五项探针均记录其中。
- 准备、结果及审计采用独占创建，已有文件时拒绝覆盖。复现需先提供新的独占输出位置并相应新建协议版本；不能直接覆盖本记录。运行依赖本地 bwrap、封存原始数据及依赖、固定 GPU UUID、CUDA 和约定的共享锁。

执行过 CPU 隔离视图预检与一次完整 GPU 回归；没有运行真实环境、物理轨迹重放或独立任务统计。下一步应按既定合同推进同一正式方法的物理闭环与来源核验，而不是把此单根回归升级为科学结论。
