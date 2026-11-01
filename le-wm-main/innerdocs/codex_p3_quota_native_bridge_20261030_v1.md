# P3 固定候选配额的原生接线状态

新桥接 `codex_p3_quota_native_bridge_20261030_v1.py` 调用既有 `moving_history_budget_scheduler_cuda_20261027_v1.Native`，并对两个来源任务分别使用该 backend 的 H2 rollout、原 `check_path` 和原 model-Q `cost`。调用方必须传入已加载的 CUDA 模型、任务、候选、计时同步和事件 sink；CPU device 拒绝。模块自身不是可生产的 launcher。

固定三策略、两个根、每根前 15 项，逐策略每根都独立真实评估 15 项；资格配额分别 10/10、15/5、5/15，权重完全相等时三策略均 10/10。每根分别 argmin，不能跨不同根的 model-Q 比较。每项 H2 真实成功返回要求 action_encoder、predict、predictor、pred_proj 各恰好两次；三臂合计每类 180 次，每臂每根每类 30 次。前置候选/总量硬上限及 attempt/同步 return 分离；异常不退款、不自动重试。metric 是实际 Python 函数调用，但该桥没有独立 metric 时间/次数 hook，不把它与 GPU 神经开销混同。

这是“相同面板计算、不同候选胜出资格”的探索控制，不是计算配额随信号变化的节省算力实验。15/5 与 5/15 只改变资格，不改变每根实际工作；额外评估的候选属于 paid shadow work。全流程加载、原始谱信号、物理、wall、FLOPs 均未等价。固定相同候选不保证动作不同；原策略碰撞和历史结果保持不变。旧单根跨组 argmin 接缝不应作为本桥的决策。

生产前置仍缺：固定准备文件及来源闭包、完整逐任务槽/原始动作身份、耐久事件和失败记录、模型前后与设备身份、锁内加载及清理、资源/磁盘限制和独立 saved consumer。当前回调测试并未调用真实模型或物理，不能据此断言 native CUDA 已实测。必须单独完成这些部分后，才可在 CUDA 初始化前取得 `/home/gbk/code/LeWorldModel/le-wm-main/results/e3_p2_host_cuda_cooperative.lock` 的现存 inode、持锁至清理退出并启动独占新输出；锁只是合作互斥。固定面板产生的模型赢家也不是物理执行授权，物理需另行绑定同根/目标/动作与预算并校验。

运行离线测试：

```
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_p3_quota_native_bridge_20261030_v1.py tests/test_codex_p3_fixed_panel_quota_20261030_v1.py
```

没有新 CUDA、物理或科学 gate 通过的结论。
