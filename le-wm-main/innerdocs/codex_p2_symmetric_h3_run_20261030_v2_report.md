# P2 对称工作量 H3 已知根探索

本次新 H3 合同不是旧 H2：四个已见根各绑定原 context、goal、rank-4 投影 Q 和模型，目标为 `mean((Q @ (z_terminal - goal))**2)`；联合 30 维 logits L2 半径 `0.75*sqrt(15)`，joint 对全变量更新，suffix 两次分别固定前 10、20 维。超界候选在不扩大半径的前提下径向向内舍入，suffix 保持固定前缀。两臂各固定两次 full 192×30 J、两次同形状 SVD、8 候选、两次完整三段 replay。模型 frozen，CUDA 单卡，未训练或运行物理。

合同在本次 v2 运行前固定，但已经看过旧结果和 v1 部分执行，不能叫预注册或独立确认。准备 SHA `b20ac436c23d697fed53d3b78423f1ebd6861428fd512c7513743b53d2236c95`。输出 `results/codex_p2_symmetric_h3_run_20261030_v2_RUN01`，report SHA `4d0a4f4399227609548678329df506e2f0f6e7e6fe0ce49b0690f0031521b90f`。

八槽全部完成，逐槽真实 hook 记录 attempt/return 各 39 packet、39 action_encoder、39 predict、2 Jacobian、2 SVD；每次 Jacobian 请求 192 数学 VJP 行，非实际 engine backward 计数。共同同步 wall 截止 450 秒；各槽实际约 0.428–0.643 秒，含同步及事件 fsync。每槽执行两次完整重放，三个保存 context 与 winner 全部精确相等。模型参数及 buffer hash 前后一致。独立读盘 audit 核 journal 哈希链、固定顺序、计数、L2 及保存 replay；没有独立重算神经输出、J、优化轨迹或所有模块内部工作，也没有保存 J 矩阵。

描述性原模型 Q（越小越好，仅供这四根已知根的本次探索）：

| root | joint | suffix |
| --- | ---: | ---: |
| 0 | 1.474871 | 1.525706 |
| 1 | 1.240401 | 1.271955 |
| 2 | 1.274898 | 1.304648 |
| 3 | 1.517492 | 1.579258 |

v1 RUN01 在沙箱缺设备时失败；v1 RUN02 在 root3/suffix 的 L2 浮点边界拒绝，保留六个完整槽和第七槽失败，未删除或覆盖。v2 仅修复投影向内数值安全，重新固定合同后跑全部八槽；不把 v1 部分结果拼入 v2。

这是等指定操作次数及共同 wall 上限，不是相同实际 wall 或全部算力：矩阵内容不同，FLOPs、引擎内部 backward 和完整加载/上下文成本未知；加载约 4.89 秒属于共享开销，未纳入每槽搜索窗口。没有全局 GPU 独占保证。P1/P2 准入和物理许可仍为 false，model Q 不是物理任务成功或因果机制证据。

复验（CPU 读盘，不重新执行模型）：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_p2_symmetric_h3_run_20261030_v2.py tests/test_codex_p2_symmetric_native_bridge_20261030_v1.py tests/test_codex_p2_symmetric_budget_20261030_v1.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.codex_p2_symmetric_h3_run_20261030_v2 audit --output /home/gbk/code/LeWorldModel/le-wm-main/results/codex_p2_symmetric_h3_run_20261030_v2_RUN01 --sha 4d0a4f4399227609548678329df506e2f0f6e7e6fe0ce49b0690f0031521b90f
```

测试 12 passed；audit `journal_and_saved_replays_checked`。audit 不是独立模型重执行，也不是科学 gate。
