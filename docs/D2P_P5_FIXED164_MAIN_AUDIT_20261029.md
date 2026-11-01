# P5 fixed164 保存结果接线：主线程核验

2026-10-29：主线程审读 `le-wm-main/experiments/spectral_reachability/agent_p5_fixed164_physical_model_join_20261029_v1.py`、其测试及配套说明，独立运行测试得到 18 passed；对四份固定 SHA 的真实封存输入运行 `analyze`，返回 164 行、80 对、2 个已知根、0 个认证新根，P1/P5 admission 均为 false。此次仅输出检查摘要，没有新增神经预测或物理执行。

接线逐行核对根、条件、任务、选取身份、raw float32 动作、producer、实际第10步 PNG/state 和 native 终态；80 对按固定顺序保留，四个 baseline 也保留。来源完整性依赖已有的 strict receipt，并非此次重新执行 full strict。LeWM goal Q 与 JEPA 共同标签 MSE 属于不同表示空间和输入权限，不可直接相减或排序；164 分支不是 164 个独立根。

这是一项已见数据的身份及指标接线检查，不是新根确认、物理收益、跨模型公平比较或 P5 科学 gate 的通过。

随后主线程审读并测试持久化包装，join 与包装联合 28 项测试通过；对真实封存输入写出 `le-wm-main/results/agent_p5_fixed164_physical_model_persist_20261029_v1_MAIN01.json`，SHA256 `25cb6846263027b1ceedd39b9060c3158c6cc6a095d323b012be30728e7511c3`，独立 verify 入口重建全部内容逐字节通过。该 verify 复用相同 join 实现，也不重跑上游 full strict；SHA 不等于外部时间认证。
