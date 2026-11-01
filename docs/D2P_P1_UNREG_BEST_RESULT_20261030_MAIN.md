# P1 最佳迭代去除信赖惩罚：开发结果

[固定开发合同](D2P_P1_UNREG_BEST_PROTOCOL_20261030_MAIN.md)的 24 条
无惩罚计划已在主机 CUDA 生成，全部物理回放并通过保存消费。
冻结模型哈希前后一致；相同中点与第一包已与信赖版逐槽核对。
新计划的末次和初始分数与旧无惩罚版本逐槽完全相同，表明本次
优化路径与旧版一致；其中 3 条选择中间迭代，其余选末次。

与信赖系数 0.1 的最佳迭代对照，24 条第二包全部不同。信赖版
原生成功 2 条，无惩罚版 1 条；两条信赖版成功均为根 0 的 zero
前缀，其中 off 条件的成功在去除惩罚后消失，on 条件两版均在
第十步成功。按原合同停止行的固定终点删失，不能补作十步端点。

22 条两侧均有完整端点的配对中，无惩罚版物体位置、角度、推杆、
联合位置误差分别有 12、6、8、9 条更低；均值差（无惩罚减信赖）
分别为 -0.552、+0.01571、+0.951、+0.477。它没有在本面板
提供一致改善或新增成功，因此不能简单断言移除信赖惩罚修复了
P1；结果也不排除其他候选、任务或独立根上的不同表现。

- [CUDA 计划](../le-wm-main/results/codex_p1_unreg_best_20261030_main_RUN01.json)
  SHA256 `8cc2ce5e72032dc3966085523a502b2c564a9c3f1cde6acbf0ad7b2bd091c8c8`。
- [物理报告](../le-wm-main/results/codex_p1_unreg_best_physical_20261030_main_RUN01/run_report.json)
  SHA256 `83099e4b6334086c0550dfbb6e5fd16a102a544c2f1ffeb54cd78bfebc12df4a`；
  [保存消费](../le-wm-main/results/codex_p1_unreg_best_physical_20261030_main_SAVED_RUN01.json)
  SHA256 `a2139af7ba866b9de3b6ade2ebbba8e8af380f6947bcd798aa02a13c0ace097f`。
- [逐槽配对](../le-wm-main/results/codex_p1_unreg_best_compare_20261030_main_RUN01.json)
  SHA256 `2d17e487bbcab49805ab5cc65f0b6822d68c65c2c8a870d9ed0563b1409c194f`。

两版后缀显式路径均为每行编码 2、预测 22、反向 20，不等于
包含结构计算的完整等预算认证。中点来自封存真实 RGB 后整体
物理重放，不是实时在线重规划；仅有两个已知根且有重复轨迹，
不能据此宣称独立泛化、P1 机制或强基线胜出。
