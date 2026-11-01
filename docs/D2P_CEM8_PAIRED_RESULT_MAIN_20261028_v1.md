# 同八 anchor：CEM 与谱 teacher_full 的实际配对结果

MAIN 已完整审读最终分析器 source/tests/innerdoc，核对冻结 SHA，并实际运行 60 项测试全部通过。此前两项 ABI/先验源检查测试失败已修复；不将它们计为理论负结果。

严格 saved 分析已退出 0，保留两个完整物理消费者与全部 spectral24 分母，不混 observed4。结果：`le-wm-main/results/agent_CEM8_physical_analysis_20261028_v1_MAIN01.json`，SHA256 `a3f874328acec265428c95c209aed3dc3fb5fcaebb2f822a3e6daeec5ac4d370`。

MAIN 另以不导入 planner/consumer 的脚本直接读取两份原物理 report，逐次重算返回步的 native7 reward/success 与八对 endpoint。结果：`le-wm-main/results/recompute_CEM8_pair_MAIN_20261028_v1_MAIN01.json`，SHA256 `dd546eb2dd9c127b9d2a95961ed7be21e961f5f3fd0d7c33ff50f1231929288e`。

| 指标（固定八 anchor，两个已知根） | CEM64 | 谱 teacher_full |
| --- | ---: | ---: |
| 原生成功 | 1/8 | 1/8 |
| 实际 endpoint 联合 XY 均值（px） | 76.44600 | 53.73473 |
| 实际 endpoint native7 距离均值 | 199.48674 | 150.74741 |

CEM−谱的平均联合 XY 为 +22.71127 px；CEM 仅两对联合 XY 更低。成功分支并不相同：CEM 是 slot0/off/D2P prefix，谱是 slot0/on/D2P prefix。不能因各 1/8 就推断动作或机制相同。

这是各自真实停止时的 endpoint 比较，不是统一 H10：CEM 一支 3 步成功停止、谱一支 2 步成功停止；共同实际 H10 仅 6/8，不补步、不输出完整八项 H10 平均。两边搜索预算不同，不能宣称等总算力/FLOPs 优势；八 anchor 不是八独立根，不作独立八样本推断。

结果保留了真实控制差异，但这些前缀本已立即改善任务误差，仍不是“暂时退步→能力改变→任务收益”的 C2 见证；独立 P1 和完整研究闭环均未通过。分析本身没有新增神经或物理调用，上游成本并非零。
