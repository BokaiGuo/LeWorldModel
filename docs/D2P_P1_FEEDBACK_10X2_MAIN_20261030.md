# P1：反向次数匹配的反馈探索

在已知结果之后固定完整 24 行，沿用既有真实前缀、冻结 checkpoint
和相同两阶段真实 RGB 反馈执行器；每阶段 10 次更新，不训练模型。
执行在主机 CUDA 和合作锁下完成，24 条真实轨迹全部执行满十步；
保存消费逐行检查动作、前缀、图像、反馈顺序及事件账本，24 条通过。
此前第一次消费时误输入报告哈希，校验拒绝；正确哈希重试通过，
没有修改实验数据。

| 比较 | 反馈 10+10 | 既有开放 20 |
| --- | ---: | ---: |
| 原生成功 | 0/24 | 0/24 |
| 反馈组物体位置误差更低 | 4/24 | — |
| 反馈组物体角度误差更低 | 10/24 | — |
| 反馈组推杆位置误差更低 | 0/24 | — |
| 反馈组原生联合位置误差更低 | 0/24 | — |

所有比较按行身份配对、负差代表反馈误差更低，不把共同根重复
轨迹当成独立样本。反馈两阶段各扣 10 backward、22 predict 和
22 action-encoder，另各编码一次；开放路径扣 20 backward、
42 predict 和 42 action-encoder，另编码一次。这些是账本
reservation，非完整 hook 覆盖认证；不同资源轴不可相加。两侧
只匹配 backward 数，不是等总成本。反馈第一阶段仅做 10 次更新，
开放做 20 次；因此本比较同时改变优化分配和反馈，**不能归因
为反馈本身无效**。未观察到成功也不证明理论不成立。

根 0/on selected 的两条重复前缀，反馈相对开放的物体位置误差
降低约 22.18，但推杆误差增加约 27.98、原生联合位置误差增加
约 14.71；这与分项任务评价失配的线索一致，不证明唯一根因。
实验为事后开发，并非独立确认或强基线对照；P1 仍未通过。

协议 `le-wm-main/results/codex_p1_two_stage_launcher_20261030_v1_prepare_10x2_MAIN01.json`
SHA `3e31c2c8d16b4fe8c3862985a2c149c559c57311ca68e98d7f2020366c711468`；
运行报告 `le-wm-main/results/codex_p1_two_stage_launcher_20261030_v1_run_10x2_MAIN01/run_report.json`
SHA `4c2916154862f8316ef070ea39d59f48e50b97620fe3b8021fa60b60e94fcbd0`；
保存消费 `le-wm-main/results/codex_p1_feedback_10x2_20261030_main_saved_MAIN01.json`
SHA `554c7d6a965916382a8fb1604f0b408882a63411bcd5768f590db6cdd3d172b5`；
配对 `le-wm-main/results/codex_p1_feedback_10x2_pair_20261030_main_RUN01.json`
SHA `50b2a9d0a38583cd3e6b6d30cd65e8b0bde6cf9189b445264b72f9b3fc3a7aa0`。
新增协议和配对测试 2 项通过。
