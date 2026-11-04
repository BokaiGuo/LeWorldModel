# 后缀尺度下限新版本：精确动力学开发结果

## 完整预算下的五臂比较

按[事前协议](PC_SUCCESSOR_FLOOR_CEM_EXACT_DEV_PROTOCOL_20261103_CODEX.md)，
对 24 个已暴露的相同 Wall 任务运行新 successor 和四个对照，每根每臂
最多 480 个五步预测 token。五臂统一使用标准差下限 0.05 的后缀／全时域
CEM；原联合精修未修改。生成与筛选、精修、最终 zero-prefix 比较都计入
预算；没有解析见证输入。

| 臂 | 原下限平均终距 | 新下限平均终距 | 新下限成功数 | token／根 | SVD／根 |
|---|---:|---:|---:|---:|---:|
| successor | 0.52813 | 0.52791 | 0/24 | 480 | 3 |
| 同库 greedy | 0.51611 | 0.51084 | 0/24 | 479 | 1 |
| 随机桥接 | 0.52655 | 0.52561 | 0/24 | 480 | 1 |
| lookahead | 0.51613 | 0.51771 | 0/24 | 479 | 1 |
| 完整 CEM | 0.52728 | 0.51259 | 0/24 | 480 | 0 |

成功阈值 `<0.05`。新 successor 相对于新 greedy、随机桥接、lookahead、
完整 CEM 的逐根终距均值差分别是 `+0.01707 / +0.00231 / +0.01020 /
+0.01532`（正数表示 successor 更差）；对应 successor 胜出的根数为
`5 / 11 / 8 / 8`。这些是暴露根的描述性比较，不是独立统计确认或
强规划器上界。token 近似匹配但结构运算不同，不等于全部计算资源相等；
新五臂各 24 根决策的测得 wall time 总和依次约 `0.666 / 0.334 / 0.467 /
0.457 / 0.150` 秒，为同进程本机开发记录，不是硬件无关的基准。

successor 四个前缀各得 `24 / 23 / 23 / 23` 个后缀样本，只有两批；
24 根中仅一根的最终终距严格优于旧下限版本，8 根最终选择 zero-prefix
对照。这解释了为什么上一轮每个固定前缀 384 样本的可行见证诊断不能
直接外推到完整方法：预算、前缀和选择均不同。当前结果没有消除候选覆盖
问题，也没有证明理论不成立。

## 最终链机制筛查

完整计划封存后，按独立的[事后开发协议](PC_SUCCESSOR_FLOOR_CEM_EXACT_MECHANISM_DEV_PROTOCOL_20261103_CODEX.md)
对最终选择链、零前缀和同库 greedy 做相同六后缀族及各 120 次同优化器
重规划。非即时进展 9/24、相对零后缀的选中后缀变化同时优于两控制 8/24、
自行重规划同时优于两控制 4/24；三条件同链交集 **0/24**。这是有限
后缀和有限预算下的筛查，不代表可行性上界，也不是 PC2 确认。

## 实现与核验

新增可供预测器注入的[首次决策编译器](../le-wm-main/experiments/spectral_reachability/pc_successor_floor_compiler_v1.py)
和[尺度下限 CEM](../le-wm-main/experiments/spectral_reachability/pc_cem_scale_floor_v1.py)；
它们尚不是多轮 RGB 正式闭环。旧下限兼容性测试覆盖末批处理，并对一根
五个臂的旧封存选择、动作、分数和 token 做等价回归。120 次新决策的每个
预测批次已重算并验证所选动作和预算；24 根机制干预的共同后缀与 8,640 次
重规划评分也由测试重算。相关测试 **5 passed**。

原始[决策](../le-wm-main/results/pc_successor_floor_cem_exact_dev_20261103_codex_plans.json)、
[预算和回放审计](../le-wm-main/results/pc_successor_floor_cem_exact_dev_20261103_codex_audit.json)、
[最终链干预](../le-wm-main/results/pc_successor_floor_cem_exact_dev_20261103_codex_mechanism.json)
均绑定协议与依赖。PC2、PC3 仍证据不足；PC4、PC5 不能以此精确状态实验替代。
