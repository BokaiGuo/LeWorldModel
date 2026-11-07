# 小扰动身份打乱对照：事后修正结果

上一轮的“身份打乱”把另一父链的**原始扰动代价**放到当前父链，
因此同时交换了父链自身的基准代价。它与原协议一致，但不是只打乱
效应的合适对照；先前的 8/10 对 4/10 不能用于解释动作身份贡献。

按[修正协议](PC_LOCAL_PERTURB_EFFECT_SHUFFLE_CORRECTION_PROTOCOL_20261107_CODEX.md)，
将每个扰动代价先减去其自身父链代价，仅错配所得变化量；当前父链的
基准代价保持不动。修正后的选链先封存，再实际执行 10 条完整链。

| 规则 | 曾成功 | 末步成功 | 前 10 步已成功 |
| --- | ---: | ---: | ---: |
| 直接最低分（原记录） | 8/10 | 6/10 | 3/10 |
| 小扰动均值（原记录） | 8/10 | 6/10 | 3/10 |
| 原始分数打乱（原记录） | 4/10 | 2/10 | 3/10 |
| 效应变化量打乱（本次重放） | 9/10 | 6/10 | 4/10 |

修正打乱与原打乱在 8/10 根选择不同；相对匹配小扰动均值独胜 1 根、
独负 0 根。10 条修正对照链与原固定池轨迹逐步比对，距离最大差为
0，成功标记无差异。测试 4 项通过，包括对每个父链同时加入不同常数
时，修正对照只按自身基准相应平移的检查。

这证实原对照存在**实验设计混杂**，不是原程序偏离其书面公式的实现
bug，也不改变直接最低分与匹配均值的 8/10 对 8/10 及其停止结论。
新对照错配的是不同父链各自随机产生的一整组扰动效应，不是严格的
相同动作跨父链身份实验；更不是行动后果描述的独立收益证明。
面板和物理结果早已曝光，本次不构成独立确认或 gate 通过；环境仍有
Gymnasium reset 类型及动作转换警告。

`code-debugging` 用于区分公式实现正确性与对照设计的因果混杂。
[封存选择](../le-wm-main/results/pc_local_perturb_effect_shuffle_correction_20261107_codex_plan.json) ·
[物理重放](../le-wm-main/results/pc_local_perturb_effect_shuffle_correction_20261107_codex_physical.json) ·
[代码](../le-wm-main/experiments/pc_local_perturb_effect_shuffle_correction_20261107_codex.py) ·
[测试](../le-wm-main/tests/test_pc_local_perturb_effect_shuffle_correction_20261107_codex.py)。
