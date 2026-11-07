# TwoRoom 小扰动稳定选择：开发结果

按[协议](PC_TWOROOM_LOCAL_PERTURB_STABILITY_PROTOCOL_20261107_CODEX.md)，
在 10 个已曝光任务各取模型最低分的 8 条父链，对每条尾部构造并评分
4 条合法小扰动；选择封存后，真实环境各执行三臂选中的完整动作链。

| 选择规则 | 25 步内曾成功 | 末步成功 | 前 10 步已成功 |
| --- | ---: | ---: | ---: |
| 全部 96 链中模型最低分 | 8/10 | 6/10 | 3/10 |
| 小扰动均值选择父链 | 8/10 | 6/10 | 3/10 |
| 扰动身份打乱 | 4/10 | 2/10 | 3/10 |

小扰动均值相对直接最低分独胜、独负均为 0，**未达到预定开发出口，
停止扩大此固定规则**。虽然与身份打乱的结果不同，但打乱本身破坏了
父链与扰动评分对应关系，不能据此证明该方法优于直接搜索或证明行动
后果描述有增量。

直接最低分在 8 根选择了新扰动链；均值臂在 8 根与直接臂选择不同。
局部扰动在六维动作块上的实际平均欧氏距离为 0.119，最大 0.246；
1,920 个扰动块坐标中有 28 个发生裁剪。三臂都使用原有 64 条及本次
32 条模型评分，原有候选生成、A 查询及物理执行另计。本次在 GPU
新增 320 条评分，真实环境执行 30 条链、750 步；不是完整反馈成本比较。

定向测试 2 项通过；22 条选中链属于旧 B 池，其逐步重放距离与成功标记
分别与旧轨迹完全一致。其余 8 条为新扰动链，未进行独立环境重放；
未单独重复 GPU 评分。Gymnasium 仍报告 reset 类型和动作转换警告。
任务与方法选择高度曝光，结果不是独立确认；该实验也未验证行动后果
描述或程序保留，PC0–PC5 不改判。`experiment-code` 用于分离 GPU
选择封存与物理评价、固定种子和记录失败结果。

[封存选择](../le-wm-main/results/pc_tworoom_local_perturb_stability_20261107_codex_plan.json) ·
[模型扰动池](../le-wm-main/results/pc_tworoom_local_perturb_stability_20261107_codex_pool.npz) ·
[物理轨迹](../le-wm-main/results/pc_tworoom_local_perturb_stability_20261107_codex_physical.json) ·
[实现](../le-wm-main/experiments/pc_tworoom_local_perturb_stability_20261107_codex.py) ·
[测试](../le-wm-main/tests/test_pc_tworoom_local_perturb_stability_20261107_codex.py)。

## 后续勘误

原打乱臂交换原始扰动分数，同时混入不同父链的基准分数，不是纯效应
身份对照。[事后修正](PC_LOCAL_PERTURB_EFFECT_SHUFFLE_CORRECTION_RESULT_20261107_CODEX.md)
只错配相对父链变化量，重放结果为 9/10；原 4/10 不应作为身份贡献证据。
直接最低分与匹配均值的原比较和停止结论不变。
