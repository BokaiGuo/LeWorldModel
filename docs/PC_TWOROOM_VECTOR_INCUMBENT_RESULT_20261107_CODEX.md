# TwoRoom：把已评分原链保留到最终竞争

在 40 个已曝光任务上，复用原有向量保留实验的四臂计划及其前缀来源，将 block-CEM 与 block-random 各 300 条候选中模型评分最低的完整链加入每臂最终竞争。模型在读取任何物理结果之前重新评分并封存选择；随后评价端在相同初态重放六条链。仅改变最后竞争候选，未改变原候选生成或保留规则。

| 规则 | 原曾成功 | 加入 incumbent 后曾成功 | 新策略独胜 / 独负 | 选择来源改变 |
|---|---:|---:|---:|---:|
| 父链 | 34/40 | 39/40 | 5 / 0 | 36 |
| 普通 lookahead | 32/40 | 38/40 | 6 / 0 | 37 |
| 向量多样性 | 36/40 | 38/40 | 2 / 0 | 37 |
| 动作身份打乱 | 34/40 | 38/40 | 4 / 0 | 37 |

两条原池赢家单独执行均为 **38/40**。加入竞争后，向量臂与普通 lookahead 的曾成功结果在 40 个任务上完全一致，成对独胜／独负 **0/0**；打乱臂同样如此。向量臂最终有 27 次选择 CEM 原链、10 次选择随机原链、3 次保留原向量臂所选链。因此先前向量臂相对 lookahead 的 4/0 差异，不能解释为完整搜索器相对已保留原链的强对照有优势。

旧四臂各 40 条链重新物理执行的逐步距离、成功及截断标志与历史轨迹一致，最大距离差为 0。原池赢家的复评分与历史分数最大绝对差约 0.000466；旧臂所选链最大差约 0.000199。160 次最终比较中的最小前两名分差约 0.0817，远大于这些观察到的单条分数差；这不是对其他 batch、硬件或候选的普遍等价认证。选择函数、平分规则和相关旧测试共 **7 项通过**。

本次每任务额外联合重评 6 条链，未测得可用于比较的单臂完整墙钟；原前缀来自两组各 300 次评分，原开发方法每臂另有 88 次逻辑评分。不能把本次 6 条计数当作整个规划器成本，也不能把两组共同候选池与仅用 300 次评分的单独基线视作等成本。

**决策：**漏掉已评分原链是具体可修复的选择覆盖缺口，加入后在此开发集改善了各臂绝对成功数；但恢复共同 incumbent 后，当前向量描述没有显示超出 lookahead 的成功增量。该修复不是已经证明的普遍控制改进，模型评分更优也不保证物理表现更好。这些任务已暴露，不能作为独立确认、完整闭环、描述机制或 PC0–PC5 通过证据。进一步研究必须用事前冻结的合法生成及同成本对照，不能继续以原来漏掉 incumbent 的对照宣称优势。

[协议](/home/gbk/code/LeWorldModel/docs/PC_TWOROOM_VECTOR_INCUMBENT_PROTOCOL_20261107_CODEX.md) · [选择记录](/home/gbk/code/LeWorldModel/le-wm-main/results/pc_tworoom_vector_incumbent_20261107_codex_plan.json) · [物理轨迹](/home/gbk/code/LeWorldModel/le-wm-main/results/pc_tworoom_vector_incumbent_20261107_codex_physical.json) · [汇总](/home/gbk/code/LeWorldModel/le-wm-main/results/pc_tworoom_vector_incumbent_20261107_codex_result.json) · [运行器](/home/gbk/code/LeWorldModel/le-wm-main/experiments/pc_tworoom_vector_incumbent_20261107_codex.py)。
