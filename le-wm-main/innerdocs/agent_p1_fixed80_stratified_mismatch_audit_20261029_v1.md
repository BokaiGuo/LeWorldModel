# P1 固定80对：时间、动作与物理指标分层审计

本报告只读已封存 saved 结果，不重新执行旧 fullstrict、神经或物理。规则按既有合同的全部 root/overlay/role、五词两两组合、原生轴和两个已完成时间面板固定，表中包括全部组而非从结果中挑选组；**这次分层是在既有结果已经存在之后制定，不是前瞻预注册**。例子只是对全表中预定义的“位置与原生距离符号相反”类型说明，不是重新挑根做结论。无 selector、Q、gate、阈值修改；两 known roots 不成为80个独立样本。

## 数据与比较规则

主面板：`results/agent_p1_pair_margin_decomposition_20261028_v1_MAIN01.json` SHA `f723c5eb73ecebd0615a2ded0aa4b741c6f132ca5f6063c673f12d5e98539143`；原 `innerdocs/agent_p1_pair_margin_decomposition_20261028_v1.md` 说明此前 MAIN 已验证来源和原 full saved 闭包，本报告只核该文件 SHA 与内部固定分母及分类，不声称重做闭包。8 anchors = 2 known roots × off/on × Z/S，各五词 `(zero,uZ,-uZ,uS,-uS)` 的全部 `i<j` 十对，总80；预测来自实际 teacher context15，编码与物理 endpoint在20。模型端和同臂真实端动作逐位绑定；两臂后缀本来不同，同动作控制不是说 i/j 相同。

全部 margin 定义 `cost_i-cost_j`，距离越小越好；success 比较沿原报告 higher-is-better 后转向，不把平局叫反向。严格反向仅符号正负相反，零/tie 和 unknown 分开。原 source 的 float64 排名不替换；其 binary32 值的 Fraction 代数校验另列，不把差一点当 tie。目标 latent 为原 identity192 MSE、除192一次；物理 goal 是完整 native goal7，末两维为 agent velocity，不能当作零或 object velocity。

第二时间面板仅并列：`results/agent_p1_prospective_action_effect_20261028_v4_MAIN01.json` SHA `5a5c1948dd37ff0725447887218f6d57e59aaae54a3f1df28925fb8fe32b01a8`，MAIN 的 v4 saved receipt；它由实际同 anchor 反馈到20后固定五词、预测及执行21–25。不是原 teacher15→20 的同一 80 对，也不是增加独立根。original/alternate renderer × matching goal 是第三个观察切面，见 `results/agent_p1_renderer_matched_goal_20261028_v1_MAIN01.json` SHA `ae6dfabb047047090d14ac9f9ab00adf4547a9c09184d7070ace050b703d885b`；固定状态上改 renderer/goal 的离线比较，不是另一个闭环 rollout。

## 固定 anchor 与原生指标全表

每行恰10对；T3 是预测→实际编码的严格反向与编码缺失；T4 是实际编码→对应物理指标的严格反向，缺编码对保留 unknown。task success 在所有可比对中平局，coverage 全80 unknown。`XY` 为 agent+object 四坐标联合误差，`velocity` 为相对目标 agent 速度误差。

| anchor | root | overlay | role | 对数 | T3反向 | latent unknown | T4 native反向 | T4 XY反向 | T4 native反向且XY同向 | success平局 |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | off | Z | 10 | 0 | 0 | 2 | 0 | 2 | 10 |
| 1 | 0 | off | S | 10 | 1 | 0 | 2 | 0 | 2 | 10 |
| 2 | 0 | on | Z | 10 | 1 | 4 | 1 | 0 | 1 | 6 |
| 3 | 0 | on | S | 10 | 2 | 0 | 2 | 0 | 2 | 10 |
| 4 | 1 | off | Z | 10 | 1 | 0 | 7 | 1 | 6 | 10 |
| 5 | 1 | off | S | 10 | 3 | 0 | 7 | 1 | 7 | 10 |
| 6 | 1 | on | Z | 10 | 2 | 0 | 8 | 0 | 8 | 10 |
| 7 | 1 | on | S | 10 | 2 | 0 | 7 | 2 | 6 | 10 |
| 总计 | — | — | — | 80 | 12 | 4 | 36 | 4 | 34 | 76 |

4 unknown 恰为索引12相关对；物理返回仍在，不能删除其 physical 行或补编码。完整主面板：T3 64 concordant/12 discordant/4 unknown；T4 agent XY 74/2/4、all XY 72/4/4、object XY 41/22/13 ties/4 unknown、wrapped angle 30/33/13 ties/4 unknown、agent velocity 37/39/4；task success 76 ties/4 unknown。native/XY 列揭示目标指标不一致，不能将速度带来的不同排名解释为编码无物理信息，也不能从中声称唯一根因。

## 固定动作对全表

动作标签是同一个已封存五词库及原 float32 signed bits，不推断 clip、饱和、接触或最优性；每一种无序词对在8 anchors 各出现一次。原 library/selector source `innerdocs/agent_p1_model_library_selector_evaluation_20261028_v1.md` 指明零动作 literal 正零、负词为对应原 float32 逐位符号翻转、五步 suffix 与 prefix/full10 分别绑定；这些是**库内固定候选**而非物理动作域全覆盖。

| 词对 | 固定数 | T3反向 | T3 unknown | T4 native反向 |
|---|---:|---:|---:|---:|
| zero / uZ | 8 | 1 | 0 | 4 |
| zero / -uZ | 8 | 0 | 1 | 0 |
| zero / uS | 8 | 4 | 0 | 5 |
| zero / -uS | 8 | 0 | 0 | 0 |
| uZ / -uZ | 8 | 0 | 1 | 4 |
| uZ / uS | 8 | 4 | 0 | 6 |
| uZ / -uS | 8 | 0 | 0 | 4 |
| -uZ / uS | 8 | 0 | 1 | 4 |
| -uZ / -uS | 8 | 3 | 1 | 5 |
| uS / -uS | 8 | 0 | 0 | 4 |
| 总计 | 80 | 12 | 4 | 36 |

这些组不是独立随机样本；对称词的较多或较少反向不能证明动作限制因果、动作边界阻塞或普遍方向规律。原主面板没有在该 summary 里直接提供每个动作与 box 边界的逐坐标数值，**本审计不推断是否触边/clip**；如需该层，须在另行固定的 saved join 中核完整 raw bits、box 定义和原坐标范围，不能用词名推断。

## 一个可复算的指标断点：anchor0 的 uZ / uS

固定 pair `(i,j)=(1,3)`，root0/off/Z，i=uZ、j=uS，原 teacher15→terminal20。模型 latent margin `+0.09754843392455048`，观测编码 margin `+0.01324575870934197`：二者均偏好 j（成本较低）。物理 `i−j`：all-XY `+2.5203652350760706`、object-XY `+4.682003484120727`、wrapped angle `+0.02486884339125428`，方向也偏好 j；native distance 却 `−18.741848306115287`，偏好 i，agent velocity error `−25.519468614977114`。两臂 success 都 false（tie），coverage unknown。

两端 native distance 分别 `235.51831490210222`、`254.2601632082175`，速度误差分别 `181.75841809174503`、`207.27788670672214`。native 是七维原始距离、同时包含位置、角度和**相对 goal 的 agent 速度**，并非只评 XY；因此这里可验证“编码与XY同向但与原 native distance 相反”的具体指标失配。它是同一对的描述性算术，不证明改变速度能单独造成更好策略，不把原 native success 代为距离成功，也不覆盖其他对。

## 时间及 renderer：并列而非伪配对

后续20→25 receipt 的 8 anchor 各10对均有返回，T3 原 exact margin 严格反向按 anchor 为 `[0,0,3,0,3,5,2,4]`，共17/80、unknown0；原 teacher15→20 则是12/76可比、4 unknown。两批动作词名虽相同，但前者是原模型所选动作抵达20后的反馈 replan、后者以 teacher15 为起点；不能把17−12归因于多一步、时间或反馈，亦不能删旧 missing 令分母一致。时间变化只说明不同实际 context 下的两个封存描述值。

固定后续终态 renderer×goal 分解全80对：原匹配与 alternate 匹配 strict reversal 4；alternate 同一真实终态仅替换 goal renderer strict reversal 7；原条件到 alternate 而共用原 goal strict reversal 7。它们互相重叠，不可相加；对 XY、native、angle 的全部 counts 见源 summary。真正说明的是已保存相同物理终态的编码／目标视图可改变部分 margin；这不是 encoder 重训、物理动作改变、另一个新根或已修复模型。

## 独立确认边界与下一核验

现有 evidence 允许说固定候选、已见两根、固定时段有 T3 排序翻转及明确 T4 指标错配；不允许声称 P1 已准入、非贪心机制已成立、独立泛化、某个项是唯一根因或移除项可部署。严格分解的共同偏置与相对响应项依参考端而变，精确恒等式只是数学关系。想测试具体可干预假设，应先在新前瞻合同固定全部候选、时间、真实来源与原 success/轴，按整批保失败；例如同时记录原 raw box 距离和真实 agent-velocity 目标残差，检查上述符号冲突是否复现，而不是按本表挑有利 root、动作或新 gate。freshness 需独立完成来源与历史排除认证。

本报告以保存 JSON 的 slot/pair 分类字段为输入，分别对80对按 anchor、词对和时间计算计数，并读取原 pair `(1,3)` 与两物理标量；未打开 reserved outcome、未写结果 JSON、未触碰封存件。复核示例：`jq -r '.pairs | group_by(.anchor_index)[] | [.[0].anchor_index,length,([.[]|select(.model_to_observed.classification=="discordant")]|length),([.[]|select(.model_to_observed.classification=="unknown")]|length),([.[]|select(.observed_to_physical.native_distance.classification=="discordant")]|length),([.[]|select(.observed_to_physical.all_xy_error.classification=="discordant")]|length)]|@tsv' results/agent_p1_pair_margin_decomposition_20261028_v1_MAIN01.json`。
