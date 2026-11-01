# P5：避免重复输入的下一辨识说明（仅设计，未授权生产）

## 当前证据与纠正

MAIN已报告cellwise actual分析SHA：`bc7665c448a89fee4b89bd3bc150cd2016d7772035cb0a0588bd8014b47003e0`。MAIN报告全164个branch的actual/rolled前两视觉、本体槽逐位相同；四cell各164份native完整输入与full3输出在whole-history/latest两执行之间也逐位相同。MAIN当时正在verify，本工作位未重跑analyze/verify，故不把待完成的verify写成已完成。

这排除了“两个已有执行实际改变不同history范围”的解释：whole-history虽按完整块取值，但本批完整块的差异实际仅在第15槽。latest新运行是相同输入的再执行验证，不是独立的早期history干预或第二组因果证据。两者不合并为328个独立branch或160个独立pair。旧结果、负结果、源码、名称和封存文档全部保留；旧合同中“不是单独15槽干预”等语义不能继续作为本批实际contrast的解释，本说明只作新补充，不回写旧文件。

已绑定源码允许最后槽关注前两槽，不能用相同输出推出架构屏蔽：

- 原loader `p5_jepa_native_action_response_20261027_v8.py`第124–128行配置`local_window=(3,-1,-1)`、feature proprio、0 condition tokens。
- 原`external/jepa-wms/src/models/utils/modules.py`第38–49行builder令`local_window_time=T`，只屏蔽未来；实际三槽块允许矩阵是`[[1,0,0],[1,1,0],[1,1,1]]`。
- 原H2第二次调用只是移位原actual0/5/10，留下actual5/10，追加rolled15。因此不存在可直接复用的“此调用中rolled5/10”资产。不能把同一数组换名为rolled history再运行。

这里的源码含义与现有数组身份一致；它不证明历史有效影响强、不证明排除所有数值/实现原因，也不解释旧全部预测误差。

## 唯一建议：固定早期history零特征消融，先验证非退化输入

下一待MAIN选择的具体问题限定为：在实际第15槽与原动作完全固定时，冻结预测器的terminal20输出是否依赖第5/10槽的视觉/本体特征？这是模型内部history必要性/敏感性诊断，不是重新命名现有actual-versus-rolled实验。

以下是新的前瞻假设，不是本轮生产授权；不新增runner、prepare、模型调用或结果。设计已在看过上述重复验证事实后提出，不能称独立确认或根据结果选择的最佳干预。

建议固定全164、原80pair和两root/off/on，不筛h、坐标、方向或非零cohort。基于原H1实际native输入：视觉`V`形状`(1,3,1,16,16,384)`，本体`P`形状`(1,3,256,16)`，原动作`U`形状`(1,3,10)`。定义唯一操作`Z`：float32拷贝后将axis1槽0/1全部设为正零，槽2完整字节不变；不重新encode、normalize、替换raw state或按均值/feature magnitude调值。

| cell | 视觉5/10 | 本体5/10 | 视觉/本体15 | 三个动作packet |
|---|---|---|---|---|
| AA | 原H1实际 | 原H1实际 | 原H1实际 | 原H1实际 |
| AZ | 原H1实际 | 固定正零 | 原H1实际 | 原H1实际 |
| ZA | 固定正零 | 原H1实际 | 原H1实际 | 原H1实际 |
| ZZ | 固定正零 | 固定正零 | 原H1实际 | 原H1实际 |

`Z`仅指embedding-zero，不是物理静止、缺失相机、原图黑色或归一化raw proprio=0。命名不能继续用P代表不存在的预测5/10。这组设计不测试rolled15与actual15交互；不扩大为8或16cell来追逐结果。

## 先封输入、后决定是否具有可辨识contrast

在任何新predict调用、输出/label评分前，未来prepare须对全部164份输入完成以下检查并持久封存完整native数组、字节SHA和逐槽来源；不是仅在doc里声明干预不同。

1. 核原H1报告/prepare/strict、实际模型state/source/loaded bytes和normalizer；沿原branch顺序、task/selection/condition/完整原动作与共同H1 actual20标签身份，拒绝错接。不能借当前名字猜历史输入。
2. 每branch的实际视觉5/10块与实际本体5/10块均须有至少一个数值非零float32元素；这是精确非退化检查，不是拟合阈值或按响应筛选。不能仅以负零字节不同当作数值干预。
3. 逐份证明AA/AZ/ZA/ZZ的完整输入组合四个不同；需要改变的块确实改变、其他块逐字节相同，尤其两模态第15槽与全部3个action packet必须相同。
4. 任何branch缺资产、模态数值全零或输入组合重合则整批准备拒绝，并列完整164/80和具体阻点；不删branch、不补其他动作/幅度/替代规则，不先运行可辨识子集再宣称完整。
5. AA仅在原native输入、模型、source和共同标签全部exact时复用；AZ/ZA/ZZ必须真实新执行，不复用whole/latest旧输出。完整native输入与视觉/本体full3输出必须逐份保存。全部新预测seal后才解共同真实label，不按反馈换干预。

若MAIN另行授权实施，计算规模应事前固定为3×164=492新predict、1476 token-frame samples，0新encoder/AD/physics/training；保留原AA继承成本，与新调用分列。不是本doc已执行或已授权的预算，工程wall/IO/cap还须在新runner前封存。不得以复用AA掩盖原成本或将新scope说成原实验等FLOPs。

## 如何解释，什么不能回答

固定报告全部branch/cell的full3与final20 exact输出差异及共同H1标签MSE；原80pair全保留，zero/null不补epsilon。先报告输入是否真的不同，再报告输出是否变化，不挑最有利patch、word、根、h或阈值，不训练新Q/readout。

- 不同早期输入使输出改变：说明该固定内部干预下存在计算依赖，不证明原滚动误差由早期history造成。
- 不同早期输入输出仍exact：只说明本批此消融的保存响应为零；源码允许边不等于实际利用，仍不能泛称架构忽略history或所有history无关。
- 标签MSE变好/变坏：只能是embedding-zero条件下的factual-label敏感性，不是可达物理反事实、控制收益、去除history可泛化改善或已修复模型。

零特征可能强烈离开训练支持；额外agent位置/速度权限继续披露。这不等于RGB-only、公平跨模型或多环境验证，不新增fresh roots，不解除P1/P5科学门。

若真正研究“多个滚动帧的累积误差”，必须另行获得语义准确的predicted5/10历史及生成它们的真实过去context/action/封存预测；当前H2第二调用不提供该资产。本说明不伪造它，也不把零消融包装为该问题的替代答案。

## 本次交付状态

只新增本独占说明；未创建代码、结果、prepare或任何生产调用，未改旧冻结件。未运行新测试/生产verify（纯文档，不存在新算法实现）。下一可推进项是MAIN决定是否授权上述固定内部消融；若不授权，保留“已有两执行输入相同、不能新增独立contrast”的结论即可，不应为增加实验数再运行语义重复包装。
