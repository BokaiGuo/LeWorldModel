# P1：同动作、同时间误差归因

本轮优先研究主线，保留全部40槽、8个上下文内80动作对；只有两个已见根，不能当作独立确认。每一臂的模型预测、真实编码和物理标签绑定同一原始动作与第20步终态；动作对两臂的后缀本来不同。

## 实际完成

MAIN完整审读 `agent_p1_pair_margin_decomposition_20261028_v1.py`、测试与说明，联合自有原生距离轴分解共53测试通过。最初两个失败来自合成fixture共享列表别名，已隔离；没有降低validator或科学门槛。

正式saved分析已完成，重建上游完整原生source/array/journal闭包后重新读原float32数组：
`le-wm-main/results/agent_p1_pair_margin_decomposition_20261028_v1_MAIN01.json`，
SHA `f723c5eb73ecebd0615a2ded0aa4b741c6f132ca5f6063c673f12d5e98539143`。
第二次独立完整verify现已exit0，`verified=true`，同80对全量重建且P1_admission=false；不是hash-only核验。

| 检验 | 本批结果 |
|---|---|
| 模型预测→真实编码 | 76可比对中12反转，另4未知全部保留 |
| 真实编码→四维agent+object XY距离 | 76对中4反转 |
| 真实编码→object XY距离 | 22反转，13物理平局 |
| 真实编码→object角度误差 | 33反转，13物理平局 |
| 真实编码→原生state distance | 36反转 |
| 真实编码→原生成功 | 76个成功标签平局；不是76次成功 |
| 精确有理数恒等式 | 每槽/每对误差展开残差为0 |
| 原float64排序与精确算术符号 | 无不一致 |

因此本批12个预测排序反转不能解释为浮点近零符号翻转。它们的预测margin约0.0105–0.0601，误差交叉项与二次项足以跨过margin；不能只看均方预测误差，或单凭交叉项大小认定唯一原因。

## 原生distance的速度项：代码/评价协议差异，而非偷偷改指标

MAIN新增独立工具 `main_p1_native_distance_axes_20261028_v1.py` 与6项测试，直接读已绑定evaluation，按原float64 state值做精确有理数分解：

`distance² = agent+object XY平方和 + 原始角度差² + agent速度差平方和`。

实际结果 `le-wm-main/results/main_p1_native_distance_axes_20261028_v1_MAIN01.json`，
SHA `25aa6a0d1ce0ff574f298d3bb60dfdc2bd2f9bb445306f313d07a9043ac344f2`。
该窄工具只核固定原evaluation及wrapper SHA，不独自重建原完整闭包；完整闭包由上面正式分析重建。

原生wrapper `external/jepa-wms/evals/simu_env_planning/envs/pusht_gym_wrap.py` 的成功判据排除速度，但state distance包含速度。任务goal state含非零agent速度，单张目标RGB不能单独确定该速度。
76可比对的36个latent→distance反转中，34个是latent与去速度的原始pose平方和同向、速度项改变了总distance排序；全80对中37对pose/distance符号因速度项而异。所有80对原distance浮点排序均与精确算术一致。

这是对原指标的**代数归因**，不是删除速度项后的新任务/新gate，也不是证明视觉任务已解决。去速度pose仍混合像素位置与弧度单位，数值上偏重位置；四维XY也可能被agent而非object主导。因此不能以“只剩4反转”掩盖object XY的22反转和角度的33反转，不能由此宣布latent→完整任务可靠。

MAIN另核本批40条原生成功全部false、native angle负值项为0。速度项能解释上述distance排序现象，**不能解释或消除原生成功门未通过**，也不是全部P1断点的唯一根因。

## 下一研究动作

1. 保留12个预测反转，逐对查看共同误差投影与相对动作响应误差，不把代数闭合等同因果归因。
2. 4个latent→四维XY反转对为 `(22,24),(25,27),(36,38),(37,39)`，仅作为完整固定面板中的定位项，不另挑成确认集；object/angle全量失配仍须检查。
3. 后续表示/任务桥必须区分可见位置、object角度、目标速度与原生成功，不拟合新Q、不用原物理标签给selector重排、不改旧门。已有第20帧固定四词同时间RGB/任务探针准备继续并行。
4. 这些是两个已见根上的回顾归因；P1非贪心能力机制、独立新根和强对照仍未成立，不能以本轮结果替代。

神经实验未降级CPU。本轮只是原生保存数组/事件的算术分析，无新增神经或物理调用。
