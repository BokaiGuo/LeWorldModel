# PC1 真实相机、完整 token 与预测传递开发协议

本协议在本次 PointMaze 物理结果产生前固定。依据用户的 E1–E6 方案，
新增实验与旧 PushT 背景诊断分开报告，不复用旧的错误 canonical camera。

## 样本、权限与退出

- 固定 archive episodes 1832–1863，共 32 roots，不按动作结果筛选。
  1832–1847 为校准组，1848–1863 为本次拟合的留出组。
  frame 0 为初态，frame 5 为目标；不是训练外或 normalizer 外认证。
- 官方 PointMaze JEPA checkpoint，冻结参数、context=1，真实完整 proprio
  `(x,y,vx,vy)`。所有预测结果属于 `official_state_assisted`；
  视觉分支诊断也不升级为 RGB-only checkpoint 或规划收益。
- canonical camera lookat=(3,3)，shifted=(3.5,3)，azimuth=90、
  elevation=-90，其余参数继承同一 legacy renderer。不得按本次结果调相机。
- canonical 初态/目标与 archive pixels 要求逐像素一致；状态初始化误差
  <=1e-10，两个相机轨迹误差 <=1e-10，clock=0.05 s/environment step。
  失败立即留痕退出，不能静默改变样本或阈值。
- 固定 raw actions `(0,0),(.1,0),(-.1,0)`，持续 25 环境步；
  每 5 步打包，主要诊断时域5步，另报10、25步，不择优时域。
  物理指标是末端 xy 到 goal xy 的欧氏距离，无新增/替换 benchmark success 规则。

## 数据流与运行顺序

`prepare → predict → replay → analyze` 四阶段可校验来源后恢复。
prepare 只渲染初态与目标，不生成动作后果。predict 将全部预测表示、
目标表示、分数、动作和已完成调用收据写出，再允许 replay。
replay 使用同一合法动作和两相机独立重放，验证相机不改状态或物理轨迹。
analyze 才能编码真实后继。真实后继和评估标签不流入 predictor。
receipt 是 successful model-call 记录，不是完整任务证据。

## E1–E6 与反证对照

1. 每个 block 保留全部257 tokens与CLS，另记正式256 patch输出。
   检查末块经官方 layer norm 与正式输出一致，容差 atol/rtol=1e-5。
2. 同状态跨视角 normalized distance、同视角动作后果 distance、
   真实后继评分的 action ranking/regret 均按root汇总。
   CLS近似相同不能证明完整hidden state不可辨识。
3. E5 只在16个校准root的初态和目标上拟合 shifted→canonical
   仿射正交映射；不使用任何后继或物理标签拟合映射。
   全tokens使用共享384维通道旋转，CLS使用384维旋转。
   保留秩亏；同时报告恒等、仅均值平移、正交残差。
   同来源双方施加同一映射的L2评分应保持，atol=1e-7/rtol=1e-8。
4. 离线动作后果检索只用校准初态+目标特征及校准物理后果标签；
   比较平均描述和动作身份打乱（固定seed71107）。
   此项仅检测有限支持上可读出性，不作为合法部署起点或方法成绩。
5. E6 同时保留模型预测表示、真实后继表示、物理结果。
   报告官方visual+0.1proprio评分及visual分量，预测/真实编码排序、
   动作符号、真实regret、预测表示MSE与persistence基线。
   不把latent MSE与物理distance相减当理论epsilon。
6. 校准组按4个连续root folds描述层选择稳定性；所有层完整报告。
   留出组不得被用于选择新层再宣称确认。root bootstrap仅描述离散，
   同根多视角、动作、时域不算独立样本。

## 研究决策和成本边界

该审计不是新在线算法，不能通过PC2–PC5。只有同一合法预测路径的
动作后果能可靠对应真实结果，才有依据接入搜索利用；改善真实后继
评分或离线检索不自动授权替换冻结predictor。负结果首先检查来源、
形状、帧序、动作打包、时钟和官方评分等价；核查通过仍不能否定所有
表示/搜索理论。记录同步GPU阶段计时、渲染时间和拟合时间，这些是
诊断成本，不称完整闭环质量—成本前沿。
