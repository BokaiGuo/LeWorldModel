# 同时间 frame20：预测表征、真实表征与原生任务桥

MAIN已完整审读新 `agent_p5_same_time_frame20_20261028_v4.py`、tests/doc及复用的实际CUDA loader/encode路径，31测试通过。实际prepare、主机CUDA、完整strict和独立原数组复算均完成；旧frame15及其颜色结果不改。

## 实际产物

- prepare：`le-wm-main/results/agent_p5_same_time_frame20_20261028_v4_prepare_MAIN01.json`，SHA `88e6403b2fd828dcfe08c803cc878fc948b112b2106aa95b4d4e64acf2f1475a`。
- CUDA report：`le-wm-main/results/agent_p5_same_time_frame20_20261028_v4_cuda_MAIN01/report.json`，SHA `96c369c9b51301d902a489f0063754cce63007ca98aa5227fa54fc52ffc644e5`。
- strict：`le-wm-main/results/agent_p5_same_time_frame20_20261028_v4_consume_MAIN01.json`，SHA `542f2717e3860903d9ea87577edef2725a1fd7ae919c5aa216b45b75c0a29684`。

实际5次encoder、0predictor/AD/physics，4.14646秒，errors/finalization_errors均空，权重前后及原模型身份一致。五个完整patch数组先持久封存再评分。新预测调用为0：模型预测沿用同word原whole20已封存终态；没有按结果重跑动作。

## 固定四词、同一个第20帧

| 词 | 模型预测到goal MSE | 真实编码到goal MSE | 原生coverage |
|---|---:|---:|---:|
| xy |1.82019808|1.87302287|0.36832320|
| yx |1.90322721|1.94143139|0.34322680|
| x_minus_x |1.63789701|1.89871782|0.37083639|
| zero_zero |1.53321063|1.14708618|0.32396671|

MAIN不调用producer评分函数，直接读取五个原NPY、float64差平方均值独立复算，全四个真实goal MSE与report精确相等。6个无向pair（12有向不是12独立样本）：

- 预测→真实编码：1/6排序反转，xy vs x_minus_x。
- 真实编码→coverage：4/6反转。
- 预测→coverage：3/6反转。

两种表征都把zero_zero判为最接近goal，但它在本批coverage最低。这是一条同时间任务度量失配证据，不再由frame15/frame20混用解释；同时预测xy/x_minus_x自身仍有失配，不能只修度量便假设模型排序全部正确。

## 限制与后续

本批是1个既有接触根的固定动作词，不是四独立根，也不是P1自动谱编译的完整非贪心见证。其原native环境返回coverage/reward/done，不与moving-wrapper的agent+object成功合同混接，`task_success=null`保持。原全patch距离不能自动视为object姿态代价；也不能把4/6拿来调权重、挑patch后宣称确认性胜出。

没有认证frame20颜色干预图，四missing继续null；goal两旧颜色输入像素相同，颜色first-stage仍为null，不是encoder invariance。全部原源、旧失败及原gate不改。

下一步独占实现全空间patch贡献分解，任何候选mask只能从目标图及固定renderer规则定义，不能由outcome/coverage挑选；若不存在无可调阈值的合法mask，就只保留全256patch逐项贡献，不拟合Q。该诊断用于确定后续事前任务度量合同，不等于新compiler已验证或科学门通过。

独立新根仍缺历史scope/真实证书，四个保留确认槽不得借calibration_only消耗。当前推进不假冒freshness，原研究完整目标保持进行中。
