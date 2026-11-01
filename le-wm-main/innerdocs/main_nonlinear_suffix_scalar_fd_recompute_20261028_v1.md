# MAIN标量有限差分独立复算

不导入模型，不做新神经或物理。先对实际producer完整consume并与外部SHA
固定的完整receipt比较，再独立解码每个native float32 loss的base64字节、
shape/dtype/SHA。固定4点×3步长×10坐标×正负，差分用原GPU loss做float64
相减再除2h，不能用NumPy重算Q冒充GPU reduction结果。

独立计算全部120对和12个完整梯度向量指标，与strict逐项比较。缺一对即
保留整个向量汇总为null，无幸存子集指标、best-h、通过阈值、AD正确证书
或P1准入。不是原GPU reduction的重新执行，也不是物理响应正确的证据。
源码固定producer SHA，全部上游由实际consumer重建；输入结果前后SHA核验。

单元测试9项仅原字节/算术/缺项行为，不是真实模型数据。实际CLI参数为
--producer-output、--report-sha、--receipt、--receipt-sha、--output；必须填
真实完成产物的SHA。输出独占新文件，不覆盖或拟合任何旧结果。
