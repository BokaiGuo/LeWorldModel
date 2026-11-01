# B8 GN重复拒绝与驻点：保存数组诊断

MAIN新增数组审计，不运行神经、模拟或新的导数；原B8 CUDA、task、component与
journal文件仅只读。7项纯数组/实际保存轨迹测试通过。

运行：`python -B -m experiments.spectral_reachability.p1_direct_stagnation_saved_main_20261027_v1 --output NEW.json`。
当前结果为`results/p1_direct_stagnation_saved_main_20261027_v1_audit02.json`，SHA
`96f3ea09bfc6257cf9d5c10c1d1d0935fa8a43fb88d7cf2c3034170eba24c28b`。
旧同名无audit02报告保留，是添加读后hash复验/fsync前的开发snapshot，不覆盖。

每轮重算sameQ gradient，绑定前一实际incumbent、实际提案/接受与完整八步轨迹。
计算双packet独立球上的单位步长投影梯度映射`x-P(x-g)`，不是联合20维球，
也不是仅拿非零原梯度当约束驻点反例。投影是float64实代数诊断，不冒称与
producer的float32 inward-round逐位等价；没有科学零阈值或新的准入判定。

GN两根最后mapping范数0.1075043519/0.2605678132，原已拒绝提案的
`gradient·displacement`为-0.0639697607/-0.2696842826。条件于保存AD正确，
这些数不支持把重复拒绝当stationarity。负内积表示局部一阶方向，不保证有限
步长真实非线性成本下降，也不独立认证AD。完整saved-J核账不等于新有限差分。

后续NEW CUDA诊断固定原incumbent与同一rejected proposal，预定四缩步，不改
goal/Q/ball/动作语义或旧source；只有新forward结果才能验证缩步是否兑现下降。
该诊断不是重新发明阈值、逃逸资格或非贪心能力证明。
