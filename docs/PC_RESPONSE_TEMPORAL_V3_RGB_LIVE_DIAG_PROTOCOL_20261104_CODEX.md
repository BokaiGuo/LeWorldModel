# v3 固定终点真实 RGB 开发诊断协议

预先固定使用同一 `response_temporal_fixed_terminal_dev_v3` 方法，
已暴露 PointMaze 根 0、response 与 smooth CEM 两臂，各自真实 RGB
反馈执行三次五步动作，不延长固定 15 步终点。每次规划共享同一冻结
checkpoint、normalizer、动作域、历史、目标和 1920 token 上限；
本次只检验真实接口与单根行为，不宣称 PC3/PC4 或独立性。

复用既有真实 RGB runner 和审计实现，但以独立入口显式绑定 v3
方法、新协议、v3 精确闭环封存及其审计，并写入全新排他产物。
模型预测、动作、图像、选择、弃权及模型和输入绑定全部记录；
诊断后审计记录、按既有原生后端重放实际动作。重放不等于
原 RGB 执行逐状态认证；模型预测审计不独立重新运行模型。
另根据完整时域已评分记录与实际 CEM 代数审计 response 的精修
起点，不能以 v1 的 RGB 数据代替 v3。任何失败保留，不换根或
覆盖已有数据。训练和 normalizer 来源独立性保持 unknown。
