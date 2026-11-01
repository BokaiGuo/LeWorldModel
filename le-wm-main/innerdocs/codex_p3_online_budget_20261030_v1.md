# P3 在线信号成本和跨根总预算：实现合同

固定预算点 20、30，各点按 uniform、signal、reverse 顺序；各策略两个已知根。signal 与 reverse 各自对两个根在线做零动作 H2 nominal 和完整 Jacobian/SVD，信号权重用于排序、之后冻结跨根配额；不共享或事后读取历史信号。动态策略高权重根分别分 15/22 个候选，另一根 5/8；reverse 反转；信号完全相等则平分，不靠结果强行分开。候选流按代码固定 SHA256 tag 种子生成，不输入历史分数或物理结果。

各点各臂相同的前向四类轴上限分别 48、68：signal/reverse 的信号前向各轴 8 次，剩余搜索 40/60 次；uniform 不计算也不接收信号，直接使用 24/34 个候选消耗各轴 48/68 次。其额外 4 个候选对应节省的信号前向，不人为补算 AD。单臂各点另有 Jacobian 2、VJP 384、grad API 384、raw backward callback 384、SVD 2 的独立上限，metric 上限 24/34。signal/reverse 实际预计用完 AD 各轴，metric 20/30；uniform AD 为零，metric 24/34。每一类独立计 attempt 和 return；候选产生的四类前向调用有嵌套，AD 多类计数也有嵌套，禁止求和冒充操作量或 FLOPs。相同向量上限并不意味着同样资源消耗，不同于逐根等预算，亦非已证明的等总算力对照。

内核 `codex_p3_online_budget_20261030_v1.py` 可将 native CUDA model、两个任务、耐久事件 sink、上限回调和同步回调接入真实在线 nominal、Jacobian、SVD、搜索与 metric；独立模型初始化、锁、源和 checkpoint 身份、存储封装、完成后的严格 saved consumer 尚未接线，因而当前无生产 CLI 或授权。任何异常应保留已尝试与完成记录、不自动重试；sink 本身故障时无法保证错误事件耐久。

每一阶段完整记录信号采集与搜索分栏，CUDA 上下文、模型读取/加载、结构信号形成中的库初始化、设备能耗、真实 FLOPs、峰值内存、总 wall 和每臂 wall 如未实测必须标 unknown。不能事后利用上次结果声称预注册，已知根非独立泛化；物理还需新赢家精确动作、源 reset/goal/warmup 身份、时域/早停和资源预算绑定，以及逐槽 saved 校验后才可执行。科学 gate 不改变。
