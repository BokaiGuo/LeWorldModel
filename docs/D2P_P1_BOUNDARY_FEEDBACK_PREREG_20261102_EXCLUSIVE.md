# P1 边界候选与真实反馈：开发性机制比较草案

本合同只定义事前接口，不授权模型、环境或物理执行，也不认证来源。
selected 为预先封存的模型 selector 选择；zero 为 index 0；
greedy 固定为 `task_greedy_direct`，即 `rgb_goal_support_direct`
index 4；random 为固定 index 7。与旧 dev303 V8/V9 草案的四臂
roster 一致。latent greedy 是同候选库按预测 cp5 latent goal MSE
最小值、索引破平的诊断项，不得替代 task_greedy_direct 物理臂。
若角色同 index，保留所有角色和分支。

每角色从同一已认证 reset/warmup 重放两条分支：五步 prefix 后分别
执行（1）事前封存的 D2P 最后五步加五步零动作共同 tail；
（2）使用该分支真实 cp5 RGB 和真实五步动作历史独立规划的十步 tail。
四臂 own-tail 采用同模型、同 CEM、同初始化规则和相同每臂上限：
1 次 RGB encode、5 轮、64 样本、8 elites、5 variants、2 packets，
10 次 action encoder、10 次 predict、3200 predict samples、1800 秒。
必须另行固定完整初始化、随机种子和优化器源码；上限相同不等于实际总成本相同。
共同 tail 只比较相同动作的条件响应，own-tail 比较真实反馈后的可规划性能；
不将前者解释为后者。保存模型选择、各候选和 tail 的动作字节及 SHA。

固定检测 uint8 224×224 的原生颜色图像：推杆和物体各至少 8 个匹配
像素，推杆单连通域，每个边界至少留 8 像素；初始、目标与每个返回帧
执行检查。边界不足、缺失或分割失败时停止该分支并记录失败，不补齐帧、
换根或退回 open-loop。筛查通过不证明无遮挡或几何完整。

cp5 前成功保留性能成功，反馈和机制指标删失；cp5 后成功保留实际停止步；
未成功的 termination/truncation 为失败；身份不匹配技术无效，不换根；
动作派发后故障保留原始失败，不为主分析重跑。缺失指标不插补。八条
分支始终在清单中，单个裁切或早停不能以结果为由改变后续 roster。

分列每分支 encoder、action encoder、predict、predict samples、
候选生成、选择、warmup 和物理步数的 attempted/returned、wall 及失败。
预分支最多 warmup 10、控制 15，八分支分别最多 80 和 120；共同工作、
selector、候选与 tail 生成开销单列，不做无效填充。
报告 cp0/cp5/终点 object XY、角度、`XY+20×wrapped-angle`、原生成功、
RGB task score、模型误差与实际停止步。若 cp5 已有物体代价优势，不得将
后续成功解释为无即时进展的重配置机制。结果仅可作开发性描述。

本草案尚需验证事前来源与动作 SHA、selector 和候选 index 4 的身份、
common-tail 对 planner 来源的绑定、同前缀 cp5 跨重放一致性、信号生成
与成本计数的实际接线、完整停止/失效 runner、固定模型和环境，以及在
任何实际干预前完成独立 freeze。当前来源未认证，状态为草案而非正式
冻结，执行授权关闭。P1 gate 不变。
