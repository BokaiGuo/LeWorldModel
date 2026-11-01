# 探索来源同渲染重建 v3：预固定合同

仅重建已封存 exploration run01 的两个 slot；原 v2 首帧失败及删失永久保留。
固定两次 reset、十条非自适应动作、0/5/10 实际 render、遇原生 terminated/truncated
立即停止等规则继承原探索历史合同；无补根、模型、训练、witness 或结果筛选。

来源 producer 构造 PushT 时不指定 `with_target`；绑定的注册没有提供该参数，
绑定的 PushT 源码默认 `with_target=True`。v3 显式使用 True 以重建原来源渲染，
而不是修改 v2 或 moving-fresh 协议。复位后的 state、goal、首帧及 goal 图必须
与原封存 NPZ 精确相等，否则保留失败、停止该 slot；事前绑定 v2 失败证据。

原生 renderer 在 `render_goal` 且 `with_target` 时将 `goal_pose` 绘到观测帧；
`goal_pose` 是环境 variation，不应未经证据等同于 `goal_state`。无论其是否等于
任务目标，target-on 图像均违反 moving-fresh 的 `with_target=False` 观察合同。
v3 全部数据仅供 evaluator 作来源重建与工程探索，禁止冒充该协议的 planner 输入、
真实 RGB3 确认来源或 P1 gate 证据。

每个已实际观测 state 均分别记录原生 `eval_state` 原值、其原生角差表达式、
严格 `atan2(sin(delta),cos(delta))` 环绕角及其诊断判定；严格值只作诊断，不替换
原生终止。记录原生 terminated、truncated，保留停止及未观测时点，不补步。
历史覆盖、训练排除、freshness、隐藏 solver 状态、完整环境闭包均 unknown。
