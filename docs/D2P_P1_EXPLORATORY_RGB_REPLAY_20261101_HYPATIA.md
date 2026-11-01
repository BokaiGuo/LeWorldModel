# Target-off v4 同源重放与隔离载荷合同

对既有 v4 两个 slot 固定各做一次新环境实例重放，不抽根、不按结果选择或补样。
源 seed/recipe、state/goal 精确绑定、target-off、十条原始 float32 动作、实际
0/5/10 render 和原生停止规则不变。逐步记录 state、动作 bits、native terminated /
truncated 及原生/严格环绕角诊断；逐帧和 goal 图哈希并持久化。记录失败和删失，
不因失败重试或替换。重放仅比较可观测 state、像素、动作和停止；无原 v4 每步
完整 body snapshot 或 solver cache，不声称完整物理闭包等价。

事前绑定 v4 源码、请求、所有输入、v4 receipt/journal 和这份协议。两个 slot
固定分母；两 reset/十步/三显式 render 每 slot 为最大值，原生提前停止即保留
缺失分母；不运行 encoder、模型、训练、规划或 witness。

可在重放完整且逐项匹配时，为该 slot 制作隔离的 **format-only** NPZ，仅含
`rgb_history:uint8[3,224,224,3]`、`raw_action_history:float32[10,2]`、
`goal_rgb:uint8[224,224,3]`；该文件不得部署、不得进入 planner 或冒充正式 blind
payload。所有 state、seed、witness、source 路径和诊断禁止写入 NPZ，单独 evaluator
receipt 可保存 provenance。按原 native 目标保留，不能声称 witness 终态构造的目标。

moving-fresh 还要求真实 construction witness 终态 goal、construction→replay0→
replay1、每步 body snapshot、历史排除完整、冻结新 encoder/normalizer 及盲输入
权限。此探索 replay 或格式匹配不豁免这些条件，不认证 freshness、训练排除或 P1。
原生动作是 raw world commands，模型消费须使用原固定均值/标准差正常化一次，
禁止提前混成已归一化动作或再次除 action_scale。
