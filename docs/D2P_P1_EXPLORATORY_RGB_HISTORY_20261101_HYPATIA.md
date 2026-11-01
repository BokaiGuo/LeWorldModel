# 探索根真实 RGB 历史：事前合同

本合同只针对已封存 exploration run01 的两个 slot，固定分母为 2；不是 freshness
认证或独立确认，也不改变任何 P1 gate。不得按模型、物理成功或可视结果挑选、补根、
重试、改动作或改时间点。history、旧运行完整性和训练数据来源仍为 unknown。

每 slot 使用原请求中选中的 random reset seed、root seed、proposal 和 attempt，
以旧 procedural source 的 `sample_state` 和原生 geometry 重构当时 placement 输入，
再以 seed 0 和该 state/goal 执行 placement reset。须严格核对实际 state、goal、
初始 RGB 和 goal RGB 与封存 NPZ 完全一致；不一致时保留失败、停止该 slot，不视为
同一物理根。state/RGB 相符仅为可观察身份绑定，不能证明隐藏 solver 状态等价。

之后固定十条 float32 原始世界动作：前五条 `[0.1,0]`，后五条 `[0,0.1]`；
在动作前及第 5、10 步后分别实际 render 并保存 RGB，与每一步 state 和 native
latest_action 共同记录。停止、截断、异常、未开始均保留，不替换或补做；无
planner、编码器、训练、witness 或未来结果筛选。

运行前先封存源码、本文、旧 request、sourcepack、journal、NPZ、旧碰撞 receipt
及环境 Python 文件的 SHA。后续动作前写持久化 journal，完成后写回执。最多四次
reset、二十次 public step、六次显式 render；native reset 和 step 还包含内部物理
与 reset goal render。明确不声称完整 native engine、训练、历史或隐藏状态闭包。
采集文件包含 privileged state，只供 evaluator；非 blind planner 输入。
