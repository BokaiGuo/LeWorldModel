# 同一探索物理来源的 target-off 视觉来源 v4

本协议事前固定两个已有探索 slot，按原 seed、recipe 和 placement 重建实际 state/goal，
再以 `with_target=False` 生成**不同于原 target-on 的视觉来源**。不得将该变体称为
原像素重现、全新或独立根；v2/v3 及其失败、结果均保持不变。本版本不因任何结果选择、
替换或补充 slot。

每 slot 先验证实际 state 和 goal 与已封存原 NPZ 精确相等；不要求历史帧或 goal 图与
target-on 原像素相等。记录本次 reset 生成的 target-off goal RGB，记录并保存真实
0/5/10 render，以及十个固定 float32 原始动作：前五 `[0.1,0]`，后五 `[0,0.1]`。
每次 step 前后持久化；native terminated/truncated 即停止，缺失帧与未完成步不补。
原生角度表达式与实际 native eval_state 和严格环绕角诊断分栏，严格值不改 native gate。

最多 4 reset、20 public steps、6 次显式 render，另有 reset 内部物理和 goal render。
不加载模型、不训练、不造 witness、不选择 outcome。固定输入、源码、协议及 v3
谱系 SHA 在执行前封存；输出含 privileged state，仅评估使用，不自动发布 blind payload。
target-off 视觉约束符合 moving-fresh 对 renderer 的单项要求，**不等于该协议或 P1
完整合规**：尚需任务/目标语义、真实 replay、来源与排除、盲输入和其他原门槛。
原来源被用于开发探索，历史和训练来源 unknown，freshness 未认证，P1 不准入。
