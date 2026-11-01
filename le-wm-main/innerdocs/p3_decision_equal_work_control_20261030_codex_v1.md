# P3 同决策点工作量负控：准备与边界

## 查重与对象

既有四槽 RUN_MAIN01 是真实反馈，但 cp5/10/15 的已测数值工作不匹配；原 reverse 方案每臂完整两块 H1+H3 的末端计数相同，也不能保证各同控制步工作相同。已有 shared panel 已实现同一决策的 H1/H3 付费计算；本实现复用它的计算原语和节点成本，将其延伸到四个真实观察反馈决策点，加入在线反转策略。它不是既有 reverse 两块合同的重命名，也不能继承旧 strict receipt。

## 固定设计

两个已知根、每根按 fixed_h1、signal、reverse_signal 顺序固定六槽。控制步 0/5/10/15，每臂分别用自己的实际上下文，完整计算两 horizon 的 signal、原 scheduler、两个四候选 endpoint bank、每候选 whole 与 segmented replay、Q 选择及两条独立 winner replay；付完同一预定节点数值轴并核账后，signal 选原 horizon、reverse 选另一个，fixed 选 H1；只提交被选完整 winner 的前 5 个动作。收到该臂真实 packet 和 RGB 编码后更新上下文，再次执行完整决策。H3 未执行的后 10 步是规划候选，不是实际物理步。所有策略均必须满足相同原 continue/progress/extension gate；不满足则该槽停止，不借 fixed fallback。

每个完整决策的共同数值 caps/attempted/completed：44 predict、2 Jacobian、384 VJP directions、4 array SVD、2 solve、2 signal compile、2 program compile、18 replay、8 metric、1 schedule，其他轴见 prepare。四次完整决策每臂 176 predict、1536 VJP directions、72 replay；六槽全完成共 1056 predict、9216 VJP directions、432 replay。这些轴有嵌套，不可求和为 FLOPs；观察编码和物理调用单独处理。控制步相同、每决策数值向量相同也不证明 wall、设备 kernel、能量或总成本相同。任何部分/失败不补零、不退款、不重试；callback 返回不保证 ACK、编码或物理操作正确。

主对照为两个已知根各自第 20 步 native7 distance 的 signal−reverse；fixed_h1 为辅助，0/5/10/15/20 的 native7、joint XY、object XY、velocity 与 native success 为辅助。早停保留真实终点、计划终点缺失，不延伸终态。每臂各用自身观察和 signal，只有独立核实决策状态相同时才能说明同一状态的 horizon 是互补；之后路径分叉，不能称为单一决策的精确反事实。固定顺序有时间/cache 混杂；两个根不是 fresh 样本。本设计在旧结果之后提出，新运行须先经独立审读、外部不可变可信时间戳和授权，不能将本地 prepare 冒充已完成注册。

## 实现与验证

`p3_decision_equal_work_control_20261030_codex_v1.py` 实现 callback seam、六槽生命周期、每决策单独的防超支 ledger、ACK/RGB/上下文绑定与 signal/reverse 结构比较。比较检查每点完整计量轴、scope、策略和动作，分别报告 hook shape 多重集；它不重建候选数学与真实来源，不是生产 saved strict，也不认证未记录调用、环境、设备或权重。实际模型和物理由注入 callback 提供；backend 标签不能证明 CUDA。工厂在返回 close callback 前发生异常的内部资源，外层无法保证回收。未返回/失败的持久事件以及存储故障目前无 durable 保证。

已用 CPU-only 源闭包重建并独占保存 `results/p3_decision_equal_work_control_20261030_codex_v1_prepare_PREP01.json`，SHA256 `df480bc6d9c55daefbeac9bdb79373bf7d3eba735148cc42f0f850d05d16b860`。它绑定原 prepare SHA `b6c8dcf650d4e752f6306ed4b6f9434861c224ced51c7c8de25c0e7ca6d323f2`、源码闭包及本实现 SHA `3629fba1abd9bbbee2dc6ed273f65290d17f88588e30ed863646d3cbb4f83500`；标记 `host_runner_ready=false`、`full_saved_strict=false`、外部时间戳未验证。预检未导入 torch。已知根不开放额外 holdout。

CPU 假后端与既有 reverse 测试共 27 项通过，覆盖四次完整账、双面板、真实 callback ACK/重编码、gate、失效/早停、六槽固定分母、清理和篡改；fixture 不证明真实执行。测试期间发现并修正每决策第二个 bank 的 SVD 预算；最后 caps 与已有 shared-panel 节点账一致。

## 真实运行前缺口

尚无针对此新合同的实际 HOST CUDA/仿真 driver、durable source/decision/physical/encoder journal 和独立完整 saved consumer，旧四槽和六槽 consumer 均不能认证新 schema。还需完成其输入/根/模型/精度的实际绑定和前后核验、每阶段事件完整数学与 native IO 重建、分槽/逐点资源及物理/编码实际计数、失败及清理保存、外部硬超时、GPU/磁盘准入，以及各类最坏资源预算；再进行独立审查与可信时间戳和主线程生产协调。当前没有可安全给出的生产命令，也没有进行 GPU、神经前向、物理或训练。P1 未通过不因本工程设计改变；任何实际结果都不能据此宣称 P1/P3 机制成立。
