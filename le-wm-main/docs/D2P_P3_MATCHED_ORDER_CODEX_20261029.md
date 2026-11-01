# P3 模型侧等计数工作块：固定顺序与信号顺序

## 资产核查与新问题

已有 `agent_p3_shared_panel_20261029_v1` 是真正的双面板 paid-pipeline 对照并已有真实运行和 saved consumer：每个实际节点两个方法都生成 H1/H3 全面板，然后选择首包。它检验相同已付信息下的面板使用，**不检验稀缺预算下只生成一个面板的先后分配**。K17 的 equal-cap 原树与六槽账本分别表明相同上限不等于相同实际开销。本设计不重复旧双面板，也不改旧资产或从物理效果选择预算。

新代码 `experiments/spectral_reachability/p3_matched_order_codex_20261029.py` 是模型侧可注入 native backend 的两块原型，不是完整 HOST launcher。固定根下比较两方法：

1. `fixed_order` 固定先 H1 后 H3；
2. `signal_order` 首个 parent 用既有 `core.choose` 的 adaptive 原决定选择 H1 或 H3；只接受 continue/progress；第二块**强制使用剩余 horizon 配额**。不支持原决定时拒绝，不替换为 progress。

每块重新计算该 parent 的 H1/H3 信号，只为选择的 horizon 生成一个 endpoints、4 候选的完整工作块；每候选 whole/segmented，选原 model-Q 最小及稳定 index 后再独立 replay winner。第二块接第一块模型预测的完整 context，绝非实时观测或物理反馈。两臂在两个 parent 都付原 signal 和 schedule，fixed 的 schedule 以及第二节点 adaptive 决策是诊断成本、并不决定顺序；它们不冒充最省成本自然基线。每条路径预测 5+15=20 个动作，但 H 顺序与上下文可变；不同时间的物理表现不在此实验内。

这是一个**新约束策略**，只检验有限的 horizon 顺序分配，不是原 K17 adaptive 完整 width/depth/type policy；没有 R 实现，没有足够的物理任务、fresh roots 或公平性证据。不保证每个 root 上原 signal 决定发生路由差异；退化为 H1→H3 也须如实保存。

## 精确的已测轴定义

完整无失败一臂：两个节点的 signal 各 8 predict；4 候选两次回放加一次 winner 回放为每个 H 付 `9H` predict，总计 `2×8+9×(1+3)=52`。其余 exact 向量包含 4 Jacobian API、768 VJP directions/grad API/raw backward hooks、6 array SVD、2 solve、4 signal compile、2 program compile、18 replay、8 metric、2 schedule；predict samples 52 和 context tokens 156，其他所有 27 轴由冻结函数返回。配对还比较 hook 的 resource/shape/directions/status 多重集；**轴匹配和 shape 多重集匹配分开报告**。发生故障或不支持时不补调用，保留失败 attempt，匹配状态为 unknown。

两块各有真实目标驱动的候选及选中路径；非选中的完整候选和 winner 独立回放是预定验证，不是末尾随机 padding。尽管逐轴相等，不能推断 kernel FLOPs、时间、显存、能耗或未 hook 的工作相等；不能把重叠轴加和。此前物理接口数量也不能填补这些缺口。

## 运行与验证接口

`run_unit(root, policy, factory, store, permit_ram=False)` 复用现有 parent task 绑定、native backend `signal/replay`、原 signal 和 model-Q/候选生成、Ledger 与原 resource hook；指定 backend `HOST_CUDA` 才能走非 fixture 接缝。backend 的 `kind` 字符串不是来源认证。run 在错误时将部分账本和失败事件交给调用方 store，不做重试；store 的持久性及容量保证由真正 HOST 集成方负责。

`verify_unit(root, saved_unit)` 重算原决策、每个 bank/候选 model-Q、排序、context 串接和账本；`compare` 检查同根、27 轴完整账与 shape 多重集。命令行 consumer 只读取三个外部 SHA 固定的 JSON，独占写入结果：

```sh
.venv/bin/python -m experiments.spectral_reachability.p3_matched_order_codex_20261029 \
  --root ROOT_JSON --root-sha ROOT_SHA256 \
  --fixed FIXED_UNIT_JSON --fixed-sha FIXED_SHA256 \
  --signal SIGNAL_UNIT_JSON --signal-sha SIGNAL_SHA256 --output NEW_RECEIPT_JSON
```

此 consumer 独立验证保存算术和事件元数据，**不重新验证真实 native IO/tensor transcript 或 GPU 来源**；这两个字段明确 false，fixture 的 consumer receipt 也绝非科学结果。`run_unit` 与 consumer 不提供自动 HOST 模型装载、源闭包认证、跨臂事务/失败可恢复的持久存储、精度/设备真实性检查、whole-job 资源门或独立原生 IO 严格核查，因此现在不能宣称新方案已可生产，也不提供直接启动 CUDA 的命令。必须先另立并冻结任务/来源、硬件与存储限额、执行顺序、持久装置和完整独立 native IO consumer，由主线程审读协调后才可主机 CUDA 实跑；不以 `kind` 标签或本 fixture 合同绕过此门。

## 测试及当前结果

```sh
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p3_matched_order_codex_20261029.py
```

RAM fixture 通过现有接口执行两种策略、比较实计向量和 shape 多重集、重建 parent/bank/decision/winner；还检查错误 attempt 保留、失败不可匹配、不支持原决策、默认拒绝 RAM、篡改以及带外 SHA/独占 consumer。fixture 没有模型/真实环境，也不代表科学结论。本次不运行神经、GPU、训练或物理，不修改封存数据。
