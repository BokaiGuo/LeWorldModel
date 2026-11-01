# 变长 CEM：开发性预算与接线审查

这是看到开发任务结果之后的新设计，**不是**先于这些结果的预注册，
不是确认性比较，也不是历史结果修订。代码不提供 GPU 或物理入口。
不得把该后定预算用于宣称旧比较等成本或独立泛化。

主线程四份报告中 structured root0/root1 的实际已封存节点 0/5/10
的 predict/action-encode 账面分别为 174/192/183；两个边界 15
均为 `not_started`，无可报告的第四个实际值。独占模块的
`audit_development_sources()` 对两份报告固定 SHA，校验节点状态、
账户及 forward hook 与账面的一致性。该核验不等于 GPU kernel
耗时、FLOPs 或整个计算成本认证。

先前独占预算预检文档中“将 174 假设为各节点上限”的方案只是
假设；根据已封存节点，它不适合描述已经执行的 structured 0/5/10
合同。该旧文件保持原样，本文件明确补充纠正。

## 新开发性配置

同一个前向节点上限 195，四轮、四精英、每节点预留一次 winner
replay，H3/H2/H1/H1：

| 边界 | 候选数/轮 | 含重放 predict 上限 | 相对上限余量 |
|---|---:|---:|---:|
| 0 | 16 | 195 | 0 |
| 5 | 24 | 194 | 1 |
| 10 | 48 | 193 | 2 |
| 15 | 48 | 193 | 2 |

`(4 × 候选数 + 1) × horizon` 只描述对应的 action encoder、
predictor 调用和预测样本轴；不得将其与 backward、VJP、Jacobian、
SVD、encoder/projector、wall time 或 FLOPs 相加。structured
仍需遵守同一前向上限的实际门控才能形成前向封顶比较；已有
boundary 15 未运行，不能事后设定其执行值或断言其会符合 195。
双臂使用同上限也不代表实际消耗相同或等算力。

独占模块提供：

1. 基于同一 CEM 实现的逐边界变长控制器。它为每个实例创建独立
   函数全局环境，不写旧模块的 `SAMPLES`；保留原滚动、打分、winner
   replay、停止与异常日志逻辑。日志首条是复用的旧固定参数头，
   第二条 `variable_bank_contract` 在任何观察/执行前明示完整
   逐边界配置并声明覆盖旧头中的 samples。消费方必须理解这一规则。
2. `node_cost` 与 `node_contract`：按同样的逐节点成本生成既有
   durable leaf-driver 格式，optimizer/config 绑定新模块 SHA、
   seed、horizon、候选数和上限；四臂槽及 `native` 标签依旧是
   兼容结构，不代表四臂运行或原生基线。
3. 纯 CPU 测试覆盖四个变长节点、已有 controller 不受实例影响、
   预算、重放、原生提前停、源报告和真实任务节点合同。测试中的
   叶子/观察/执行是 fixture，不是真实模型或环境。

## 尚未接入真实 host，禁止启动

当前并未提供新的 host runner、预封存 prepare、物理 dispatch 接线
或 GPU 运行命令。尤其旧 host 的 `node_cost`、`verify_node_events`
和 node-contract 原本依赖固定 `SAMPLES`，不能简单替换 Controller
就声称接线完成；必须另建新 host，按边界接入本模块的节点合同、
逐个核对 4×样本+1 的持久事件及真实 driver 结算，然后证明动作
来自该节点 seal，再处理提前停止和失败。还需在新控制器所用的
module scope 内提供真实模型与观察身份验证和独立计量覆盖。

若将来做确认实验，应在**未见结局的独立根/目标**上，先冻结两个
控制器、共用 cap、种子、失败规则、cost vector 和配对身份，再
经独立授权运行。当前开发根属于已见数据，不满足该条件。
