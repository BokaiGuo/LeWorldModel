# Variable CEM host：已知根探索性预封存合同

独占 host 在现有真实模型、RGB、物理接口和合作 GPU 锁上接入
`codex_p1_variable_cem_20261101_exclusive.py`；该代码和合同针对已见
开发结果之后设计，不是独立确认、等算力证明或 P1/E3 admission。
当前测试与 dry-prepare 均不执行模型或物理。

边界 0/5/10/15 使用 horizon 3/2/1/1、四轮、四精英，每轮候选
16/24/48/48；保留一次 winner replay，对应每个节点 action encode、
predict 和 predict samples 预算 195/194/193/193，均不超过固定的
195 前向上限。其他节点计数为零，但并不代表完整成本为零；RGB
编码及物理按既有独立预算记录。未知的 structured 边界 15 不作
补造。共享上游成本、反向、FLOPs 和独立模块 hook 覆盖未认证，
不声称等成本。

执行器使用隔离的旧执行函数环境，注入变长 controller、
`variable.node_contract`、逐边界计数和本 host 的持久事件核验器。
每个节点完成 ledger 后、物理动作派发前，检查 schedule 合同、四
bank 和候选计数、全局 winner/重放及所有 rollout、action encode、
predict 的 attempt/return；不满足则失败停止，不执行该节点动作。
节点合同的 optimizer、配置及成本由新组件固定，controller 的旧
header 中 samples 为兼容字段，第二条 `variable_bank_contract` 的
逐边界 samples 是准确信息，消费者必须识别该覆盖关系。

仅接受本 host 同 root 的独占 prepare 和输出名称，prepare 绑定新
host、组件、测试和本文等来源。真实运行需要新 authorization、
显式 CUDA/物理标志、来源复核、既有合作锁和现场资源审核；
本次工作不构成执行授权。dry-prepare 只构建来源合同，不创建运行
数据，不证明真实模型、物理或计量覆盖已经成功。
