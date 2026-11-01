# GRASP interleave 候选：v1/v2 来源与失败预算审查

只读对照，未执行CUDA/physics/fullstrict，不更改封存三件套。原候选 `agent_p2_grasp_interleave_cuda_20261029_v1.py` SHA `815ed0598b213d7fe7e86ab9330e7091f1a305bc1c4ff705a9d6b875c7f32738`。当前本审查不授权生产。

## 两版事实

直接逐文件diff：v1 HOST SHA `a7d572d9dd7b6fea8aa5b29cf76c69036fb19b8d0d704280bed7cb1581d3c596`，v2 SHA `4744fe156c547d53d75750d485473e31e027d51eac70ad83029c5b9e1f48d1f2`。`GraspBackend`、`ModelFacade`、`Tape`、`StagedLedger`、`drive_grasp` 及其直接算法语句在两个文件中无diff；它们共同引用相同封存 `box/common/grasp` 模块。这是**静态源码同文**，不是两个实际CUDA job等值证明。v2改变 `NAME/AUTH`、J `AD_return` 的事件上界和byte proof、`internal_failures` 汇总及consumer失败拒绝；尤其spectral失败边界和整体job status不同。候选从v1 import、实例化v1 `GraspBackend`/`Tape`，并不实际调用v2实现，不能把结果称v2整runner复现或证明整个v2算法/数值等价。

候选 `build` 第24–31行调用审计模块的四项SHA固定核验，并把四项和候选自身三件套加入prepare；其中v1 HOST确实有 `a7d572…` 绑定，common和原Native及installed LeWM也有其各自固定SHA。**v2源码不是审计 `SOURCES` 的一项，也不是由候选明确固定的同文对照对象**。审计模块本身 `agent_p2_grasp_interleave_audit_20261029_v1.py` SHA `d18b1b0cb362f71409665e534af21204e4a8033f637da4224168fb2c8b3d50d8`，但候选显式加入的是该模块返回的四个source以及自己的三件套；未对审计模块自身作固定SHA声明。不能用“候选引用v1的SHA”代替“v2已绑定且同文证明被完整封存”。建议独占新版本在prepare中分别固定 v1、v2、audit和比较规则源码，并以字节/AST固定必要类方法的完整闭包，或者直接以v2类执行；前者只可称静态局部同文。

## 失败与账本缺口

1. `Store.event` 第61–69行仅为事件预留1MiB；`put` 第70–78行允许普通文件消耗到CAP。`run` 第235–248行的finalizer没有隔离 report 写失败或为所有普通写预留report空间：磁盘、上限或fsync失败可能保留journal但无法生成失败report，不能保证全部失败终态有report。须先明确有界分区/实际序列化上界及 final reserve 对每次写均有效，再加真实临时盘满额/短写/fsync注入；OS极端错误仍只能如实报告不可保证。
2. 第60行的900秒仅在事件前合作检查；loader、神经内核、side封存和最终hash不能抢占，未证明硬上限。HOST 后端检查每次瞬时allocated/reserved；原后端检查峰值；候选report未记录/认证全程实际peak和精确wall scope。需要明确soft watchdog与边界、各时段及实际峰值，不把900秒说成硬截止。
3. 第243–247行在两侧成功后写死6 predict/2 backward；HOST侧保存ledger并在saved重放，但original侧无attempt/dispatch/return/capture/backward完整实账；中途失败时两个总数均为null而不是已完成的资源前缀。不能称完整失败费用审计。需要对原hook、HOST账本和both backward进行独立持久化前缀计数、失败不退款，区分实计与未知；不能以声明常数代替实测。
4. 第151–177行重建HOST的packet/账本，但score回调为None、backward只消费chart hook；第283–295行比较所存loss及梯度，未独立计算loss或证明梯度本身正确。原side仅校验事件计数/结构，未完整独立重建其source控制状态；逐packet state/action stride有保存，module IO stride未保存。应把比较与认证分开；增加来源/shape/链/算术及原侧独立重建fixture，不得把两侧相等称作独立梯度验证。
5. 即使相同的六次预测，原路径 hook + HOST facade 捕获/同步和新的局部 Layout 包装会影响GPU执行交错；比较只能说明所测固定输入/条件，不证明无捕获时或完整GRASP优化轨迹的等值。`replay_host` 使用numpy float32/CPU公式和固定插值逐位门是另外的检查，真实CUDA是否逐位符合须留待实际验证，不可在结果后松容差。

## 验证与下一步

现有测试的原Native三step hook及HOST三step/fake tape是非神经控制fixture；整体fixture原侧由事件构造，不能当真实双CUDA路径。下一独占版本应先实现明确v2绑定/静态局部对照或直接v2接线，再补完整两侧实付资源账、失败磁盘分区、设备峰值、前后闭包和完整临时盘prepare→双路径→失败留存→consumer篡改回归。保持root0固定、原source与判据，差异实报，不从效果调容差。GPU由MAIN调度，任何生产需另行授权；不把known root、相同predict数或该probe称fresh、P2机制、等FLOPs或物理收益。
