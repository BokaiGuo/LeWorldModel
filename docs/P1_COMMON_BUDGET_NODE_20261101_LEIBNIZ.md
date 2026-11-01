# P1 事前共同节点预算：独占计费接线

这是开发性代码，不是已冻结的确认实验、真实闭环运行器或 gate 修改。
195 的前向限额由已观察的开发记录启发；旧结果不能由此变成事前等预算。
新实验必须在任何受试根的结果产生前单独确定输入、种子、模板和协议摘要。

两臂四节点使用相同的 durable-driver caps：action encoder、predict
和 predict samples 各 195；backward/VJP 各 240；Jacobian/SVD 各 1；
replay/selection 各 64；其余节点轴为 0。上限是每节点相同的调度
预留向量，不是等实际 FLOPs、时间或总成本。视觉编码、物理和共享
setup 在此节点账本之外，必须另行计费。CEM 现有实现没有通过这些
replay/selection 轴为其候选选择计费，故不能将它们作为已统一覆盖。

`make_protocol` 固定 arms、四边界、预算、共同根／模型／目标／动作
域、各 arm 各节点优化器身份以及显式来源哈希；调用方要自行在来源中
包含两个控制器、执行器及其传递依赖。本模块强制绑定自身和 meter、
driver、leaf 的来源，但**不证明其他来源闭包完整**。`bind` 在每节点
检查协议摘要、来源、身份后生成新的 ledger contract；真实上下文及
动作历史可随实际反馈变化，不能靠模板证明其因果真实性。

`open_node` 使用原生 durable Driver 和 LeafAdapter，前者在实际回调
前阻断超过合同的预留；后者对真实模型强制 frozen eval CUDA。调用方
负责保存协议及摘要、关闭并消费账本、证明所有模型调用经过该 leaf、
做 GPU 独占和真实环境身份认证。`EpisodeOrder` 是独立状态组件，
保留提前终止的实际步数，并对错误包回执 fail-stop；尚未挂到宿主。

## 未完成、不可直接运行真实实验

- structured 规划器内部重新构造并比较旧合同，尚未接入此新合同；
  同样需正确接入真实 RGB／历史、节点后验证和物理 dispatch。
- CEM 控制器／host 尚未接入此共同 driver；其 winner replay 和
  candidate selection 的计费语义与 structured 不一致，须补全并审计。
- 两臂完整来源闭包、物理逐步回执、持久化的协议摘要、模块覆盖、
  停止/失败分母和独立未见根均待实现或认证；不提供 CUDA 执行命令。

## CPU 合约测试

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_common_budget_node_20261101_leibniz.py
```

测试使用真实旧合同构造器、真实 durable driver 和非神经 analytic
fixture；验证双方四节点的同一 cap、实际调用前阻断、来源/身份漂移、
CPU 神经模块拒绝和提前停止。fixture 不代表真实性能或闭环证据。

主线程另发现：未请求的包回执原先虽会抛错，却未将状态置为停止，
此后仍可能开启节点。已改为所有无效回执均 fail-stop，并添加先失败、
修复后通过的回归测试。该组件尚未接入物理执行器，因此这不说明历史
执行发生过错误，也不构成闭环或预算公平性的实验证据。
