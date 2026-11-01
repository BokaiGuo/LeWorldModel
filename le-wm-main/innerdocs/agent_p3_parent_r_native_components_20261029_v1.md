# Parent R bank-independent native components v1

## 交付边界

仅公共原生接缝，不实现bank/八槽调度/eligible beam，不跑模型，不签原始R等价、P1/fullC3。旧RAM evaluator及所有旧源码保持封存不改。API设计技能用于明确注入边界、失败与版本隔离；遵守用户要求的新独占版本，不改共享canonical。

模块 `experiments.spectral_reachability.agent_p3_parent_r_native_components_20261029_v1`，无prepare/run/fullstrict CLI。`--contract` 和 `--help` 仅打印合同/帮助。

## 显式分离的入口

- `Components.ram(root,caps,verification_reserve,factory,sink)`：factory(local_task,store)必须返回kind=RAM_FIXTURE的signal/replay对象，不能注入HOST backend。fixture结果永远 `native_closure_verified=false`。
- `Components.host(root,caps,verification_reserve,model,device,sink,main_authorized=False,guard=None)`：必须MAIN显式授权、device.type=cuda和可调用guard，内部直接构建原 `NativeTree`，不接受factory。构造本身不访问模型；实际调用前先执行guard。无CPU/torch.load/权重加载路径。
- guard() 是MAIN提供的原模型/精度/来源检查闭包，需在实际调用前后执行；返回快照原样保留guard_before/after，guard异常立即failclosed。组件不以一个布尔值或空返回值自证完整模型身份；接生产runner时MAIN必须提供完整原闭包。未冻结生产bank和eligible语义时不能启动NN。本次测试仅用拒绝guard证明尚未触及model。
- sink(event)负责外部保存；组件同时保留deepcopy事件。它不是持久receipt生产器，不称已经完整保存到磁盘。事件含call_id、channel、完整payload；原native/resource payload不改编码。

root为原core.validate_task合同、identity192，保存root SHA并每次核不可变。所有调用提供完整root→当前postcontext的 `prefix_actions/prefix_path`，经过checked_path，不接受一个孤立parent terminal。该校验认证历史链/动作时序，不独自证明prefix latent是模型生成；将来生产runner须绑定前序真实native transcript/receipt。本组件不替bank或上游来源签字。

## 两个公共操作

`h1(prefix_actions=...,prefix_path=...)`：

1. 从root完整链得postcontext，调用原 `parent_task(root,postcontext)`；保留完整root/local-task/binding及prefix。不是创建新root证书。
2. 严格构造原H1 signal request，通过实际NativeTree.signal，即原NativeSignal.signal。raw10零点、norm在AD图内、192方向scalar VJP，不重写神经方程或缩放J。
3. 绑定原native attempt/return及所有嵌套ledger事件/数组；HOST调用原 `parent_native_cuda.audit_native` 验证完整IO、nominal/AD、192grad/raw/J、来源及费用。RAM只能校验fixture身份、路径、形状、费用，不冒充原生模型校验。
4. 从**同次调用**唯一phase=nominal、step0的boundary解码latent(1,3,192)/history(1,2,10)，用零raw5 checked_path且末latent逐值等于response.nominal；保留transcript索引、call_id和boundary SHA。没有补nominal replay。
5. result为RAMkernel同构的 `{nominal_path,jacobian,input_context_sha256,jacobian_chart}`。derived_resources单列AD primal predict1、J元素1920、VJP192；仅HOST完整审验才native_verified=true。不改变旧AXES，也不把派生值与旧ledger计费重复相加。

`suffix(prefix_actions=...,prefix_path=...,params=10float64)`：

1. `suffix_raw(params)`执行float64 tanh→一次float32 cast→reshape(5,2)，不normalize、不clip、不加载模型。
2. 对原postcontext local-task的NativeTree执行raw5 whole replay。mean/std在NativeTree内部恰好一次normalize，与原incoming/outgoing3/between2顺序相同。
3. 保存params_float64/raw_float32、原program_packet中的normalized token及完整path/native IO。HOST审验原replay trajectory及费用，result为 `{path}`。

suffix仅H1 lookahead，不进入执行edge；这里没有solver或资格评分，也不新增bank。调用方传params时须另绑定原ridge输入/输出，不能把任意params当已认证solver产物。

## 资源与失败合同

组件拥有一个原core.Ledger，不能拿RAM tariff再对同次native调用收费两遍。每次先检查 `attempted+expected(operation)+verification_reserve <= caps` 所有原AXES；不满足时factory/model均未派发、0新增计费、返回失败异常中的完整回执。reserve不被消费。九槽bank原子预检仍是未来orchestrator职责，不能把单primitive原子性说成整组原子性。

H1：signal_compile1；predict2及实际action_encoder/predictor/pred_proj的对应sample/token轴；J1/VJP192/grad192/raw_hook192。suffix：replay1、predict1及各对应嵌套轴。compile/metric/solve不在这两个primitive执行，必须由调用方真实入口计费。encoder/projector0；未知FLOPs/CUDA时间/backward_engine不填0。

每次结束逐轴对账expected与ledger实际attempted差额。原scalar wrapper之外的完整native transcript单独交原audit，外层wrapper不能混进旧audit census或遗漏收费。receipt保存资源事件、native事件、root/local双身份、result、原数组以及完整失败prefix。

任一callback、guard、source、返回校验失败抛 `ComponentFailure`，其 `.receipt` 保存本次所有事件/attempted资源，组件永久fail-stop，不退款、不重试。输入非法在派发前直接ValueError。失败不冒充科学负结果；completed ledger只表达原operation返回，不取代最终组件校验status。sink失败不声称持久化成功。

## 来源闭包

每次调用前后核固定NativeTree、NativeSignal、parent adapter、RAMkernel及原solver源码SHA；具体值在 `PINS` / `kernel.SOLVER_SHA256`。HOST实际installed类、weights/precision before-after属于MAIN guard/外部生产closure，不因这些本地pins就宣称已齐备。无globals monkeypatch；复用旧NativeSignal实例级observe且由其原finally恢复。

## RAM测试与CLI

从 `le-wm-main` 执行：

```sh
.venv/bin/python -m pytest -q tests/test_agent_p3_parent_r_native_components_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p3_parent_r_native_components_20261029_v1 --help
.venv/bin/python -m experiments.spectral_reachability.agent_p3_parent_r_native_components_20261029_v1 --contract
```

覆盖真实fixture postcontext变化、nominal边界恢复无额外predict、suffix完整动作历史及单次normalizer、所有资源reserve轴拒绝、错phase/重复boundary/错history/错nominal/错身份/漏J收费拒绝、失败prefix不退款、RAM/HOST不可混用、HOST拒绝guard前不触model、冷导入无torch/gymnasium。fixture不是NN/物理，不能称native生产验证。

首轮68pass/1fail：fixture把原ledger标量grad错误带shape，触发原scalar metadata拒绝；仅修fixture调用方式，不放宽旧ledger。后续69项通过；最终新组件72项+冻结RAMkernel69项联合 **141 passed in 19.24s**，CLI help独立通过，冷导入/contract由回归执行。SHA由冻结交付消息列出。未执行任何生产prepare/NN/fullstrict/physics；未冻结bank/eligible，未建新runner。
