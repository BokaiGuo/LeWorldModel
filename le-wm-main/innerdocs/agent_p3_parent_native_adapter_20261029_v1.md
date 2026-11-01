# ParentNativeAdapter：纯回调progress组合核心

仅新增本版本source/tests/doc；不修改封存core、shared_panel或结果。无生产runner、CLI、模型loader、CUDA/物理执行或saved fullstrict。本版本不是full-policy、P1/P3/C3准入或真实模型证书。

## 可调用接口和复用

源码：`experiments/spectral_reachability/agent_p3_parent_native_adapter_20261029_v1.py`。

```python
from experiments.spectral_reachability import agent_p3_parent_native_adapter_20261029_v1 as adapter
result = adapter.run_core(task, factory, sink, mode='adaptive', permit_ram=True)
# 生产接线不能开启permit_ram；本交付没有生产加载/执行入口。
```

`task`为原`core.validate_task`接受的192维任务，含原root_context、goal/Q、动作normalizer及完整bindings。`sink.event(dict)`接收ledger、双身份映射和生成事件；内存sink不是持久生产journal。`factory(parent_task, sink)`返回实例：

- `kind`必须是`HOST_CUDA`或显式授权的`RAM_FIXTURE`。字符串仅是接口防误用，绝非模型/设备真实性证书。
- `signal(native_request, ledger)`接受原`core.request`的finite-horizon-J接口，返回原六字段response；完整nominal192及J[192,10H]必须有限float32。nominal和AD forward分别付费，所有实际内部调用必须通过ledger。
- `replay(parent_context, raw, segmented, ledger)`返回每5个原raw动作后的完整context列表；原三latent/两token、不止terminal192。callback负责真实whole/segmented执行；本核心检查原动作归一化、移位和整个path精确相等，不证明callback内部模型。

实例不修改模块globals。复用冻结`agent_p3_runtime_contract_20261028_v1`的`policy/local_signal/choose/request/checked_path/model_cost/Ledger/audit_ledger`，复用冻结`agent_p3_fullcommit_live_20261028_v1.generate`的原bounded ridge及4/8 fractions；不复用其remaining-fit物理dispatcher。`run_core`是独占纯回调orchestrator，不冒充旧producer receipt。

## root authority / parent context双身份

外层request保留原root binding；`parent_task`深拷贝原task，仅将root_context替换为完整parent context、将对应context binding改为该context SHA。原root_task SHA/binding另行保留，不能把rebased parent说成新的root证书。

每次signal实际调用对应parent实例的H1/H3。`signal_map`保存`root_task_sha256/root_binding/parent_task/outer_request/native_request/native_response/outward_response`及整项mapping SHA。native response不改写；外层response明确用本adapter源码SHA作producer，并映射回外层binding。child查找键为完整parent SHA和H，generate必须匹配该parent两个已付费response，不接受拿root J填child。

`validate_parent`从原root逐edge调用`checked_path`，检查原rawbits、全context、input/output/action SHA和source SHA格式；每步保留3latent/2token及归一化动作链。这里不独立认证传入producer SHA所指文件，生产source/model/module闭包仍需未来runner完成。`signal_sha256`在`run_core`由真实local_signal receipt计算；直接调用generate时另强制local_response与该实例已付response相等，但不提供独立saved receipt认证。

## 原策略、原子队列、停止和拒绝

固定原choose阈值、depth3、beam4/8、max144请求；默认每次调用保留144槽，不按结果删根或删失败槽。一个调用是一项task×method，不是四任务生产panel；两known root×fixed/adaptive固定四槽将来由runner显式调四次，不签fresh。

每个parent：新H1/H3信号→原choose完整decision→整组4/8请求→整组cap预检→候选实际生成/replay→原Q、depth、indices排序。原continue=false时付完该parent信号后停止其扩展；不会强迫三层。原子性指整组身份分配和预算预检，不是事务回滚：分配后callback失败，已attempt照收，未返回候选明确failed/blocked。

原extension_type非progress即拒绝整个队列，保存原program_type/extension_type和4/8请求；不改成H1、不用旧best放行、不将R标成progress。非root原program_type=composition且extension_type=progress可以逐edge串接；这是progress组合路径，不是任意composition/R compiler。

原policy中adaptive width8与R关联。本版本generic8 bank和原子8队列有RAM完整执行测试，但原policy的R8整组拒绝，因此`adaptive_width8_executed=false`。`full_policy_executed/all_program_types_covered/full_C3_demonstrated/native_execution_certified`始终false；realized depth/width单独记录实际值。

## 完整预算和失败语义

每parent H1/H3至少8predict（4nominal+4AD）、2J、384VJP、2signal-SVD；RAM测试实际各384grad/raw callbacks。生成每parent bank一次SVD/solve，count个compile/metric、每edge H次预测。两次最终replay从真正root走全部raw，不只比较终态。

|单task×method默认cap|fixed|adaptive progress外壳|
|---|---:|---:|
|parents / edges|9 / 36|17 / 136|
|inclusive predict API及batch1 samples|114|562|
|J / VJP、grad、raw各自|18 / 3456|34 / 6528|
|signal compile / schedule|18 / 9|34 / 17|
|array SVD / solve|27 / 9|51 / 17|
|program compile / metric|36 / 36|136 / 136|
|最终replay|2|2|

predict的AE/predictor/pred_proj及sample/context-token/row轴由`predict_axes`分别展开，完整attempted/completed/events经原audit_ledger核验；不是只报predict总数。未测scalar_backward/AD_primal独立计数/J元素保持null，不以理论上界代填实际值，不签equalFLOPs。562不是R generator预算，也不是承诺一定做满。

`atomic_queue`比较整bank下限加当前完整赢家双replay reserve，**不提前收费或虚构已预留资源**。callback额外工作仍可触发真实cap，保存partial，不退款。不足整组返回`resource_cap_before_node_queue/expansion_cap_before_node_queue`，model_plan=null；R返回`refused_unsupported_original_type`，即使已有diagnostic_best也无model_plan。callback异常返回`failed_or_budget_exhausted`和真实attempt/return差。成功仅为`callback_progress_complete_NOT_certified`，保留whole/segmented原path。

## RAM验收与已知未完成

测试命令（仓库`le-wm-main`根目录）：

```bash
.venv/bin/python -m pytest -q tests/test_agent_p3_parent_native_adapter_20261029_v1.py
```

fixture是显式numpy动力学，不是神经CPU实验：真实执行动作转移、完整历史shift，再按该转移解析生成各parent J。覆盖fixed三层、adaptive根H3后child原depth1停止、child J非root J、generic8全部callbacks、R8严格拒绝、整组cap、嵌套失败不退款、root/parent/raw/older history/edge篡改及冷进程无torch/pymunk导入。最初child-stop fixture因不可控clock残差触发原R，测试失败被保留为开发事实；仅缩小合成clock尺度并同步调整其解析J，未改生产策略/门槛，此失败不是科学结果。

本次实际回归：23 passed in 7.70s；未运行NN、CUDA、physics或生产fullstrict。

未来必要但本次不实现：实际CUDA parent factory（原冻结模型全来源/精度/IO保存）、固定四槽prepare/run/strict与持久排他Store、真实child模型与whole/segmented复算、R原始proposal/评分/计量协议及实现。无R不得宣称真实adaptive8/fullpolicy。无新物理release；单棵树最长9tokens/45raw不等于20步MPC。P1、freshness和科学准入全部独立且false。
