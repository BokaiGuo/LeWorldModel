# Hi-LeWM high-only solver batch bridge v1

独占三件套同 stem `agent_hilewm_solver_batch_bridge_20261028_v1`。这是新 integration contract，不是官方未适配代码复现通过；不改官方 policy、安装 solver、Q/cost、采样、topk、mean/std、warm-start 张量或首 target 选择。无 loader、无生产 run、无 foreigncode 执行。RAM 测试不是模型实验。

## 真实来源与错误

只读取 `.venv/lib/python3.12/site-packages/stable_worldmodel/solver/cem.py`，SHA `a6662e7f1f2c6ecaf0a8527255d531aa05ed583339aabdf2798d0206382ef51b`。`solve` 的 `total_envs = len(next(iter(info_dict.values())))` 与 `v_batch = v[start_idx:end_idx]` 是两个独立问题。configure 的 `_n_envs` 不覆盖该局部 total_envs。

官方保存文本 `results/p1_hilewm_text_contract_20261027_v1/texts/package__code__h_le_wm__planning__policies.py.source.txt` SHA `d6e7d476b45e47c5015494775b4d96727b8ebb8c801f719a100ba32878f288d1`，792 行 high_info 顺序 `planner_level='high', z_init, z_goal`。因此首值长度恒为4；chunk=1 时 metadata 依次 h/i/g/h；B<4 访问空尾片，B>4 丢尾行。B=4 也不能消除 metadata 被切片。只重排键能修计数却不能修 metadata。

安装版实际计数/切片行是131/163。官方 `models/jepa.py` 保存文本的 get_cost(529行起) 还允许按 z_init/z_goal 推断 high 路由，所以不能仅凭碎片 metadata 断言每次必报错或代价一定走错。这里证实计数和字段合同损坏，而非虚构实际模型运行结论。

`source_evidence()` SHA + AST 核实真实安装文件与官方文本，不 import/compile/exec 它们；来源漂移立即拒绝。`numpy_batch_seam` 是本文件自有逐语句等价的 count/slice/NumPy-repeat 重现，不是完整 CEM 优化器，也不模拟 torch.expand 的广播/报错。因此 NumPy B<4 的空尾批诊断不声称等同 torch 实际异常。

## 接口与边界

`HighBatchBridge(original_info, tensor_batch=B, chunk_size=C, num_samples=S, native_cost=model.get_cost, abi=..., max_cost_calls=..., max_candidate_rows=..., model_parameters=model.parameters)`：

- 严格仅原三个字段、metadata 必须 high；显式 B 与两 tensor leading dimension 和完整 shape 相同且非空。保持原 tensor 对象/字节和原字段顺序的快照。
- `dict(bridge.solver_info)` 是唯一送入 CEM 的新 info：只有 z_init/z_goal，不是官方 policy 的隐式改写。两者首维均 B，所以首值计数不再依赖 metadata。调用者必须显式选择该新接线。
- 新 solver 的 model facade 是 `bridge` 本身；`get_cost(expanded_info, candidates)` 验证 B,S,H,D 和 expanded shape 后重新注入标量 `planner_level='high'`，直接调用原 model.get_cost；返回同一个 cost 对象，绝不归一化/修改公式。`parameters()` 明确透传原模型 iterator，未绑定即拒绝，避免 CEM 静默回退默认 dtype。
- 本 facade 没有 get_action，不使用宽泛 `__getattr__`。只适用已核实无 actor/get_action 的官方高层模型。若未来 loader 引入 Actionable，必须拒绝本合同并新版本处理 actor 的 metadata；不能偷偷取消 actor。安装 `solver/utils.py` SHA `d30c6225a9c4c83782132180a94c94188b07e0381d2c938c658fe5a49aed48e1` 的 prepare_init_action 分支已只读检查。
- 未来模型绑定用 `HighBatchBridge.bind_high_model(original_info, native_model, tensor_batch=..., chunk_size=..., num_samples=..., abi=..., max_cost_calls=..., max_candidate_rows=...)`，检查没有 get_action 后显式绑定 get_cost/parameters；不调用模型方法。有 get_action（即使值为None）保守拒绝。直接构造器是受信callback/RAM seam，不为任意未知模型签发兼容证书。
- 不包装/调用真实 solver，不改变 init_action；未来 MAIN 需显式新建 solver(model=bridge)，configure 原配置，再 solve(dict(bridge.solver_info), 原 init_action)。原 solver/model 实例、policy/global 均不修改。这段是 integration ABI，不是已获准的模型运行命令。
- `TensorABI(accepts, fingerprint, label)` 必须由 MAIN 提供可信纯 tensor 类型检查及 dtype/shape/精确 bytes 哈希，不得转换数据。内置 `NUMPY_ABI` 仅 RAM。尚未执行/认证 torch dtype/device/expand/实际 CEM 联调；expanded 值的 row lineage 记录但不凭 shape 认证其真实 solver 来源。真实生产仍依赖完整合法模型资产、loader、配置/预处理/坐标合同及独立实际联调。
- original_info/solver_info 两份 tensor witness、cost-expanded/candidate/cost SHA、逐 attempt/return/error 保留。前后检查拒绝 callback 改写。mapping 是快照只读视图；不冻结底层 tensor，只检测变动；非线程安全。接入层不得并发改输入，不得信任不纯 fingerprint callback。receipt 是深拷贝。
- before_dispatch 检查 cost call 与 candidate rows 两种独立 cap，失败不退费。candidate rows=B_chunk×S，不等于 predictor samples/宏步/model API：这些真实值未测即 null，不相加，不宣称 equal total compute/FLOPs。callback 异常保留 attempt，重抛不伪造完成。

## 可实际执行（repo root）

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_hilewm_solver_batch_bridge_20261028_v1.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B experiments/spectral_reachability/agent_hilewm_solver_batch_bridge_20261028_v1.py check
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B experiments/spectral_reachability/agent_hilewm_solver_batch_bridge_20261028_v1.py reproduce --batch 5 --chunk 2 --samples 3
```

CLI stdout JSON 给真实来源 SHA/行号以及旧行为、仅重排、新边界三者对照；不写 results/manifest。测试覆盖真实源码 SHA mutation、B=1/2/4/5/6、尾批、逐行不丢不重、cost 原值/对象、输入变动、失败不退费、预拒 cap 与真实路径 CLI。未执行 foreigncode、torch、GPU、神经、physics 或网络；无新科学结果，未开放 P1/freshness/科学准入。旧36tests adapter 封存不动。
