# P3 spectral8 parent bank v1

## 范围与事前依据

实现 MAIN 全文冻结的 `innerdocs/main_p3_r_spectral8_variant_protocol_20261029_v1.md`，SHA `3a9bddc658204ca837892cf17d69eafe205f7c3f3629c383df81bcadae38792e`。只新增本三件套，不修改旧producer、normalizer、policy、canonical或结果。接口设计技能用于明确saved-source/authority边界；新版本隔离遵守用户封存要求。

这是明确的新探索性R bank，不签原proposal等价/P1/P3/C3/freshness/physical_release。没有NN/physics/生产orchestrator，也不实现资格评估。MAIN已固定的后续行为是全体eligible进入原modelQ beam，no_witness停枝；不是把kernel的reach最小diagnostic winner当完整beam。

## 数值规则

同parent原始已付raw-zero signal，h严格1/3、nominal192、J192×10h、identity192 Q。float32原J转float64后仅计算一次 `Q @ J`，调用一次 `np.linalg.svd(...,full_matrices=False)`；J的normalizer已经在原AD图内，不再除/乘std，不做列归一化、unit scaling或sketch。

保存原始float32 J/nominal/source mapping及原生blob，保存projected_J_float64、完整left/right原始向量、sign-fixed right、全部singular values。每个right row按第一argmax(abs(row))决定符号，最大绝对坐标非负；无rank cutoff、方向择优、随机seed或结果参数。

取两个半径 `sqrt(5)*.25`、`sqrt(5)*.75`，顺序radius→mode0/1→sign(-1,+1)。八logits分别 `sign*radius*right[mode]`，float64 tanh→一次float32 cast→time-major reshape(5h,2)。bank不做raw normalization；以后NativeTree按原mean/std执行一次。zero另显式全零raw5h，不占八索引，也不借用任意proposal充当control。未计算root-DLS，不把未算的费用记作已付。

H1 fixture对原 `pusht_full_packet_compiler_20261027.spectral_bank` 的last8 params/right/kinds/singular逐位核验；raw逐位核对的是这些原logits按**本协议float64 NumPy tanh规则**产生的bits，不声称与旧compile_arm的torch float32 tanh逐位相同。H3直接以192×30的J实现新增30维路径，不调用原仅2/10维函数、不重复H1向量。总logit半径相同、不乘sqrt(3)；不宣称同单packet强度/作用/算力。

## Python接口

`build(*,root,prefix_actions,prefix_path,mapping,evidence,ledger,verification_reserve,sink,permit_ram=False)`。

- root为原core.validate_task，Q必须identity192。完整prefix raw/path经checked_path重建parent；空prefix必须原root。映射中的local_task必须逐字段等于原parent_task(root,parent)，完整root/parent双身份不能仅用末latent192代替。
- mapping精确接受已冻结 `ParentNativeAdapter.signal` 的 `.signal_map` schema：schema/root_task_sha256/root_binding/parent_task/outer_request/native_request/native_response/outward_response/mapping_sha256。重算原mapping digest，严格核对原request、原producer SHA、outer/native binding转换，不接伪造新schema或root J重标child。
- evidence精确字段：`kind,artifact_path,artifact_sha256,call_id,native_call,origin_index,nominal_index,jacobian_index,signal_compile_entry`。artifact_path须绝对，SHA显式传入；组件**不会读取该路径或签该文件bytes真实存在**。indices定位提供的native_call中原origin/nominal/J事件；保存所有原始pointer和完整call。
- origin必须nominal phase，解码原latent/history/rawzero/mean/std blob，float32形状及字节逐项一致；J/nominal response与nativeblob字节一致，不修复精度。source前向/J值原样保留，不用谱摘要代替。
- `kind=RAM_FIXTURE`只能显式permit_ram=True；其资源native_verified=false/source.resources=null。`kind=HOST_CUDA`对**传入的这一项saved signal invocation**调用已冻结 `parent_native_cuda.audit_native` 校验全部native IO/192方向/AD/J/计量，不运行模型、不重跑producer/fullstrict。不能用kind字符串单独证明GPU实际执行；生产source authenticity仍绑定外层真实artifact与runtime closure。
- signal_compile_entry是该已付外层ledger entry，要求returned、原delta、scope与index/时间；HOST内层所有resource_attempt须被该外层entry覆盖，source.resources为内层实际向量加该wrapper一次。旧已付source账与本次bank新增账分栏，绝不重复加费。

调用方ledger须为原core.Ledger、scope非空且active；caps和verification_reserve完整原AXES，不允许漏轴补0。sink接收bank事件；ledger的原sink由调用方负责持久化resource事件，本组件不改ledger.sink/global。返回结果保存本次完整ledger_events和bank events以供后续严格绑定。

## 实际bank计量与异常

新增计费为一次 `program_compile` bank入口，内部一次真实 `array_svd`。solve0、predict0、encoder/projector0；原192×10h J来源成本保留在source中，生成器不凭空再计J。完整费用向量来自实际原ledger callbacks并校验attempted/completed差额，authority_manifest绑定该ledger_events SHA及资源向量。NumPy版本与dtype保留；跨NumPy/LAPACK环境不签bitwise复现。

任何派发前先逐轴检查 `ledger.attempted+bank_cost+verification_reserve<=caps`；不足返回resource_cap_before_bank，无SVD、无新增费用、无bank/authority。只有本bank的原子预检；后续九postcontext和最终root whole/segmented verification的完整组预检仍由orchestrator执行，不能冒称在此已做。原114/562硬上限不变，suffix/H1/J原费用不退款。

输入/mapping/来源不一致派发前ValueError。实际计算或sink抛错返回partial_failure、完整账和已生成部分，不发布成功authority，不重试/退款。若sink本身持续失败则继续上抛，不能保证持久化；调用方必须保留外层失败记录。生成器没有catch后fallback到progress/zero或缩减候选的路径。

## Authority交付

result.actions为八raw edge、result.zero独立zero、params/kinds/directions保持所有八索引；authority_manifest绑定原root/local-parent、signal mapping/response/evidence、bank/zero/params、协议及所有明确source SHA、当前generator SHA、bank真实ledger和numeric环境。source所有文件每次核hash，不修改原文件。

`result.authority`符合冻结RAM evaluator的精确输入schema，可直接配同root/parent/prefix进入RAMkernel。bank_authenticated=false/authentication_sha256=null是因为本函数没有写入并认证外层生产artifact；不是否认MAIN已经选择本科学协议，更不新增approval流程。未来MAIN runner保存结果、绑定真实source bytes/receipt后可在自己的own-schema closure认证；不得将函数正确代替on-disk artifact认证。HOST source单call通过也不会自签整个tree/fullstrict/P1。

## MAIN可执行RAM命令

从 `le-wm-main`：

```sh
.venv/bin/python -m pytest -q tests/test_agent_p3_r_spectral8_bank_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p3_r_spectral8_bank_20261029_v1 --help
.venv/bin/python -m experiments.spectral_reachability.agent_p3_r_spectral8_bank_20261029_v1 --contract
```

fixture验证原last8逐位、H3完整30维/同半径、等绝对坐标sign规则、八方向顺序、J不再次缩放、原始mapping不修改、完整prefix及authority/kernel兼容、所有资源reserve轴原子拒绝、source blob/pointer/normalizer/Q/producer/paid-wrapper突变拒绝、假HOST不靠标签通过、账不退款、不计算root-DLS、冷导入无torch/gymnasium。旧last8参考函数仅在RAM测试中额外调用；其内部root-DLS不属于新build生产路径或其计量。

实际新增回归59项通过；最终联合bank59+冻结components72+冻结RAMkernel69共 **200 passed in 24.75s**，CLI help独立通过，冷导入/contract由回归执行。本轮无真实权重/NN/物理/生产数据读取或fullstrict。未完成项仅后续own-schema生产orchestrator与实际artifact认证/主机运行；本bank不代行。行数和SHA见冻结交付消息。
