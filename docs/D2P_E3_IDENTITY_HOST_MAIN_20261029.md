# E3 identity-bound HOST 独立回归

主线程对 `agent_e3_identity_host_20261029_v1`、`agent_e3_encoder_identity_20261029_v1` 和 `agent_e3_integrated_host_saved_20261029_v1` 的联合 fixture 重新执行，23 项通过。候选将 loader 返回的同一模型对象关联到编码与数值节点，并拒绝缺少新身份记录的旧日志。

当前仅是合成 checkpoint、假模型和假设备的工程验证，没有新 CUDA 神经推理或真实物理执行。真实 encoder 双节点同环境端到端验证、实际模型身份事件容量与全链 saved strict 仍未完成；原有 `full_saved_strict=false` 和科学门不变。事件身份是源码可信前提下的运行时关联，不是对内存对象的密码学证明。

随后容量预检已接入候选 HOST，主线程独立复跑 HOST、容量和身份合同共 12 项 fixture 通过：模型加载后、创建环境前计算事件大小，超限拒绝，saved consumer 重算 journal 行长度。实际 checkpoint 的记录大小仍未测量；此 fixture 不证明实际事件能装下，也不授权真实 episode。

预检 v1 的潜在代码故障已针对性修复：编码中途失败时，完整写入的数组可能尚无 descriptor，导致旧 partial consumer 无法核验。新的 v2 运输为逐数组写入记 intent/return/failure，并在完整落盘后保留 descriptor；不完整文件或无法持久化的日志仍 fail closed。主线程独立复跑运输、预检、容量和身份四组测试，16 项通过。这是 fixture 中验证的代码路径修复，不是既有科学负结果的根因证明；真实 authority 必须加入 v2 和新运输源码绑定后才能运行 CUDA 预检。

新增 v2 authority prepare/verify 接口，扩展原 authority 的源码绑定并拒绝覆盖、冲突和篡改；主线程独立运行其两项测试通过。这两项测试使用临时构建的 authority 和已保存的上游来源，尚未生成真实运行 authority，亦未重新执行上游 full strict；不能因此启动或声称完成 CUDA 预检。

进一步审计数组账本发现其核验未独立禁止 token 重用和并发重叠写入 intent；现加顺序与身份约束，对应预检运输和 authority 回归共 12 项通过。此修改尚未进行真实 CUDA 故障注入，不能作为已有任务科学负结果的成因。

随后使用实际保存的 control5/resolver 构建 identity authority（SHA `da4c22605d02333369195778e838c44dc27454932b7094c2fb9ac2d58762619c`）和 v2 authority（SHA `f6b3836fcac6832492340a45dacc254b625af76f866a30ea8f950c6c3e30dbe5`），各由 prepare 内部重建核验。主机 CUDA 预检 `le-wm-main/results/agent_e3_encoder_cuda_preflight_20261029_v2_MAIN01` 确认 RTX 5060 Ti、CUDA 设备分配，加载冻结模型后在编码前被 capacity guard 拒绝：实际 identity bind 超过每事件 16385 字节上限。报告 SHA `386eb89ee8418c7399fa9f6e89b18c68f3131ea7e20196080dfa332f898903fd`、completion SHA `15ea151f48c08bdb60b79832db48078542a389116676e37d6468c4f1e1472441`；partial consumer 成功，数组尝试 0，physics 0。此为确实发生的工程容量问题，不是模型预测或物理失败；不放宽上限，下一步需分块并严格重建身份。

v3 将 identity 描述符分块，保留原单事件上限；主线程联合回归 21 项通过。新 authority SHA `848b6e51502eadddc24f8c8c9553b7b409eef354c6fd6b95e4b199d049c68064`；主机 CUDA 两帧预检完成，identity 实际大小 29215 字节、5 块，两个边界的 encoder/projector 均返回。完整保存消费成功，报告 SHA `06b0e3daec376d4039892ed6e8ab732b0d41480d9c6904f3fee7b19069168074`、completion SHA `1ea6de9d6ad8d56513b50776b10345fef600d03601b6b6f9056c69edda1c53f9`。此证据解决 v2 的容量故障，仅为固定合成帧编码工程验证；未重算神经 parity、未进行物理执行、不是 full saved strict 或科学门通过。

将分块 identity 接入真实 E3 host 后，主线程复跑 17 项 fixture，通过；实际 authority SHA `bc392565fadf384e476a324f145d59294cf22cf7a0b57ea3513a7a5503ae180e`。主机 CUDA 运行 `le-wm-main/results/agent_e3_chunked_identity_host_20261029_v1_MAIN01` 的 identity 分块与 loader 成功，但模型加载后、创建环境前，module census 因 `.venv/lib/python3.12/site-packages/81d243bd2c585b0f4821__mypyc.cpython-312-x86_64-linux-gnu.so` 未预先绑定而 fail closed。报告 SHA `f135b6562d5eafbb7a9ee864e67902308572120d61058733183f95ab1a386cf0`，completion SHA `80d24d4e7a2b256959f4963d6b50c01b68006ec8c3b0c152dc8ba87a3b003140`；partial inspect 成功、物理步数 0。这是工程来源闭包故障，不是任务失败；须在新运行前固定并核验实际依赖闭包，不能关闭 census。

独立 CPU 模型加载探针预封存项目内启动模块，主线程新进程 precheck 验证 6129 个模块名；保留 CUDA-only 延迟导入可能性。随后 `MAIN02` 主机 CUDA 重跑实际进入环境，两个编码边界均返回，partial consumer 核验 32 个真实观测和终止行；但结束后 census 因延迟加载 `PIL/BmpImagePlugin.py` 未预封存而拒绝。报告 SHA `f93406827a4d58c817c6892b21262cf783a72c2c9f1e21d286e2886ba73e5b7f`，completion SHA `3f99164c1347ef9f54ecfc026c5fba7614f4e29092d4340c57a523eefe63fa18`。整体仍 partial、full strict 和科学准入均 false；此错误是闭包工程故障，不可作为任务负结果。

补充渲染依赖闭包后，`MAIN03` 主机 CUDA 再次进入环境，两个编码边界返回并记录 32 个真实观测；最终 census 仍因延迟加载的 `sklearn/_loss/_loss.cpython-312-x86_64-linux-gnu.so` 未预封存而拒绝。报告 SHA `61489b51a893830585b681ad414d85f1bb1970885d1e77ec4173c089c1989517`，completion SHA `66c30d63b5885a97bb8112af4221d24bff22ee86f768300fd70fe84edbf0c5f7`；partial inspector 成功。报告仍为 `partial`，`full_native_strict=false`、`scientific_admission=false`；这不是完整反馈链验收或科学负结果。下一步应系统性预封存并核验允许范围内的安装依赖，而不是逐个事后补模块或关闭 census。

系统安装依赖闭包经主线程独立验证 31526 文件并复跑 3 项测试后，`MAIN04` 在主机 CUDA 重跑。两个编码边界返回，partial inspector 核验 32 次 render 与 30 次 public step；结束 census 因项目内 `experiments/research_runtime/__init__.py` 不在封存范围而拒绝。报告 SHA `7314667f2ef70a83711671f987bb94fdc09a6b2e99f9ae47c179ae773020d200`，completion SHA `6698a0170a5212be0f539013e43adbb3b28f552157e6f8e5eae33544dd3ba47a`。这是安装依赖范围之外的第二类源码闭包缺口，不是物理失败；仍为 partial、无科学准入。项目源码闭包需另行系统封存和独立核验，不能仅事后补一个文件或关闭 census。

协调并发源码写入暂停后，项目源码 closure MAIN06 独立验证 4018 文件，SHA `8b7366bf31139c27a413609279dfc63717bd7e970bd09b04c28c5e1808ce41b5`；authority SHA `f554fbeabfe230456c836b9afc11f2197acdbfa872711dc262b3302a5fec0933`。随后的主机 CUDA `MAIN05` 报告 `completed`，两个真实反馈编码边界 10、15 的 encoder/projector 均返回，事后独立 inspect 成功，分块身份完整且编码绑定无 pending。报告 SHA `8ff1580f2c79482ecef10ec370367e0cc8f4b9b43e96ade0cbcea54913a5bb01`，completion SHA `f12b2a20d46b7268ee8015e3848313d370bb6b095e9521ca3aef75033d990d28`。这解决本次 source census 中断，但仅是一个已知 cell 的工程完成；consumer 明确 `full_saved_strict=false`、`scientific_admission=false`，尚缺完整原生快照和失败语义、真实 loader/model/source guard 的完整证据与物理/神经 parity；不能当作整套 E3 通过或多任务泛化。

并发开发后来改变了该 authority 绑定的 P3 源码，原完整 inspect 当前会因 source drift fail-closed；上段 inspect 成功是当时的历史结果，不代表当前可重验。新增 saved-only 语义分析在不重认证模型/source guard 的条件下，独立重算 32 个 observation、37 个 snapshot 出现、20 次派发/返回及 cleanup，检查 warm10/root、cp5 和 encode10/15 的保存连接；主线程复跑 7 项测试及完整 saved 结果 verify 成功。分析 SHA `20470106ec6da1465171f6084d0124fbee4ecca210dda5ccbe4888281f06e67e`。它不能替代失效的当前 source guard 验证，也不能消除 renderer/RNG、失败路径和 parity 缺口。

基于 MAIN05 两次实际 census 中观测到的路径，另生成未来运行的精简候选 authority（SHA `ea2d7d747e02c0054eca90092ca947e136cdba10aca3a204dfa85d8b01ec64e9`）；它仍继承安装依赖 authority 并对未知加载模块保持原 census 拒绝，不是完整导入闭包，也不能替换 MAIN05 的旧 authority。主线程独立验证和 6 项测试通过后，主机 CUDA `MAIN06` 完成，报告 SHA `3ee72bd9b3ade29020083359f878b67accf4ec531dbccefa2acb3d94f1065bea`，completion SHA `9adfc093f3b6293c92e56ec245daeb431d0a0c59548805dcd3d67560f745806f`。原 inspect 成功，核验 32 个 observation、两个 encoder/projector 边界、20 次控制和完整 terminal row；运行结束后的候选 closure 重新 verify 通过。`full_saved_strict=false`、`scientific_admission=false` 仍不变；该已知 cell 重跑不是独立新根、物理 parity 或 E3 科学通过。

MAIN06 另由 native saved 语义审计独占产出 MAIN02 并独立 verify，SHA `738aeb2995df07ae53e95b9ed04fd0a8eda75735ca726ed1666fcf63337cbd39`。该审计补强快照字段及事件连接的记录级检查，仍不声称完整失败语义或真实物理/神经 parity。
