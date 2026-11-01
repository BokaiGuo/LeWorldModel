# HiJEPA owned dense macro loader extension v2

仅本stem三文件；v1/Bohr/旧运行依赖不改。使用experiment-code固定来源/RAM回归，用户边界优先。不执行pickle、foreign dynamic imports、训练/拟合、网络、worker神经/物理。不是官方完整baseline复现，不宣称跨模型科学通过。

## 实现和真实来源

v1的完整491项/425非None精确键、dtype、shape、storage和NPY字节验证及CUDA复制保持不变。本版调用其`construct_cuda`，无monkeypatch，不实例化当前不兼容ViT。返回`OwnedDenseMacro` facade，其`.graph`是完整冻结owned graph，`.encode_macro(chunks, action_mask=None)`新增明确dense无mask接口。

Bohr实际属性JSON：`results/agent_hilewm_runtime_attributes_20261028_v1_MAIN01.json`，SHA `5415f2a2e97e9bdeb282200e4420b9270c607190bf15a1a482963755b850b3c8`。audit绑定实际字节、Bohr三文件SHA，并使用其`extract/literal_value/constructor_values`从同一固定IR和官方text逐字段完全复算报告（含source_closure/own_sha256），不猜参数、不另解析pickle、不把提取报告当模型等价证书。MAIN单独属性verify回执仍是上游独立验收，不伪造其状态。

macro合同：grouped action宽10而非raw二维动作；model192、latent32、max_seq_len15、2层、4头/每头48、MLP768、GELU、norm_first=false、norm eps1e-5。全部used权重含bias、position/CLS及每层参数精确shape；used属性逐层literal、IR node与Bohr origin join。保存dropout0.1但本入口冻结eval，实际dropout0。dense MHA使用torch functional `multi_head_attention_forward`（无causal mask、need_weights=false），post-norm/residual/MLP按显式顺序；mask/nested路径**拒绝**，不默默丢mask。max_seq_len来自实际15，不用原text默认256。原始类/历史torch fused/nested执行与本owned dense算子的数值parity尚未认证。

原生`latent_action.py.source.txt` SHA `d06aa5955c5fdc6e61ec2f7a06317d024d8c4ad6ec6ceba7802e3d203e518e58`封存官方文字：input projection→CLS/position→TransformerEncoder→CLS output projection。v2不导入/执行该文字；不声称pickle对象bytecode与文字同版本。worker测试仅真实JSON/text+symbolic shape/call-order，无torch导入、权重值读取或CPU神经。symbolic tracer不是生产后端或数值证据。

入口限制B1..256、T1..15、float32、同CUDA设备、所有参数冻结/模块eval、有限输入/输出。输入拒绝不dispatch；真实kernel前attempt+1，异常不退款，完成同步/shape/finite后return+1。不是solver/predict预算；一次macro API不可与high predictor调用相加。construct本身宏调用0、encoder0、predict0、AD0、physics0；425次tensor copy属于数据装载，不是模型预测。

## MAIN准确命令及仍缺前置

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。只construction，未运行任何forward；排他新建receipt，不保存pickle/model对象。原export report固定SHA `3cd7dd6783129893f397ddef63bd253e447fbef93582dd5f4dc8d7144310efd3`。MAIN已报告export verify完成，但目前未向此adapter提供**持久化verify JSON路径/SHA**，因此construct仍failclosed；下列两个环境变量必须由MAIN填实际值，非未来hash默认、不能用export report代替verify receipt。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_hilewm_trusted_loader_20261028_v2.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_trusted_loader_20261028_v2 audit
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_trusted_loader_20261028_v2 construct --export-report results/agent_hilewm_storage_npy_export_20261028_v1_MAIN01/report.json --report-sha256 3cd7dd6783129893f397ddef63bd253e447fbef93582dd5f4dc8d7144310efd3 --export-receipt "${HILEWM_ACTUAL_EXPORT_VERIFY_PATH:?MAIN actual saved verify JSON required}" --receipt-sha256 "${HILEWM_ACTUAL_EXPORT_VERIFY_SHA:?MAIN actual saved verify SHA required}" --main-authorized --out results/agent_hilewm_trusted_loader_20261028_v2_construction_MAIN01.json
```

Python callable与CLI相同四个authority参数、`main_authorized=True`返回`facade,receipt`；后续MAIN source-first授权实际grouped输入后可调用`facade.encode_macro(cuda_float32_B_T_10)`。无默认输入、无伪造normalizer、无额外smoke神经实验。高层latent `.graph.predict_high/rollout_high/get_cost`沿v1，未获forward parity认证。pixel ViT、action Embedder、low路径仍未实现/拒绝；normalizer/raw grouping/public latent身份、历史ViT源码/许可、embedded-low pairing仍不闭合，因此**不是full baseline runnable**。构造成功仅精确tensor工程验收，不能宣称baseline或P1/freshness/scientific_admission放行。

资源沿v1：JSON32MiB/单NPY128MiB，425项串行复制，没有新增模型初始化、optimizer或随机参数；没有新增wall/RAM硬限制承诺。调用者保留失败日志与partial状态，不删除旧结果。未来修改新版本，本三文件freeze后不追加。
