# HiJEPA 受信自建 loader 最小版

独占 source/tests/innerdoc 同 stem。使用 experiment-code 的固定来源与可复算测试要求，不训练、不拟合任何normalizer/readout/Q，不执行pickle对象、foreign dynamic imports或模型loader，不重做Bohr tensor导出。worker只读取JSON/text及RAM合成NPY测试；未import torch、未实例化模型、未运行神经/physics/网络。

## 实际来源与已核事实

安全IR：`results/agent_hilewm_inert_pickle_ir_20261028_v1_MAIN01.json`，SHA `709b31e28bee8039d577668cbd9613347be1d070a7619e3db3ca61e3674cabbe`；完整531模块、491参数/buffer项，其中66显式None，425非None/425独立storage。三alias表均空，当前所有view都offset0、整块contiguous；本版明确只支持这个布局，任何alias/非连续/切片storage拒绝，绝不复制退化后称保持alias。

MAIN实际export：`results/agent_hilewm_storage_npy_export_20261028_v1_MAIN01/report.json`，SHA `3cd7dd6783129893f397ddef63bd253e447fbef93582dd5f4dc8d7144310efd3`。本版要求MAIN另给**实际verify回执路径与SHA**；接受冻结exporter `run(...verify_sha=...)` 的精确return：status=`saved_data_verified_not_model_equivalence`、report_sha256、arrays_verified=850、ir_sha256、trusted_pickle_loading_allowed=false、scientific_admission=false。尚未收到实际回执SHA时不能check-export/construct，不能把测试中手写的receipt schema当真实回执。

逐字段join：IR graph/所有491 entries、`state_dict_paths`、None、persistence、class声明与GLOBAL文字、子模块node ID、全部storage/alias定义、CRC/byteorderlittle、完整attempt-return journal、IR和exporter三文件source闭包。不打开checkpoint ZIP/data.pkl或storage NPY；只根据已认证export读取entry NPY，逐file SHA/size、dtype/shape/C-order、payload SHA、无trailing与NOFOLLOW/读前后fd identity检查。每entry传CUDA后回读原bytes hash，注册后再次核验425 keys/全部value hash；显式None注册为None，buffer与parameter分别保留，所有parameter冻结。NPY NaN payload/负零保持原位，不默默转换精度/端序。

reader采用O_NONBLOCK并要求regular file，拒绝FIFO/symlink；NPY v1头在分配前核对dtype/shape/准确payload长度/128MiB cap，防止小文件伪造巨shape触发分配。

具体架构事实，不套默认constructor：

| 项 | 实际IR |
|---|---|
| encoder | ViT CLS192，224/patch14，12层，3 heads，intermediate768，eps1e-12，sdpa |
| low predictor pos | `[1,3,192]` |
| high predictor pos | `[1,4,192]`，不是默认history3 |
| projector/low_pred_proj/high_pred_proj | 三者 `net.1` 均 BatchNorm1d(2048)，不能用本地MLP默认LayerNorm |
| macro input/output/position | `[192,10]` / `[32,192]` / `[1,16,192]` |
| macro_to_condition | Linear32→192 |

只读官方 `models/jepa.py.source.txt` SHA `bb5338bbb4f9e76b0bd15b94767624168883c994ab4c2ad8209337a3d70022a2`；`models/latent_action.py.source.txt` SHA `d06aa5955c5fdc6e61ec2f7a06317d024d8c4ad6ec6ceba7802e3d203e518e58`；本项目 `module.py` SHA `0b258a9e8dc24c29fcb1e8c50a09ec78b8ea85aeb79e21dd8adf712396646620`。不import/exec这些文件，不按IR声明import模块；声明类名仅做固定allowlist的数据标签，不构成执行授权。

## 实现到哪、没有伪装什么

`construct_cuda(..., main_authorized=True)` 是实际构造/装载实现，不是又一份manifest。延迟import **torch库**，必须可用HOST CUDA；建立本文件自有 `TrustedHiJEPADataGraph/TrustedTensorNode`，通过nn.Module注册精确子模块名和全部parameter/buffer，没有随机模型初始化、没有native类反序列化，返回内存root与construction receipt。CLI只持久化receipt，不另存模型对象/权重。CPU仅用于NPY读取与字节hash/传输，不做神经CPU fallback。

受信自建类不是原native classes，不能称native runtime state_dict等价。owned `forward` 对Linear/LayerNorm/BatchNorm/GELU/SiLU/Sequential/MLP/FeedForward/Attention/ConditionalBlock/Transformer/ARPredictor实现了本项目源码的明确eval公式，BatchNorm使用实际running stats，attention causal SDPA、dropout=0、AdaLN公式保持。root提供显式high-latent-only `predict_high/rollout_high/get_cost`，使用封存high context4、macro quantize identity、终态平方L2 sum，不改成MSE、不拟合。**这些kernel在worker未执行，尚无实际CUDA forward或原模型parity证据**。construction命令本身不调用它们，receipt forward_calls=0。

其余kind（尤其pixel ViT、latent macro TransformerEncoder、action Embedder/Conv）调用forward明确抛错。根默认forward也拒绝；没有encode/low/flat/publicRGB端到端入口，不返回随机/0代替结果。`train(True)`拒绝；`eval()`合法。本版不是full HiJEPA inference实现，不能交给共同controller就宣称baseline runnable。

## 精确阻塞及下一最小前置

1. 安装transformers的ViT已使用 `attention.q_proj/k_proj/v_proj/o_proj`、合并MLP等结构；IR实际为 `attention.attention.query/key/value`、`attention.output.dense`、`intermediate/output`。直接实例化当前ViTModel会key mismatch，不能宽松load/rewrite键后冒称匹配。IR config `transformers_version=None`，所以没有认证历史ViT forward版本/插值/attention backend闭包。需MAIN取得可核查旧实现来源或审定独占旧布局kernel，并另做固定CUDA source-fidelity；当前明确阻塞pixel编码，不猜历史版本。
2. 保存的latent_action源码构造会写input_dim/latent_dim，但IR该对象仅有max_seq_len等事实，说明对象版本与当前文字不能凭类名认证为同一bytecode。shape可从tensor读到但不等于forward认证。macro encoder kernel本版未实现，需新的source-first实现与固定CUDA核验，不执行原hi_module。
3. owned high-latent kernels待MAIN审读和独立固定latent CUDA检查；本轮没有任何神经输出、不得当官方模型等价证据。数据图加载成功只解除安全tensor到精确键/shape/dtype/值这一工程缺项。
4. embedded low与外部flat权重的语义配对、许可、raw action normalizer/history/latent bounds/public RGB身份仍独立未闭合，不从object来源或tensor CRC推断。baseline_runnable=false/scientific_admission=false永久保留本轮receipt。

## MAIN命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。第一、二条不读tensor值、不import torch；后两条必须给MAIN真实verify回执，无未来hash默认值。下列尖括号是必填实际参数，不是可直接复制的占位证据。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_hilewm_trusted_loader_20261028_v1.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B experiments/spectral_reachability/agent_hilewm_trusted_loader_20261028_v1.py audit
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B experiments/spectral_reachability/agent_hilewm_trusted_loader_20261028_v1.py check-export --export-report results/agent_hilewm_storage_npy_export_20261028_v1_MAIN01/report.json --report-sha256 3cd7dd6783129893f397ddef63bd253e447fbef93582dd5f4dc8d7144310efd3 --export-receipt <MAIN实际verify_JSON路径> --receipt-sha256 <MAIN实际verify_SHA>
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B experiments/spectral_reachability/agent_hilewm_trusted_loader_20261028_v1.py construct --export-report results/agent_hilewm_storage_npy_export_20261028_v1_MAIN01/report.json --report-sha256 3cd7dd6783129893f397ddef63bd253e447fbef93582dd5f4dc8d7144310efd3 --export-receipt <MAIN实际verify_JSON路径> --receipt-sha256 <MAIN实际verify_SHA> --main-authorized --out results/agent_hilewm_trusted_loader_20261028_v1_construction_MAIN01.json
```

构造只在明确MAIN授权后执行，worker本轮不prepare/construct、不读生产NPY。CLI新receipt排他xb，无覆盖；没有新增model序列化/模型下载。构造异常不产生成功receipt，由MAIN日志保留失败；内存partial对象随进程退出释放，不通过resetCUDA/删数据“修复”。cap：JSON32MiB/单NPY128MiB，固定425项串行读取与122MB量级CUDA权重，没有承诺硬wall/RAM峰值；MAIN自行按主机资源有界执行。后续修复必须独占新版本，不改旧seam/bridge/Bohr导出或封存结果。
