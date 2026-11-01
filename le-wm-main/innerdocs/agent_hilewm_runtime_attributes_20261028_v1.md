# Hi-LeWM saved runtime属性/constructor输入 v1

独占三件套，只读固定IR与source text，使用experiment-code的固定来源/最小实现/RAM回归要求。不读取checkpoint/NPY/storage，不执行pickle、foreign imports、AST/default表达式、hooks、训练或拟合；不接管Pauli loader。worker不运行生产extract；MAIN source-first审后执行。

## 输入和来源范围

IR固定`results/agent_hilewm_inert_pickle_ir_20261028_v1_MAIN01.json`，SHA `709b31e28bee8039d577668cbd9613347be1d070a7619e3db3ca61e3674cabbe`，6,103,904bytes；复用冻结exporter metadata binding，source SHA `1e88a27e5c1cff0d4c852c2f75e7f804cdc69a23b3c956a0b0ef2163560eacce`。该绑定复核IR/decoder/scanner/官方source原闭包并从符号IR重建nn元数据，不读取生产storage。新输出不依赖导出值或模型执行结果。

官方source只沿已封存`results/p1_hilewm_text_contract_20261027_v1/contract.json`（SHA `c25edf542f38faa9cf0150df4822198d28753e2cf77a0d243e990ffabe849ded`）的明确路径/SHA读取：jepa.py、latent_action.py、eval/hierarchical.py、baseline/adapter.py、发布config.yaml。`hi_jepa.HiJEPA`与`hi_module.LatentActionEncoder`按官方兼容别名作为**源码映射候选**，不是checkpoint代码一致性证书。`module`/`_baseline_lewm_module`只静态映射本地`module.py` SHA `0b258a9e8dc24c29fcb1e8c50a09ec78b8ea85aeb79e21dd8adf712396646620`；明确local candidate，不冒充已下载官方third_party源码或不可变版本。torch/transformers实际class实现未绑定源码，保持source_class=null，不查找/导入安装包来猜版本。静态source映射含每个constructor参数、未执行default源码、自self读写位置、特殊state/hook方法及精确path/SHA/line。

## 逐字段可用数据，不是class名单

`modules`覆盖IR全部已见module路径。每module输出全部直接saved BUILD state字段（含private flags/training/eps/dropout/normalized_shape等），每字段有key/value/state节点、IR SHA、JSON pointer及opcode位置。`_modules/_parameters/_buffers`保留结构origin，并展开为`registered_fields`逐name/node引用，避免把注册成员误报为直接state缺失。source的self赋值字段与直接+registered字段作差，具体保留缺失项；源码中写过不等于构造版本一致。

`object_states`输出所遇内嵌惰性对象的完整state，包含ViTConfig所有字段、int-key id2label、tuple/list/set和float.hex，而非只列类名。`literal_value(record)`只返回我们创建的builtin literal；`object_literal_fields(record)`要求该对象**每字段**都literal，否则拒绝。GLOBAL/未知REDUCE/encoding-sensitive字节/循环/object/reference不会当函数或默认值执行，opaque始终有origin/reason；未知object BUILD布局保留opaque。已保存object state不代表执行__setstate__后的运行语义。

`derived_constructor_fields`提供具名、带精确来源的候选值：low/high预测器num_frames/input_dim由pos_embedding shape取得；depth由完整连续子层路径计数；hidden_dim/output_dim由真实norm/proj属性取得；heads/dim_head/mlp_dim/dropout逐全部层检验相同后取值，dim_head需qkv整除；emb_dropout取实际Dropout.p。latent_action input_dim/model_dim/latent_dim取真实投影层属性、max_seq_len取saved字段，num_layers与实际连续层数闭合后对所有层提取num_heads/mlp_dim/dropout。action_encoder输入宽度和macro_to_condition输入/输出宽度同样有原属性origin。缺失/层间异构/不整除输出unknown/null，不用第一层、默认值或发布config补齐。推导规则以已审source映射为条件，不能单独授权constructor。

公开helper `constructor_values(report, prefix, required_names)`返回明确请求的已知候选参数（Python builtin值），缺失/unknown/重复请求拒绝。MAIN自行列严格constructor完整必需参数，再审这些规则与所有保存子模块属性，包括activation/GELU、norm类型/eps、bias与training/freeze状态；工具不实例化模型。

action_encoder另提供smoothed_dim/emb_dim与整除才成立的mlp_scale；三个MLP提供input/hidden/output维数、实际norm_class/activation_class声明和对应子模块node。class声明只是字符串列表，不是可调用对象/动态import授权，绝不把实际BatchNorm换成源码默认LayerNorm。若其子模块布局不符，字段unknown而非套默认布局。

worker只读IR字段核查：ViTConfig保存image_size224/patch_size14/hidden_size192/12层/3heads；low/high pos_embedding为[1,3,192]/[1,4,192]；latent_action pos_embedding为[1,16,192]且max_seq_len15；input_proj.in_features10/output_proj.out_features32；action_encoder.patch_embed.in_channels10。这些不是模型运行证据。当前官方LatentActionEncoder源码给self.input_dim/self.latent_dim赋值，但实际父state只有max_seq_len；对应维度须由子层推导，不能伪称直接属性存在。运行器不得用新源码默认256覆盖保存15。

发布config的history3/action_dim2/frameskip5等逐行分栏为`released_config_scalars`，`saved_object_value=false`。只解析简单scalar leaf，不使用YAML/OmegaConf，不执行`${eval:...}`或`${...}`；插值保持raw/opaque。它们不覆盖实际低/高context3/4与动作编码输入宽度10，尤其不能擅自把10解释成唯一raw-action维数或把同一history套到high。

## normalizer / hooks的精确边界

normalizer输出所有module state字段名和IR声明中匹配明确predicate的候选及origin（架构normalized_shape也只是候选，不认证dataset normalization）；不因“没找到scaler”声称外部不存在。官方eval/hierarchical.py的build_process_map明确在324..330行新建StandardScaler、读取dataset并fit；输出AST精确call/line，**不执行fit**。外部数据normalizer的mean/scale/身份不会因此恢复，`frozen_normalizer_values_verified=false`；原训练数据与冻结预处理仍需独立来源。

每个保存hook/extra_state字段具体检查：None/空literal容器为保存状态空；非空或opaque逐module/path/field列风险，不执行其中对象。source特殊方法定义附line，未绑定torch/transformers inherited/custom methods保持unknown；没有保存hook不证明类hook/extra_state不存在，不推断class version。所有module_source_identity_verified/constructor_approved/model_equivalence/low_high_pairing/scientific_admission始终false。

## 工程与命令

属性IR decode<=100000次/深度64，source单文件<=2MiB，输出<=24MiB；只排他新建本stem `results/*.json`，NOFOLLOW/fsync，计入已有asset/decoder/scanner/IR/exporter及本stem报告，3GiB总量与10GiB余量检查，不删除失败或旧结果。输出source/IR/自身三文件前后闭包一致。verify重新从固定IR/source抽取并与指定SHA报告逐字段完全相等，无GPU/physics/网络。没有独立运行时长硬限制；约束是固定小型metadata输入及计数/深度/输出cap。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_hilewm_runtime_attributes_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_runtime_attributes_20261028_v1 plan
# MAIN审读后生产saved metadata提取（不是neural/physics）
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_runtime_attributes_20261028_v1 extract --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_runtime_attributes_20261028_v1_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_runtime_attributes_20261028_v1 verify --input /home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_runtime_attributes_20261028_v1_MAIN01.json --sha256 <MAIN实际SHA>
```

测试是合成IR/AST与冷进程metadata绑定，非模型/架构等价证据。实际生产提取是否完整、构造版本匹配、可信loader逐key加载及行为等价仍由MAIN/Pauli独立审核执行。三件套冻结后任何修订新版本，不改冻结链或其他worker文件。

worker实际回归：本任务36项RAM/metadata通过；联合冻结前序共324项通过。覆盖builtin精确保真、ViTConfig全部字段、注册成员与self属性差异、无默认值、跨层不一致/非整除/缺层拒绝、latent全部层、Embedder/MLP维数及真实norm声明、具体hook/normalizer出处、循环/重复键/cap、AST恶意default不执行、冷禁止pickle/foreign/storage读取、排他保存与saved变异拒绝。没有运行生产extract或任何模型/物理。
