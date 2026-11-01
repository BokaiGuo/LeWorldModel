# P5 architecture_contract：本地来源合同，不依赖P1

实际结果：local_architecture_contract_verified_with_explicit_unknowns。63份当前本地文件
已SHA绑定，包括三份完整JEPA权重、encoder缓存、三份配置、原C1实际导入源码、六份
source-seal副本、许可/模型卡、normalizer与缓存来源元数据。不是空字段清单：下述事实
已实填。仍有6类人类/来源缺口，official_authenticated、architecture_contract_complete、
P5_admission、execution_authorized均false，不等待或冒充P1。

本轮仅apply_patch新增本stem code/tests/本文/JSON。无网络、下载、训练、GPU、physics、
模型导入、checkpoint反序列化、dataset读取；旧manifest、共享文档及封存结果不改。

## 完整交付路径

- /home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/p5_architecture_provenance_20261027_v1.py
- /home/gbk/code/LeWorldModel/le-wm-main/tests/test_p5_architecture_provenance_20261027_v1.py
- /home/gbk/code/LeWorldModel/le-wm-main/results/p5_architecture_provenance_20261027_v1.json
- /home/gbk/code/LeWorldModel/le-wm-main/innerdocs/p5_architecture_provenance_20261027_v1.md

以下实物路径相对 /home/gbk/code/LeWorldModel/le-wm-main；JSON保存精确路径/全文SHA/字节数。

## 已补实的身份与边界

| 字段 | 本地已核验/已填内容 | 不能越过的边界 |
| --- | --- | --- |
| 架构 | JEPA-WM AdaLN；DINOv2 ViT-S/14，224图像、16×16 patch、视觉384；proprio embedding16、block宽400、六层predictor、primitive action2、打包10 | 三种任务权重不是三种架构，不等于P5机制/任务收益复制 |
| 参数形状 | 当前权重完整SHA匹配saved restricted-meta；重核384×384投影、400×10 action encoder、1200×400 qkv、2400×400 AdaLN、16×4×1 proprio（PushT/PointMaze），Wall为16×2×1 | 本轮不fresh torch.load；形状来自保存meta报告，不冒充鲜活张量审计 |
| C1实录 | compiler与ready文件SHA接通；model_before=after=loader预期，实际strict-load/frozen state SHA保留；20个实际import模块文件逐一对prep SHA | import文件不证明所有函数分支执行；不重新认证binary/NPY/journal完整闭包 |
| C1源码封存 | 原source-seal manifest/args及6份archive，顺序/size/SHA/总86133 bytes，且与当前源文件逐字一致 | 只认证此精确封存闭包，不扩大为整个环境认证 |
| JEPA revision | 本地HEAD为13cf1d9c7e476f53c17714d2e0f1dc239a883ce0，与acquisition声明相同 | 本地Git对象，不是远端revision/签名认证 |
| Local edits | 5处tracked修改，每处HEAD blob SHA256、当前SHA、C1 source绑定/import分类；Git census含untracked（当前0） | gitignored不在Git状态census；byte身份不等于语义等价性 |
| 权重来源 | PushT完整SHA与本地facebook/jepa-wms acquisition声明一致；本地模型卡列出三任务 | 声明无不可变HF revision，PointMaze/Wall无独立task-specific获取链；不自认official |
| Encoder缓存 | 当前safetensors完整SHA匹配C1；本地HF metadata revision ed25f3a31f01632728cabb09d1542f84ab7b0056、ETag等于该SHA | metadata本身也是本地声明，不能升级独立远端认证 |
| Encoder实现 | C1实际import external/dinov2_official_7764ea0f912e53c92e82eb78a2a1631e92725fc8源码；逐文件SHA接通 | directory revision仅声明；HF缓存映射到官方实现不等于独立取得官方weight |

三份JEPA及encoder完整SHA256：

    9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb  PushT
    a01d99c4592fbedf44af076cf4c339de230c56f9f377c7559f584b97569b59bc  PointMaze
    8efb0623cfba1cb3ca210de26f7579c83dd24936635f11989c515afcb23bea1e  Wall
    ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1  cached DINOv2

## 5处local edits，不能称clean release

相对external/jepa-wms：

| 文件 | 与HEAD实际差别 | C1分类 |
| --- | --- | --- |
| app/plan_common/datasets/pusht_dset.py | decord缺失时OpenCV解码fallback | source绑定；compiler AST读取stats，不在实际模块import列表；解码等价性未认证 |
| app/plan_common/models/dino.py | 优先local TorchHub，仍有remote fallback | source绑定，非C1实际module import；本轮未调用loader |
| app/plan_common/models/prop_embedding.py | use_mlp/tokens_per_step改为getattr默认 | source绑定且C1实际import；不认证所有动态对象下语义等价 |
| app/plan_common/models/vit.py | use_sdpa改为getattr默认False | source绑定，非C1实际module import |
| app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py | wrapper补normalized-zero历史动作 | source绑定，非C1实际module import；不能据此解释direct C1 history |

当前C1直接给三帧视觉、两包between-history加新outgoing，不是wrapper缺历史后补零。
“不在import列表”不泛化为任意代码路径绝不使用。历史HF配对等价失败不外推为后来
native official实现不可运行。

还发现saved-meta报告中的inspector SHA a7ce67…与当前2821b5…不同。JSON逐项保留
saved_sha256/current_sha256/matches_saved，未篡改旧seal、未追认当前inspector生成旧报告。
原生模型/config源码仍须和saved-meta严格一致；当前只有历史分析脚本有此差别。
本轮证据范围是同权重SHA的保存meta与当前原生源码/config对应，不是旧检查器fresh replay。

## 许可不是空unknown，也不是法律放行

本地external/jepa-wms/LICENSE明确Attribution-NonCommercial 4.0 International；
external/jepa_wms_hf/README.md frontmatter/正文也声明CC-BY-NC-4.0，全文SHA与
acquisition一致。不能因此认证远端当前许可或商业授权。

THIRD-PARTY-LICENSES.md全文SHA保留；其Apache/MIT组件义务不能遗漏，部分
your-username占位链接不是精确依赖来源。pusht_dset.py、dino.py、native pusht_env.py
头部明确MIT，分别核查记录，不能把逐文件条款扁平化为一种主LICENSE。

当前C1 DINOv2源码LICENSE为Apache-2.0；encoder缓存没有已绑定模型卡/weights license，
故encoder_weight_license仍unknown。代码许可不自动覆盖另取缓存权重。human owner/legal
reviewer须确认实际用途；legal_clearance与unrestricted_commercial_permission均false。
这不是法律意见，也不是断言一切研究用途被禁止。

## RGB/proprio、normalizer和history已经补实

原C1 constructor实际with_velocity=true、relative=true、action_scale=100。native源码
以state[:2]拼state[5:]生成agent x/y/vx/vy；物体x/y/angle不在这四维proprio中。
额外低维状态相对RGB-only是特权信息差异，RGB_only=false；不能静默去掉proprio、
宣称纯RGB或给baseline不同权限。物体state作为evaluator/goal构造另记，不混为planner proprio。

context帧0/5/10，两包历史，每包5×2，time-major打包10维，再接新outgoing包。
bridge support[-1,1]连同原support_scope保留，不改称环境广告的绝对位置Box。
C1图像走224全图/RGB÷255、ImageNet通道mean/std；训练YAML augmentation不是C1
预处理，不能合并。AST只读提取以下字面量，与原C1 prepare逐值相同：

    ACTION_MEAN = [-0.0087, 0.0068]
    ACTION_STD = [0.2019, 0.2002]
    PROPRIO_MEAN = [236.6155, 264.5674, -2.93032027, 2.54307914]
    PROPRIO_STD = [101.1202, 87.0112, 74.84556075, 74.14009094]

stdlib IEEE-float32舍入算术独立重算20个保存history scalars及10个nominal scalars，
逐值一致。投影为pixel32=(100*u64).astype(float32)，再÷float32(100)，再按source
mean/std标准化；不是u/100。native零不是normalized零。centered ridge inverse沿用
保存C1收据，无动作域/阈值/β调整。

源码仅称precomputed dataset stats，不能定位不可变训练dataset/split/episode过滤及
估计器。因此数值及部署接线已知、统计来源链unknown；不读outcome重估常数。

## 预训练成本：已知配置不伪装实耗

三份当前权重匹配的保存meta均epoch=50，同时steps_seen/tokens_seen/videos_seen全0；
原样归档为无信息/占位计数，不能证明未训练或训练成本为零。

| 配置 | nodes | tasks_per_node | loader batch_size | configured num_epochs |
| --- | --- | --- | --- | --- |
| PushT | 4 | 8 | 8 | 50 |
| PointMaze | 2 | 8 | 8 | 50 |
| Wall | 2 | 8 | 8 | 50 |

三份当前YAML选取的27个scalar字段均逐项匹配saved-meta完整config，还含freeze_encoder、
transition_model optimizer、LR、history、frameskip、rollout、dtype等。只解析选定scalar，
不执行YAML构造器、标签或插值，非选定列表仅由文件SHA绑定。

配置拓扑/epoch不是实测GPU数量×时间/FLOPs；iterations_per_epoch=null，实际训练
步骤/样本/tokens/硬件/GPU-hours及encoder训练dataset/compute仍unknown。C1/本轮
training=false只证明这些阶段不更新模型，不否认发布前预训练。

## 六类剩余前置，均不是P1结果依赖

JSON unknowns都有needed_from与required_evidence：

1. remote_weight_authentication：publisher或可信独立获取源给三任务及encoder的不可变
   revision/摘要/来源链；本地matching不补远端身份，本轮不自动联网。
2. license_usage_clearance：human owner/legal reviewer核准实际用途和组件条款。
3. normalizer_statistical_lineage：publisher提供dataset revision/split/过滤/统计估计记录。
4. actual_pretraining_cost：publisher真实run logs/协议，不能按配置猜时长或用零counter抵账。
5. local_edit_equivalence：独立批准的源码/必要runtime审计；本工具不启动神经/物理。
6. observation_contract_acceptance：main/human预声明native RGB+proprio及统一baseline信息
   权限，或明确异信息协议；不静默删除proprio，也不为补表新训练head。

来源/合同工作可继续，不等待P1；完成它也不等于P5科学收益/执行授权。旧closure manifest
不修改，本contract独立交main决定下一版本接线。

## 命令、实跑、SHA和资源

cwd /home/gbk/code/LeWorldModel/le-wm-main：

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p5_architecture_provenance_20261027_v1.py
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p5_architecture_provenance_20261027_v1
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p5_architecture_provenance_20261027_v1 --verify-contract results/p5_architecture_provenance_20261027_v1.json --require-complete

producer stdout-only；JSON用apply_patch新增。第二条实跑0表示本地核查一致；第三条实跑
重建与保存JSON canonical一致，返回预期3，表示六类unknown未补齐。坏文件、漂移、磁盘
不足返回2。Git HEAD/status或任何绑定文件改变时拒绝旧contract，不悄悄找latest。

新增51项fixture测试通过；联合closure/native/baseline共170 passed in 0.41s。
覆盖AST恶意表达式、配置scalar、float32 history、错SHA/形状/proprio、Git白名单/修改
census、C1六份源码archive、非有限JSON、漂移/symlink/限额/reserve、责任方、official/
RGB-only/零成本/整数布尔篡改、无神经import。这些仅是工具逻辑测试，不是许可或效能
证据；真实producer与verify另行实跑。

实现中初次因旧inspector SHA变化fail-closed，随后显式记录历史分析源码差异但仍严格
拒绝模型/config漂移；再遇optimizer整数path，改为只抽取模型形状并补该fixture。
未更改旧产物，未放宽native模型SHA。

    fda1d8d818e4f38a83f956a65ba10117fcbdff0ccf72bd79e9f1d8c14e22b3cd  code
    2461120df946c9a94479aeea65f0bcec71600eec7eb502e526431805d243cf4a  tests
    6c169f0685bb227913ae6deb821d74093e15eed29f063584e919bae892cde7b0  actual contract JSON

本轮约1.45GB只读I/O（1,449,664,609 bytes含末尾全输入重hash）不是磁盘新增或tensor
加载。blob≤512MiB、chunk≤1MiB、小文件≤2MiB、累计读取≤2GiB，contract输出≤512KiB。
code/tests/JSON共87546 bytes，加本文远低于1MiB；pytest无cache/pyc，临时fixture很小。
启动先查reserve200MiB，不删除任何数据。保存JSON后磁盘263MiB可用，main并行活动可能
改变余量，每次启动重查。无提交、无代理。experiment-code技能只用于自包含日志/测试，
未采用与用户范围不符的训练和效果图要求。
