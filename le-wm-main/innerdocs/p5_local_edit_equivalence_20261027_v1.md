# P5 本地五处修改的条件等价合同（2026-10-27 v1）

结论：已完成可复算的源码/符号数组合同，不是官方认证、三任务端到端等价性或P5科学准入。没有执行目标模块、反序列化checkpoint、模型forward、CPU neural、CUDA、物理或网络。所有目标源与旧结果保持原样。

按 experiment-code skill 采用自包含JSON、实际输入绑定和测试记录；其训练/作图通用流程不适用于本任务，服从用户只读静态约束。

## 新产物与命令

根目录：/home/gbk/code/LeWorldModel/le-wm-main

- 代码：/home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/p5_local_edit_equivalence_20261027_v1.py
- 测试：/home/gbk/code/LeWorldModel/le-wm-main/tests/test_p5_local_edit_equivalence_20261027_v1.py
- 实际报告：/home/gbk/code/LeWorldModel/le-wm-main/results/p5_local_edit_equivalence_20261027_v1.json
- 本文：/home/gbk/code/LeWorldModel/le-wm-main/innerdocs/p5_local_edit_equivalence_20261027_v1.md

从上述根目录运行（不写输出文件，不产生pycache/pytest cache）：

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p5_local_edit_equivalence_20261027_v1.py
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/spectral_reachability/p5_local_edit_equivalence_20261027_v1.py
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/spectral_reachability/p5_local_edit_equivalence_20261027_v1.py --verify results/p5_local_edit_equivalence_20261027_v1.json
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/spectral_reachability/p5_local_edit_equivalence_20261027_v1.py --verify results/p5_local_edit_equivalence_20261027_v1.json --require-full-equivalence

前三条预期0，最后一条预期3；输入/SHA/预算/合同不一致返回2。verify真正重新读取源码、Git HEAD、配置、保存元数据与凭据，重算AST和数组合同，并重新流式散列三份权重，不只是检查JSON内的pass。输出仅stdout。生产器固定200MiB空闲门槛；输入小文件2MiB、权重单文件512MiB、累计读2GiB上限；权重约606MiB，稳定性复核再读一次，不复制到磁盘。新文件总量远低于2MiB。

本次测试50 passed；实际报告32个SHA绑定。报告不是mock；测试中的符号案例、变异源码和伪造receipt是逻辑测试，不是科学证据。

SHA256：

    code   45cc4131a99cb57227590fc10db9e0fdf8786f1307e94d5c1cc828eaa73edeb7
    tests  1804b130dcb44c9604a0e51f12732a3d19e2fad5d36104ffcd950b9007d49da8
    report d540d529cf6af8e653952fbf93d108c42c0205a112baa011d5ec09f96b317653

## 三checkpoint：本次新增的可验证部分

基于现有architecture contract（SHA 6c169f0685bb227913ae6deb821d74093e15eed29f063584e919bae892cde7b0）限定五处编辑及Git本地HEAD 13cf1d9c7e476f53c17714d2e0f1dc239a883ce0。HEAD仅为本地Git对象，不认证远端来源。

本次独立重建完整state schema，不再只核验5个代表shape。六层AdaLN，视觉宽384，proprio feature16，总宽400；action encoder 400×10；qkv 1200×400，MLP 1600×400/400×1600，AdaLN 2400×400。predictor共92键、17,626,480参数。proprio patch_embed共2键：PushT/PointMaze输入4维、80参数，Wall输入2维、48参数。逐键检查名称、shape、float32；Wall套PushT proprio schema必须拒绝，而不是把跨任务shape差异称为checkpoint损坏。

额外以AST核对已保存strict CUDA loader的全部构造器关键字及PROPRIO_DIMS，验证5×2按时间再分量打包成10维的符号索引。配置/AdaLN/module源码实际散列绑定；schema是人工从这些源码重建、由全部保存meta tensor记录和已运行严格加载凭据交叉核验，不是通用Python架构推导器。

| task | 当前权重完整SHA256 | 本地接入证据边界 |
| --- | --- | --- |
| PushT | 9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb | 原生4维proprio；保存CUDA strict load |
| PointMaze | a01d99c4592fbedf44af076cf4c339de230c56f9f377c7559f584b97569b59bc | 原生4维proprio；保存CUDA strict load |
| Wall | 8efb0623cfba1cb3ca210de26f7579c83dd24936635f11989c515afcb23bea1e | 原生2维proprio；保存CUDA strict load |

原始元数据：results/p5_jepa_strict_metadata_20261027_v3/report_native_interfaces.json；其SHA在新报告绑定。三份同目录的 task_native_strict_cuda_main_v1.json 明确 model_forward=false、encoder_runtime_verified=false。本次核验严格加载/冻结/eval/设备/计数/参数量及前后checkpoint和source SHA，并读取source当前字节匹配。不能把保存的strict-load CUDA升级为三任务预测/导数对照。

## 五处修改：精确可证部分与不可证部分

下表路径相对external/jepa-wms；每项新报告同时绑定HEAD原blob SHA、当前SHA和C1是否实际导入。完整AST比较忽略位置/注释，保留表达式/操作程序；未执行这些AST。

| 修改 | 当前可复算合同 | 仍须实测/人工决定 |
| --- | --- | --- |
| app/plan_common/models/prop_embedding.py | 两个精确getattr回写属性后，全文件AST等于HEAD；构造器直接无条件初始化use_mlp/tokens_per_step；其余计算程序不变 | 仅限普通构造实例、属性保留、同输入/运行时。缺属性旧代码报错而新代码用默认，不等价；不认证任意pickle/自定义属性语义 |
| app/plan_common/models/vit.py | 一个use_sdpa getattr回写后全AST相同，构造器赋值不变 | 不证明SDPA与manual两分支数值相同。若主线程要求跨backend/precision比较，须同mask/dropout/eval/input的CUDA输出与导数对照 |
| app/plan_common/datasets/pusht_dset.py | 投影保留decord路径后全AST等于HEAD；72个RGB/BGR/255缩放符号分量检查 | OpenCV真实seek、codec、色彩、帧顺序、dtype须实际decoder配对；此项可CPU解码，不必CUDA，但本次也未执行。下游特征/导数比较才需CUDA |
| app/plan_common/models/dino.py | local_source不存在时，投影恢复原remote fallback后全AST相同；其他分支/forward保持原样 | 当前默认本地目录为HF shim，不是相同实现；环境override未读取/认证。fallback AST等同不代表允许网络执行，也不代表当前encoder等价 |
| app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py | 精确识别padding节点、设备/dtype/拼接顺序；删除新节点并恢复原参数后残余全AST相同；完整切片条件可算 | 缺历史补零是输入语义改变，不是等价修复；视觉短于window也可能额外截断。实际预测/JVP影响待CUDA，而非靠数组差异判理论失败 |

五项中，保存C1实际导入的修改模块只有prop_embedding。C1直接使用AdaLN、official目录DINO和显式真实历史，不经dino.py的shim或vit_enc_preds.py的补零wrapper。其他四处虽被source绑定，不能误称被C1运行认证。源码图、路径、SHA一致是接线证据，不等于forward一致。

## Wrapper切片的精确判据（本次发现的额外边界）

令V为实际视觉token数，A为累计动作token数，W为窗口，均为正整数。原实现传最后min(A,W)个动作；新实现传C=min(V,W)个，A<C时前补归一化零。

对不重复的动作索引符号，新旧传入序列相同当且仅当：

    min(A, W) == min(V, W)

所以“A>=C即可等价”不正确：V=1,A=2,W=2时原输入[0,1]，新输入[1]，不涉及补零也不同。V=2,A=1,W=2时原[0]，新[零,0]。V/A/W各1..6的216个可复算案例：106相同、55补零改变、55截断改变。这是有限索引合同，不能外推为106个成功任务。

相同输入数组加相同残余计算程序，仍以相同运行时/随机性等条件为前提；并不通过本次审计认证数值稳定性、内核determinism或端到端性能。

C1保存两段5×2历史，共20个归一化分量；本次从dataset字面常量重算float32投影全部一致，20个均非零，替换为零时输入最大绝对差0.637444257736206。video_wm.encode_act的action_encoder_inpred分支只reshape后return a；因此在该分支归一化零不能理解为原生零动作，也不能代表真实历史。此数值是输入差，不是预测误差/收益/损害。

## 可直接交给main的剩余前置

1. 先固定实际接入路线与历史合同：direct C1真实历史还是wrapper补零。不要通过调beta/放宽容差消除不同输入，也不要拿两种合同混做同一模型对照。静态判据能先筛掉不可能输入等价的wrapper调用。
2. decoder fallback若将实际使用：固定episode/frame ids、变换前RGB字节、dtype/scale/transform和异常帧规则，真实decoder配对；此项不依赖P1，不一定需要GPU。仅本次用户禁令下未执行。
3. 若声明HF shim与official目录实现等价，必须面对已保存 results/p5_jepa_official_equivalence_20261027_v6/pusht_official_vs_hf_cuda_main_v1.json：8个episode均passed=false，status=paired_equivalence_failed。本次只读重验其SHA/范围，未重跑；不能忽略或通过选择结果/改阈值认证。该旧结果只约束原协议，不证明所有encoder或理论失败。
4. 三任务原生checkpoint仍需由main在冻结输入/配置/设备/精度和预先定义的比较规则下生成实际encoder→predictor输出与所需JVP等接线证据；PushT C1已有局部运行证据不能扩成PointMaze/Wall全链路证据。对无输出合同变化的getattr修改，不需要为了静态分支同一性虚构新的网络测试。
5. 本合同不解除权重远端认证、normalizer统计沿革、许可/训练成本等旧unknown，更不解除P1停滞/两段可达独立根、强基线本地配对与P5完整科学闭环依赖。full_equivalence/official_authenticated/P5_admission/execution_authorized均为false。

因此本次真正缩小的是“哪些本地编辑改变了什么程序/输入、在哪些条件下完全保留原程序，以及三checkpoint的完整接口是否吻合”。不将mock、reported异协议数字、严格加载或contact=0当作P5成立或理论否定。
