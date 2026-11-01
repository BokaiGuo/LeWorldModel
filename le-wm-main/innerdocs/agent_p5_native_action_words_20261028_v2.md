# Native action_words v2：只修复v12 owner digest边界

本轮仅新增同stem source/tests/本文；v1三文件继续冻结，MAIN v1空prepare目录及任何结果均不修改。使用experiment-code技能作定点修复和回归；无训练、拟合、adapter、CUDA/CPU神经、physics、网络或新根选择。

## 唯一行为修复

v12继承v11的digest：`sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())`。源码已只读核实v11的这个实现；旧v1错误地将本模块的默认带空格JSON摘要用于该上游边界，导致MAIN真实prepare拒绝，没有理由放宽其他predicate。

v2新增专用`v12_manifest_digest`及单值拒绝检查`verify_v12_manifest_digest`，仅在prepare读完原v12 manifest/report后校验这一边界。原真实owner compact摘要严格为 `9283f25481795ecee6c0f8bdfbf5ee9f83bee96b945bd4ca80e071332dd83aa1`，不接受默认空格摘要作为第二个合法值；不改权威值。原manifest文件SHA `41ef9bf8f5c5ae5c4ff8890a25e6a83137b7e637058a8033281a5eabc9cea07f`、report文件SHA `d8667a4e8537a5e005d13507d4d05b8d7907996549889decc8a00e1db873eda6`及所有原资产pin均保持。

本模块自身`digest()`逐字保留v1默认空格语义；v2自己的prepare/运行报告/事件链/saved consumer仍用它，不能全局改compact。除版本TAG及这个owner边界外，source从v1复制，无程序、root、normalizer、原生递推、预算、错误清理、阈值或admission变更。tests的上游fake v12 fixture改为真实owner口径；下游fake v2 report仍用自有digest。

## 保持的协议与限制

原生接口与详细科学边界见已冻结 `innerdocs/agent_p5_native_action_words_20261028_v1.md`，其SHA `0acd2cc38ad9719fff0f49fad76230b8845015ca47c9b45a43e27f29a9e2fb55`。v2仍固定三根2026102702/03/04、四个两包raw10程序XY/YX/X−X/零零，已保存的真实历史视觉/动作/proprio；视觉及预测patchwise proprio共同递推，不能用真实未来或normalized-zero padding替代。共享proprio一次/root，无分支退款、replay或JVP。总24 predictor API/24 samples、3 proprio API/3 samples、0视觉encoder forward/AD/physics；各单位分别保留，FLOPs=null，不称等总算力。96MiB output、200MiB floor、600s wall及4GiB CUDA free scratch工程要求不变。

prepare/verify/consume仅stdlib/NumPy与原始文件hash/无pickle数组；原v8严格CUDA loader仅在显式cuda入口调用。保留12program失败分母、partial输入输出/attempt-return、SHA事件链、source/model状态检查及saved原值复算。consumer仍要求MAIN提供**实际新v2报告文件SHA**，本轮不预造。所有physical_release/scientific_admission/action_witness_established仍false；known三根不是fresh，不建立新checkpoint/新架构或C1/C2/P5科学通过。

## MAIN执行命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。选NEW目录；不要重用原v1空prepare目录。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p5_native_action_words_20261028_v2.py
mkdir results/agent_p5_native_action_words_20261028_v2_prepare_MAIN01
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_native_action_words_20261028_v2 prepare \
 --manifest results/agent_p5_native_action_words_20261028_v2_prepare_MAIN01/manifest.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_native_action_words_20261028_v2 verify \
 --manifest results/agent_p5_native_action_words_20261028_v2_prepare_MAIN01/manifest.json
```

MAIN审查批准后HOST CUDA，worker不运行：

```bash
env -u PYTHONPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_native_action_words_20261028_v2 cuda \
 --manifest results/agent_p5_native_action_words_20261028_v2_prepare_MAIN01/manifest.json \
 --output results/agent_p5_native_action_words_20261028_v2_cuda_MAIN01
sha256sum results/agent_p5_native_action_words_20261028_v2_cuda_MAIN01/report.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_native_action_words_20261028_v2 consume \
 --manifest results/agent_p5_native_action_words_20261028_v2_prepare_MAIN01/manifest.json \
 --report results/agent_p5_native_action_words_20261028_v2_cuda_MAIN01/report.json \
 --report-sha256 "$MAIN_ACTUAL_V2_REPORT_SHA256"
```

## 回归与未运行项

保留v1全部35项RAM测试，新增真实已保存v12 manifest/report文件SHA及owner摘要回归；tamper覆盖seed、源binding、model receipt、report摘要、错误自有摘要，并验证fixture重新pin文件也不能绕过owner边界，另核自有digest不变与v1三文件freeze。实际v2 **44 passed（1.88s）**，CLI --help通过。真实JSON只读，tamper只改RAM副本；没有重算/读取权重、运行模型或伪装新科学产物。未对真实资产调用prepare/verify/cuda/consume，未写MAIN results。最终三文件SHA在交付时列出。

剩余前置不变：MAIN真实prepare的完整源/资产闭包、CUDA strict load及运行尚待执行；缺失/漂移必须失败，不能通过修digest绕过。实际物理task witness、独立根、接触任务对齐与C2/P5仍未建立。
