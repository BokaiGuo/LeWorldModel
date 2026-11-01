# v5 独立 terminal RGB runner：真实 saved join 闭合

仅新增本 stem source/tests/doc。v1–v4和失败文件均保留，不改旧source/globals，不再用任何 monkeypatch/delegate runner。v5拥有自己的prepare/verify/run_cuda/consume/encoder/meter/输出事件链，旧v2仅复用已绑定SHA的constants和stdlib saved IO函数；protocol在import时deep copy，不随旧globals改变。测试真实九文件prepare→verify，另以纯hook/token stub走v5自身run_cuda成功/失败/cleanup→consume路径，不是神经实验。

## 修复闭合

公开顺序唯一：goal prepare/report/receipt → physical manifest/report/receipt → numeric manifest/report/receipt。函数内给内部saved join的顺序明确为goal→numeric→physical，verify从各具名key还原public顺序。physical wrapper.consume接收manifest Path `sm`，不是解析dict `smv`。path与report对象使用不同变量名；bindings仅为绝对真实path→实际SHA，不把标签当路径。goal/numeric/physical的三个真实saved consume返回值均与各自实际receipt精确相等；任何缺项不降级。

goal patch、四prediction token、真实RGB、full precision state与transition及所有source/三件套进入闭包，SHA/bytes逐项验；verify重新prepare深比较。native7尾两维为agent_vx/vy，完整block/agent double来自实际public body，不误用state7为block velocity。

## 独立 CUDA 与消费

MAIN明确确认才加载原冻结loader：仅HOST CUDA，无CPU神经fallback。只编码xy/yx/x_minus_x/zero_zero全部四个frame20 RGB，新增4API/4samples/4frames；batch1。goal与predictor不重跑；原make_transforms224、x_norm_patchtokens256×384、math SDPA与原loader来源限制不变。所有模块前置hook就位，proprio/predictor调用拒绝；失败attempt不退款。finally清理hooks、核expected==before==after、最终实际import源码闭包、重新verify；任何工程/cleanup漂移使strict拒绝。

每词完整float32token独占写入+fsync/目录fsync，实际attempt→return→artifact共12个SHA链事件；consumer严格重放事件顺序、word顺序、meter API/sample/frame、全四token path/SHA/bytes/shape/dtype、精确目录inventory、wall/cap，重算所有指标后再次核authority。partial保留4分母与实际失败，不补0，不覆盖重试。

诊断逐词列predicted→goal MSE、real RGB feature→goal MSE、prediction→real feature MSE和前两项之差；全用float64 saved算术，不做CPU神经。coverage/reward/done/agent+block doubles独立列，不相加或调分；RGB agent/目标raster等混杂不作因果物体归因。无selected word、无新task成功门、不称zero选择为理论失败。单已知构造OOD根，非fresh、非跨模型闭环、非C2决策收益，科学准入始终false。

工程边界32MiB output含1MiB终端reserve，200MiB disk floor，600s从saved preflight后计时覆盖CUDA加载到最后verify，4GiB CUDA free scratch非峰值保证。SIGALRM对C/CUDA可能延迟，strict最终墙钟仍须低于600s；无法持久化终态即无有效回执。新旧资源单列，不把API/samples/wall或physics相加，不称等总FLOPs。

## 测试与精确调用

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。真实saved tests会读取九文件并调用原完整saved validators，但不会import运行环境/模型，不执行神经/物理。stub测试只记录hook，不是科学证据。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p5_terminal_goal_attribution_20261028_v5.py
```

以下prepare输出必须尚不存在，所有失败目录/文件保留；本次worker实际运行的saved prepare验证使用tests的临时目录，不占用MAIN命名输出：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_terminal_goal_attribution_20261028_v5 prepare \
 --manifest results/agent_p5_terminal_goal_attribution_20261028_v5_prepare_MAIN01.json \
 --authority \
 results/agent_p5_frozen_goal_scoring_20261028_v1_prepare_MAIN01.json \
 results/agent_p5_frozen_goal_scoring_20261028_v1_cuda_MAIN01/report.json \
 results/agent_p5_frozen_goal_scoring_20261028_v1_consume_MAIN01.json \
 results/agent_p5_contact_action_words_physical_20261028_v1_prepare_MAIN01.json \
 results/agent_p5_contact_action_words_physical_20261028_v1_physics_MAIN01/report.json \
 results/agent_p5_contact_action_words_physical_20261028_v1_consume_MAIN01.json \
 results/agent_p5_contact_action_response_20261028_v1_prepare_MAIN01.json \
 results/agent_p5_contact_action_response_20261028_v1_cuda_MAIN01/report.json \
 results/agent_p5_contact_action_response_20261028_v1_consume_MAIN01.json
sha256sum results/agent_p5_terminal_goal_attribution_20261028_v5_prepare_MAIN01.json
```

MAIN填实际prepare SHA（不预造）：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_terminal_goal_attribution_20261028_v5 verify --manifest results/agent_p5_terminal_goal_attribution_20261028_v5_prepare_MAIN01.json --manifest-sha256 ACTUAL_PREPARE_SHA

env -u PYTHONPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_terminal_goal_attribution_20261028_v5 cuda --manifest results/agent_p5_terminal_goal_attribution_20261028_v5_prepare_MAIN01.json --manifest-sha256 ACTUAL_PREPARE_SHA --confirm-host-cuda --output results/agent_p5_terminal_goal_attribution_20261028_v5_cuda_MAIN01
sha256sum results/agent_p5_terminal_goal_attribution_20261028_v5_cuda_MAIN01/report.json

OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_terminal_goal_attribution_20261028_v5 consume --manifest results/agent_p5_terminal_goal_attribution_20261028_v5_prepare_MAIN01.json --manifest-sha256 ACTUAL_PREPARE_SHA --report results/agent_p5_terminal_goal_attribution_20261028_v5_cuda_MAIN01/report.json --report-sha256 ACTUAL_REPORT_SHA --output results/agent_p5_terminal_goal_attribution_20261028_v5_consume_MAIN01.json
```

真实CUDA编码与strict回执尚待MAIN审读后执行；worker没有运行神经/物理/CUDA/网络。
