# P2 step5 原冻结 encoder/Q 输入桥与严格 consumer

## 交付范围

独占同名 source/tests/innerdoc 三文件。旧 gap、P2 executor、封存 source/results 与 MAIN 文档完全不改。本文件说明实现而非执行结果。worker 没有运行 CUDA、CPU 神经、physics、训练、网络或后代代理。MAIN 独立审读后决定实际 HOST CUDA 调度。测试只用 RAM NumPy/PIL、假的模块接口以及真实 saved metadata，不把 mock 数组当研究证据。

上游固定：MAIN gap report `results/agent_p2_native_composition_gap_20261028_v1_MAIN01.json`，SHA `fc9d1ae5ff7751821380ea92b1e07c85c374534abb1e7d8cc249d85c9dedd612`。`collect()` 检查该 SHA、冻结 gap source SHA，并重新构建 gap 的完整四根八臂 saved 结果，不能换根或计划。输入依然是原四个 known roots、joint/composition 两臂、执行10步；不是 moving 两根，也没有 fresh-root 资格。

## 已核查的实际资产

1. 原 P2 winner 中两臂均有真实、已付费 `segmented_contexts=[initial, mid, end]`，对应原 source 的 selected verification replay，而非新搜索边。原 joint 的一条10步 witness edge 不被改造成两条边。
2. midpoint 证明：初态等于 selected plan predictor_context；end 等于 whole replay、selected terminal 与原 witness receipt；mid 三 latent 的前两项等于初态后两项，end 前两项等于 mid 后两项；原 float32 mean/std 对 raw5 归一化后逐值等于 mid incoming token，raw10 对应 end 两个 token。`mid.latent_history[-1] == end.latent_history[1]` 是核对结果，而非直接假设一个 index。composition 还必须与原 prefix edge 的 output_context/raw5 精确一致。
3. 原 ledger 最后四条 replay 请求分别核 path `[80]/[81]/[82]` 或 `[40]/[41]/[42]`、packet、返回状态及 predict/sample/action-encoder 计数。对应三个 replay contexts 的原程序源码、root JSON、原 protocol 及 normalizer 均继承 SHA；不导入旧 CUDA executor。缺 mid、缺调用证据或历史不闭合，prepare 报具体错误，不跑新 predict 补齐，不拿终态 latent 顶替。
4. actual step5 由每臂封存 observation journal 的第5条返回连接到精确 PNG path/file SHA、原七维 state、全部10条动作/返回 trace。逐项核对 source report artifact SHA、第五条 command、原 root/history/goal RGB 身份、raw5/raw10、plan selection/handoff、原 goal0/Q 身份。
5. 已扫描实际44个 P2 `*.encoding.json`，没有与这八张 step5 PNG SHA 匹配的已有编码。范围是既有 P2 endpoint receipt 约定，不宣称读懂任意未索引 NPY。匹配到旧 step5 receipt 时 failclosed，交 MAIN 核既有证据复用，绝不默默再编码。旧终态8个真实 CUDA 编码已由 gap 身份闭合，consumer 直接复用，不再编码。
6. root0 两臂 raw prefix、mid context 与实际 step5 PNG 相同，因此是八臂、七张唯一 step5 图像；仍按事前每臂一次计8个样本，记录 unique count=7，不去重偷减账、不补预算、不称八个独立状态。

`existing_encoding_inventory` 排除本版本名称下的自有输出，且只扫描 `.encoding.json`，本版本 prepare/run/consume/analyze JSON 不影响 collect/verify。测试覆盖本版本同名 `.encoding.json` 也不污染 inventory；新的其他版本匹配 receipt 会明确阻断并要求核对，不按结果选复用资产。

## 真 config / transform ABI 与冻结精度

原 checkpoint config：encoder 为 tiny ViT，image_size=224、patch_size=14，CLS 之外 `16*16` patches，所以 encoder output **`[1,257,192]`**，不是197。`encoder_abi(config)` 从绑定 image/patch/projector input/output dim 推导并写 protocol；runner 输出持久化入口与 strict consumer 均按此 config 检查。测试使用257，另以197构造所有 NPY SHA/journal 都有效的 RAM 结果，仍必须因 ABI 不符被拒绝。

开发期间 MAIN 预审捕获过197硬编码；尚未实际 prepare/run，修正前 mock 通过不代表真实 ABI 通过，更不是研究负结果。保留这一限制说明，不能将未执行的代码错误当理论失败。

原 transform 原样调用 `p0._image_tensor + image_transform()`：uint8 HWC → float32 scale(1/255) → 原 public ImageNet mean/std → Resize224 → `[1,1,3,224,224]` CUDA。public constants 用 AST 从被封存的 `stable_pretraining/data/dataset_stats.py` 提取，不导入 torch、不拟合 normalizer。PNG 原本就是224方图，Resize为identity；strict consumer 用独立 float32 NumPy mul/sub/div 顺序精确复核 normalized buffer，并核 `normalized_pixels[0] == encoder_input`。worker 没有执行 torch transform 或任何神经 forward。

仅显式 `run --authorize-host-cuda` 可导入 torch/loader：

- 原本地 weights/config；`pretrained=False`，没有下载；全模型 frozen eval CUDA，不存在 CPU forward fallback。
- 显式 `sdpa_kernel(SDPBackend.MATH)`；flash、mem-efficient、cuDNN SDPA 均须 disabled。
- `torch.backends.cuda.matmul.allow_tf32=False`，`torch.backends.cudnn.allow_tf32=False`。
- 显式 `torch.autocast(device_type='cuda', enabled=False)`，每次 native module 进入前检查 autocast disabled。
- 持久化 precision_before、每个 module entry precision、precision_after；strict consumer 全部逐项核原合同。model state hash 与 source/config/weights 文件 SHA 前后检查，退出时恢复精度上下文/TF32设置，恢复失败计工程错误。

源码闭包还包括 torch attention/backend 实现、原 loader/transform、LeWM/projector、ViT、torchvision transform 与数据统计源。这里只证明实际运行时检查了配置与 buffer；worker 未运行 CUDA，不能提前宣称设备端等价性已经成立。

## 资源与持久化

固定 root-major、每根 joint 再 composition；全4根8臂、每臂step5一次。新工作恰为上限 **8 encoder entries + 8 existing projector entries**，batch=time=1，0 predict/predictor/action-encoder/pred_proj/AD/physics。原 goal/Q、动作 normalizer 不变；不训练，不重编译，不重选计划。根或帧未知缺失直接 failclosed，不填0。

meter 包住本次新模型实例的真实 encoder/projector forward，不调用旧 execute、不改旧 module globals。predict/predictor/action-encoder/pred_proj 与公开 AD 入口被拒绝；CPU module 输入在 callback 前拒绝。模块 dispatch attempt 先 fsync，再保存原生输入 NPY，记录 native enter 后才调用；native return 与数组 saved 分开，失败收费不可退、不可重试。encoder 完整 last_hidden_state、projector 输入/输出、原 RGB、normalized input 和 native192 持久后才进入下一帧。

每个 NPY 保存 float32/uint8 原精度、shape、dtype、SHA，禁pickle。strict consumer 核原 RGB bytes/pixels、归一化输入、encoder `[1,257,192]` CLS→projector `[1,192]`→native192 全闭包，精确核所有路径与顺序；不是只相信最终一个 latent。journal 有 index/previous/event SHA 链，并核精确 frame attempt、module attempt/enter/return、frame saved 序列和终态 SHA。旧终态不能代替 midpoint。

constructor/加载、GPU allocation probe、CPU transform、hash/IO 是 setup/总wall开销，不隐称新的 predictor API。公开 module attempted/entered/returned/saved 分轴计数，并保存同步后的 module wall；`CUDA_ns/FLOPs=null`，不是0。继承 planner/derivative/search、旧physics/终态编码费用另列，不称等全资源成本或任何物理收益。

独占 NEW prepare dir/source snapshot、NEW sibling launch claim、NEW run dir、NEW consume JSON，xb 拒覆盖。不改 frozen prepare；一次 launch claim 不能自动重试。64MiB输出 cap、200MiB剩余空间 floor、900s operation-boundary wall，非强制抢占 watchdog。工程失败保全 full8 rows（failed/blocked_not_attempted），不换动作、不补 neural 调用、不删除失败目录；strict consume 拒绝 partial。

## 可执行接口（仅 MAIN 实际 run）

目录 `/home/gbk/code/LeWorldModel/le-wm-main`，以下 prepare/verify/consume 为 saved-only。source/tests/doc 冻结且 MAIN 审核后再 prepare，随后保存 prepare 返回的实际 SHA；不伪造未来 SHA。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_midpoint_teacher_20261028_v1 prepare --output results/agent_p2_midpoint_teacher_20261028_v1_prepare_MAIN01
```

MAIN 将 prepare 输出的真实 `prepare_sha256` 赋给 `P2_MID_PREP_SHA`（不得填假值）。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_midpoint_teacher_20261028_v1 verify --prepared results/agent_p2_midpoint_teacher_20261028_v1_prepare_MAIN01 --prepare-sha "$P2_MID_PREP_SHA"
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_midpoint_teacher_20261028_v1 run --authorize-host-cuda --prepared results/agent_p2_midpoint_teacher_20261028_v1_prepare_MAIN01 --prepare-sha "$P2_MID_PREP_SHA" --output results/agent_p2_midpoint_teacher_20261028_v1_cuda_MAIN01
```

仅 MAIN HOST CUDA 执行第二条。终态完成后，MAIN 把返回的真实 `report_sha256` 赋给 `P2_MID_REPORT_SHA`：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_midpoint_teacher_20261028_v1 consume --run-dir results/agent_p2_midpoint_teacher_20261028_v1_cuda_MAIN01 --report-sha "$P2_MID_REPORT_SHA" --output results/agent_p2_midpoint_teacher_20261028_v1_consume_MAIN01.json
```

consume 输出全8 midpoint predicted/teacher 原Q、latent/projected MSE、原七维分解、旧终态预测/teacher/native指标及四对 signed differences。只有同Q相减；native metric另栏。没有新增科学阈值、胜者选择、误差修复或模型拟合。

## 测试、未运行与科学边界

已实际 saved `collect()`：8个真实 midpoint/frame闭包、44份旧编码 inventory 无匹配、7张unique step5图。单测 RAM fixtures 不启动模型：257通过/197拒绝，真实config推导，public normalization闭包，精度状态变异，mid缺失/终态替代/错token/错动作/ledger缺失拒绝，CPU/shape/order predispatch拒绝，首次失败保全8分母，resource/journal/NPY/model hash变异拒绝，public prepare/verify/consume RAM端到端、冷子进程禁止导入torch/gym/physics并实际读saved。子进程是隔离测试，不是子智能体。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_midpoint_teacher_20261028_v1.py tests/test_agent_p2_native_composition_gap_20261028_v1.py
```

最终实际测试计数及三文件 SHA 以冻结通知为准。未运行真实 prepare/run/consume，未获得本轮新teacher latents；这些是 MAIN 下一步待执行项，不能把 RAM 通过等于设备端通过。原 midpoint prediction 已存在，没有需要新predict补齐的缺口；剩余真实依赖是 MAIN source审读、有效hostCUDA、冻结依赖/weights不漂移及本轮8次编码返回。7张unique图不构成7个独立root。P1/P2/freshness仍未准入；step5/10只能定位已观察packet边界，不能宣称误差发生的精确子步、因果机制或理论不可能。0/4旧成功记录保留，不改变原success/计划/门槛。

个人完成条件是 source/tests/doc、RAM/saved证据、可执行MAIN接口、SHA后冻结；不是科学goal完成。冻结后不再 append，任何改动只能新版本。
