# Live overlay F/Z/M v2：原四分支、各自 view 的固定条件消融

2026-10-27。新增本 stem 的 source/tests/本文，不修改 v1 renderer/CUDA/prepare/result 或共享文档。
本交付只执行纯 RAM mock tests；没有 prepare、神经、物理、网络、训练或新实际误差结果。
个人 goal 已创建，active、无 token budget，API 无占用冲突。交付后由 MAIN 审源码、prepare、独占 HOST CUDA。
不另开独立结果审计支线。

## 1. 冻结实验：只填 v1 未测的动作条件控制

固定原四分支顺序：root0 line_search_GN、root0 full10_conditional_D2P、root1 GN、root1 D2P。
每分支固定 off、on；每 view 固定 F→Z→M，无随机候选、outcome选择、阈值调参、early best、重试或缓存。
共8个原 observed contexts，24次独立 H1（包括重复 F/M 输入也再次真实调用 AE/predict）。

唯一图像来源是 MAIN 完成的封存 renderer：

`/home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_live_overlay_fidelity_20261027_v1_main01/report.json`

本次只为实现接线核其文件 SHA：
`567e58769764f68107e099f4e2494159d9ef0076c7bbd73445a62245be6d4e80`。
这不是本 agent 对实际 renderer/CUDA 数值的独立审计。
不读 v1 CUDA report/latent/误差来决定控制、选择图像或复用其32次编码；本次另真实编码32次。

复用封存 v1 CUDA 的 `build` 仅作 saved-input 链验证：原 physical recipe/全88off文件/16pairs/32selected、
source/model/PNG seals；**不调用其 run**。它已有 fixed4原分支/default-goal-pose/配对纯度/资源验证，
此处不新建 renderer，也不改 goal_pose、goal_state、_goal 或原 literal commands。

硬绑定依赖 source：

- renderer SHA `27becaabeb6a54cc34810cb53655957fded9b0d7145470d95c9b599c7ce7534f`。
- v1 CUDA saved-input infrastructure SHA `67b028da9d5204f86a9be51596269a4ece2e66d13c3ca8f9a255fc13dfe1b2dd`。

任一来源不是完整 sealed 成功产物、SHA漂移、pair或原off轨迹证据缺失，prepare拒绝，不静默替代。
运行末尾重新验证完整输入链。未读取 sourcepack/certificate/witness payload。

## 2. 同 view 实际 context 与 labels-only 防火墙

每分支各 view 固定取 warmup5、warmup10、prefix5、terminal10，共4帧；4×2×4=32。
沿原 frozen encoder+existing projector 流程，RGB224 batch1、float32 native192；所有32分母，不去重。

| 字段 | 固定含义 |
|---|---|
| states3 | 该 view 的 actual warmup5、warmup10、prefix5 |
| between actions2 | warmup5→10的5条真实raw zero；该原arm的5条实际prefix raw |
| factual outgoing | 该原arm已封存且实际执行的suffix5 raw |
| evaluator label | 该 view 的actual terminal10 native192 |

publicnormalizer float32 mean/std不变，time-major reshape与tile；不额外÷100、不fit、不训练Q/readout。
raw zero必须先归一化，不等于normalized token zero；不清空history，也不把prefix5当outgoing。

实现把编码结果分为独立 observed字典(24项)与labels字典(8项)。`make_requests`只接受严格observed3白名单，
有terminal key即拒绝；没有label参数。将所有24个完整请求先写入 `requests.json.gz`，才开始预测。
请求只有branch/view/control/context/raw_between/outgoing等批准字段；没有terminal或其它view的latent。
预测回调只获context、outgoing、control；所有24个预测持久保存完成后才调用 `evaluate` join自身view标签。
这提供显式代码接口/时序隔离，不冒称OS级隔离。没有任何selector或下一步action选择。

## 3. F/Z/M 的具体干预位置

固定每个context分别计算：

- **F**：history的两包token保持实际值，outgoing=原sealed suffix5；AE处理全部三包，正常model.predict。
- **Z**：history完全同F，仅outgoing替换成5条raw `[0,0]`，再使用同public mean/std归一化；
  仍调用一次完整 AE 和一次 model.predict。不是normalized-zero，也不是全history-zero。
- **M**：AE输入完全同F，**实际重新调用完整AE**。取得 `[1,3,192]` 后clone，
  仅 `embedding[:, -1, :] = 0`，前两history embedding逐元素不变；用此conditioned embedding调用predict。
  不是attention mask、input token zero、全部embedding zero或另一条合法raw命令。

F/Z/M都从相同 observed context独立H1；没有条件之间的递推，没有用F预测替代observed prefix。
M不修改原AE输出，不改参数/buffers或模型API；与F一样全额计AE三包tokens。

NativeBackend 的 F/Z 路径与原 `make_transition` 同语义：
CUDA normalize_live_packet → cat(history,outgoing) → model.action_encoder → model.predict → last native192。
M只在AE输出到predict之间插入预声明clone+last-token mask。
GPU实际tokens与纯数组public映射精确比对；所有AE输入、mask前后embedding与完整predicted3保存。
model.predict内部仍执行predictor和pred_proj，pred_proj输入是flatten后 **(3,192)**，不是(1,3,192)。

## 4. 资源、durable payload、失败合同

| 资源 | 新调用完整期望/cap |
|---|---:|
| encoder / existing projector | 32 / 32 |
| H1 model.predict entry / predictor.forward | 24 / 24 |
| action_encoder / pred_proj | 24 / 24 |
| predictor latent input tokens | 72=24×3 |
| action_encoder packet tokens / pred_proj token samples | 72 / 72 |
| AD / training / optimization / physics / render | 全0 |

AE与predictor复用原module_guard，新增严格24 predispatch cap；不误复用v1的8或offline的20 cap。
model.predict method与内部pred_proj有实际attempt/return receipts，并同步CUDA后才标completed。
不会把model.predict、内部predictor和pred_proj相加宣称72次H1推理。
事件带image或branch/view/control/request_index scope，逐次sync-flush gzip，最后Z_FINISH。

输出包含：

- `binding.json`、32个native192 NPY(file/raw SHA、dtype/shape)；PNG只引用封存renderer，不复制大资产。
- `requests.json.gz`：24个先封请求，只包含合法observed context及控制，不包含terminal标签。
- `prediction_00..23.json.gz`：实际tokens、完整AE embedding_before/used、完整predicted3，按request_index绑定。
- `events.jsonl.gz`：各module/method实际attempt/completion、mask输入输出、图像/预测事件。
- `fzm_metrics.json.gz`及report.groups：固定8组、每组F/Z/M与自身factual标签的完整算术。
- `report.json`：固定32images、24rows（含not_started/failed_partial）、实际资源、首错与后续错误、
  模型tensor hash before/after、source_verified、全部已写artifact SHA。

继承原frozen checkpoint/source/model seals、CUDA only、MATH SDP、TF32false、eval/no-grad/AD拦截。
GPU allocated≤3GiB/reserved≤4GiB；内部wall600s，MAIN外部watchdog650s。
总输出cap8MiB包含partial/final，final预留256KiB；prepare<256KiB；始终保留200MiB floor。
任一操作失败不retry、不换帧/分支/标签/renderer；已完成预测与未完成attempt日志保留。
如step/module调用抛错，不能把没有返回的输出补成0或标completed。无CPU neural fallback、下载、模拟器import。

成本继承另列，不抵扣：renderer已有物理816space.step/80publicstep和104explicitrender；
v1 CUDA32encoder/8predict属于旧结果，v2新增32encoder/24predict，不能宣称本次整个pipeline只24成本。

## 5. 固定算术与不可跨越的解释边界

每branch/view列：`E_F/E_Z/E_M=mean((pred_condition−own_actual_terminal)^2)`，
`E_P=mean((own_actual_prefix−own_actual_terminal)^2)`，`E_F−E_P`、`E_Z−E_F`、`E_M−E_F`，
`mean((pred_F−pred_Z)^2)`、`mean((pred_F−pred_M)^2)`。
正的Z−F/M−F仅表示在这条factual标签上F较好；没有新通过阈值或最好条件选择。

只有F有实际匹配的outgoing→terminal轨迹。Z/M没有新物理truth：这些是action-conditioning消融，
不是zero/masked动作的真实动力学误差或物理因果效应。M不对应合法raw动作。
off/on各自target latent坐标不同，严格own-view评分，不将off预测与on标签混比。
overlay可改变representation/label/persistence；不能仅凭某view低误差归因训练overlay mismatch已确认。
checkpoint实际训练overlay/normalizer lineage仍unknown。已知四分支、最多两原roots，不是新独立样本，
不提供P1/P2准入、非贪心规划、成功控制或理论否定。

## 6. MAIN CLI

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`，只使用NEW paths。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_live_overlay_fidelity_20261027_v2_cuda.py

.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_live_overlay_fidelity_20261027_v2_cuda prepare --paired-report results/p1_constructive_live_overlay_fidelity_20261027_v1_main01/report.json --paired-sha 567e58769764f68107e099f4e2494159d9ef0076c7bbd73445a62245be6d4e80 --output results/p1_constructive_live_overlay_fidelity_20261027_v2_cuda_prepare01.json

.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_live_overlay_fidelity_20261027_v2_cuda verify --prepared results/p1_constructive_live_overlay_fidelity_20261027_v2_cuda_prepare01.json --prepare-sha "$V2_PREP_SHA"

timeout 650s .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_live_overlay_fidelity_20261027_v2_cuda run --prepared results/p1_constructive_live_overlay_fidelity_20261027_v2_cuda_prepare01.json --prepare-sha "$V2_PREP_SHA" --output results/p1_constructive_live_overlay_fidelity_20261027_v2_cuda_main01
```

`V2_PREP_SHA`必须是MAIN prepare实际输出的完整64位SHA；不是本交付捏造的prepare seal。
不再跑renderer，不覆盖旧prepare/results。本agent不执行prepare/GPU，也不接独立审计支线。

初轮RAM测试：**28 passed in 0.39s**，不创建tmp/results/prepare/pyc/pytestcache。
最终与v1两份冻结tests联测：**76 passed in 0.61s**（本v2新增28，v1原48）；旧两个source SHA复核未变。
涵盖F/Z/M语义、history/time-major/public mapping、label白名单与cross-view隔离、24独立调用、
method/projection24 cap及72token receipts、首错不重试、旧saved-input build复用不改v1。
mock通过不等于实际CUDA conformance或科学通过；待MAIN审读执行。

本实现采用experiment-code skill的先冻结协议、逐调用JSON持久化、显式失败边界；
按用户mock-only/无训练授权，不采用通用模板的训练、作图或“低成功率即代码错误”规则。
