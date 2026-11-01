# P2 full20 sampling：工程校准，不是阶段准入

仅新增同 basename 的模块、测试、本文；不得修改任何已有 sealed bytes。
P1 未通过，P1/P2_admission 始终 false。不训练/readout，不按物理结果选根、调阈值
或修改 proposal。本文不报告 GPU 或效益结果。使用 experiment-code 技能的固定调度、
可复现 seed、逐候选日志要求；本任务限制优先，不生成图、训练目录或效益声明。

## 动作与模型递推

原 compilation 固定为
`results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json`，
核验其既有 SHA、source seal、官方权重/config 和全部 input bindings；再核验
原 P2 paired report、protocol、源码镜像与动作见证。保留原全部四根、原顺序、
完整 root/goal RGB 绑定、goal_vectors[0] 和保存的 Q，不读取物理 outcomes 做选择。

每个 proposal 是 raw float32 `[2,5,2]`，按 time-major reshape 为 `[10,2]`。
20 个坐标独立采样，不重复二维命令、不从旧 bank 取值、不额外 tanh。
每个 raw 坐标在 [-1,1]，这是源 live command 盒，不是 normalized action 盒。
使用与 `pusht_full_packet_compiler_20261027.make_transition` 相同的公开 mean/std
和 `normalize_live_packet`：每 packet 按时间展平后 `(raw-mean)/std`，2维 stats
按 tile5（不是 repeat-interleave）。prepare 记录 stats 并核验 domain normalizer SHA。

生产复用 RootAnchoredCudaReplay/CudaReplay：emb `[1,3,192]`，between history
`[1,2,10]` 来自 `compiled.input_context.actions[:,1:]`。拼接新 token 得到三个
outgoing packets，调用 action_encoder 和 predict，取最后 successor，emb 左移，
history 更新为 outgoing[:,1:]；第二个 packet 从更新后的完整上下文继续。
不额外 bootstrap、不编码新 RGB、不训练。神经 CPU fallback 被 adapter 拒绝。

## 冻结调度和比较边界

- 每 root/arm 40 个完整 H2 proposals，batch1，恰80 planning predict calls/samples。
- CEM：4代×10，elite2，逐坐标 Gaussian mean0/std0.5，clip raw box；每代第一个
  proposal 是当前 mean，population std ddof0，floor0.05，无平滑。无额外 mean evaluation。
- Random：第一个 proposal 为 zero，后39个 full20 raw uniform。zero 包含在40内。
- 每 root 两 arm 分别重建独立 RNG，seed 依原 root_index 固定为
  20261027/20261028/20261029/20261030。best-ever 最小有限保存 Q cost，首次 tie 保留。
- winner 单独逐段重放2次，然后从原 root 重新完整重放2次，要求 search/segmented/full
  的完整上下文精确一致。4次 replay 归 evaluation，不混入 planning。compose 只封装
  已核验状态，不进行隐藏模型调用。receipt 含完整 context、raw actions、edges、provenance。
- 每 arm PredictionBudget 上限84；全部正常时四根两 arm 共672次 predict。
  记录每次实际 predict/action_encoder entry、batch、context tokens、scope、完成/异常，
  与逐 packet dispatch 及分阶段预算独立核对。正常每 arm 两类调用各84次、各252 context
  tokens。失败不退款、不补候选、不 padding；保留全部尝试，任何失配不许消费为完整计划。

v1 的 constant2 repeated5 实际是4维；新 runner 是20维；旧16-command bank 则是
离散有序双段。三者 prediction budget 可对齐，但空间/proposal prior 不相同，不能
把差异单独归因于组合结构。仅匹配 predict calls/samples，不匹配 FLOPs、walltime、
上游 encoding/bank construction 或总算力。Q 未修复，不据此宣称物理任务收益或官方复现。

## 主线程命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。先停止修改源代码，再 prepare，后由主线程
调度 host CUDA；输出必须是新的独占目录，重复启动/prepare 拒绝覆盖，不支持续跑。
`--paired-report` 默认解析为项目内原 P2 paired report 的绝对路径；下列命令显式写出以便审计。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p2_full20_sampling_20261027_v2 prepare \
  --paired-report results/pusht_p2_matched_composition4_v1_20261027_cuda/p2_matched_report.json \
  --compilation results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/pusht/lewm \
  --output results/p2_full20_sampling_20261027_v2_cuda_run1

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p2_full20_sampling_20261027_v2 cuda \
  --output results/p2_full20_sampling_20261027_v2_cuda_run1
```

prepare 不分配 GPU、不实例化神经模型；只核验已有输入并创建新 protocol/prepared 和
源码镜像 seals（包括本文、测试、compiler、adapter、normalizer 和已加载依赖）。
cuda 在分配前重新核验所有绑定/源码，运行后再次核验参数/缓冲区 hash、frozen eval、
输入及源码。输出独占 started、逐候选 JSONL、逐 root-arm JSON、最终 report。
加载/环境 fatal 异常可能只有 started/journal；不伪造完成报告，也不自动重试。

## 物理消费接口（不自动执行）

`load_prepared(output)` 先核验输入和 seals，再读取 report 并要求其 source_seals 与
prepared 完全一致；执行前独立绑定 report/protocol/prepared 的文件 hash。
`selected_actions(report, protocol, compilation)` 核验全部四根两 arm 的 receipt、
40候选、full20映射、best cost、goal0/Q、provenance、actual entries，返回
`{(root_sha, arm): ndarray[10,2]}`。
`plans_from_saved(...)` 导出八份兼容 `p3_matched_physical_20261027_v1.physical_branch`
的 plan（含 selection_sha256），按 root_index 奇偶交替 arm 顺序。
物理主线程仍须独立核验实际 root/goal RGB、绑定物理 executor 源码并持久化全部分支，
不重选、不补零。旧 sampling physical CLI 写死 v1 导入，不可直接使用新 schema；
本任务只提供消费接口，不增加或运行物理 driver。
主线程独立维护 `p2_full20_physical_20261027_v2`，通过本模块 `plans_from_saved` 消费，
不应修改或 monkeypatch 本 runner，也不应复用 v1 的 constant2 校验。
保存内容和 hash 自洽不等于真实性证明；真实独立 replay 只在 CUDA 阶段完成。

## 非神经验证

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_p2_full20_sampling_20261027_v2.py
```

覆盖20坐标/time-major映射、两 packet 非常量、固定调度、seed与tie、失败记账、
replay/witness、非法输入/篡改拒绝、metadata-only实际调用 wrapper/恢复、八份物理
plan export，以及真实已有文件 prepare/load/seal（写入测试临时新目录）。
归一化测试仅 CPU tensor 算术，无神经 forward。合成回调不是效益证据。
