# Fixed164 observed raw-coordinate native physical adapter v1

本版是独立 ambient raw-coordinate 配对诊断，不是原 P1 合同替换，不部署候选，不签 gate/freshness。只新增本 source、test、innerdoc；旧 observed_suffix_physical 三件套保持冻结。worker 不运行真实环境、神经 CPU、CUDA、训练或物理生产。

## 真实接缝与固定设计

使用 Nash 冻结 `agent_observed_raw_coordinate_fd_20261028_v1` 的真实 prepare / `verify` / `plan_jobs(prepared, report, strict_receipt, authority)`。输入为 MAIN 实际 FD prepare 绝对路径、prepare SHA 和 producer source SHA；无未来 report SHA 假设，也不需要等待 FD CUDA 结果来安排物理动作。

prepare 完整重建 FD prepare（其 verifier 重建 observed-v2 编译全 strict、72/32、原 task/source/bytes），再沿原 observed-v2 midpoint→terminal→native authority 取得原物理 prefix51/cp5。不是拿 status/receipt 标志替代消费，不伪造旧 schema。新 execution plan 明确绑定 producer compact job hash 和 physical own plan hash。

每 panel 41 项：baseline 原 full10；随后 h=.01,.001，各 time-major raw coordinate0..9，sign −,+。4 panel 共164，2 known roots、0 fresh roots。original iterate0 suffix/prefix 锚点固定，不读新 winner/物理 outcome 排序。每项 literal float32(float64(x)+sign*h)，其他 raw bits 不变；不 clamp/drop/replace。域外或折叠使整份 manifest 非 ready，保留原164项诊断，不以合法子集开跑。仅 baseline 保存 original_AD_context/original_iterate0_context；其余必须为 null，并用同 panel baseline_reference_id，不能当丢失上下文重新生成。

observed_raw10 是 cp5 incoming raw history（warmup 尾5+prefix5），绝不是 future prefix5+suffix5。公共断言认证原 prefix5、original suffix anchor 和完整 task binding；baseline 另外逐字节要求 full10==原物理 plan full10。task 全体 goal192/Q/normalizer/context 由 producer/core 原认证，不拟合、不替换。

## 原生执行与完整消费

仅复用冻结 mixed.episode/native_hooks/render/labels：每项 fresh env、从 root 完整 replay/warmup，再执行原 prefix5+probe suffix5。新局部 Meter 在 step6 dispatch intent 和 callback **之前**认证全部 prefix51、cp5 state/snapshot/PNG/raw，写入 accepted cp5_guard；不匹配绝无 suffix callback。finally 仍保留 prefix_match 闭包。复用的 rows_for/audit_guard/audit_all_labels 按传入长度迭代，没有 fixed4 限制；新全局编排/validator 独立 schema。

成功全 horizon 上限/完整 census：

|资源|预算|
|---|---:|
|constructor / close|164 / 164|
|reset|328|
|public_step|3280|
|physics_step|33456|
|set_state|656|
|goal_setter / goal_render / goal_bind|492 / 328 / 164|
|explicit_render / PNG|10824|

physics=2×reset+10×public；每 env warmup10+plan10，66 PNG。所有失败/not_started/native_stop 仍留164分母，不补帧、不继承、不筛根。真实 native stop 可以少于上限；成功 census 检查实际步数公式与 reset328，不把早停伪作完整 horizon。

wall1800s（run 从 verify 前计时）；输出1GiB、磁盘余量 floor1GiB；预留128MiB final report，metadata 单件/单 journal record cap128MiB，不能误用8MiB或原32MiB目录 cap。Store 用实际 byte counter 与增量 SHA，避免10824文件每事件遍历/整条 journal 重哈希；consume 独立前后全盘 census/hash、完整 journal 链/事件 allowlist/嵌套 dispatch/receipt/每返回 native7 label/goal绑定 chronology、10824 PNG 字节/pixels/原视觉 guard。缺帧仍拒。不能闭合的损坏 journal 不签 receipt，原文件保留。

physical own digest 保持 mixed.core.digest（带默认空格）；producer job/manifest 使用 producer.digest（compact separators）。测试证明普通 [1,2] 两编码不同，不能整体切换 own codec 掩盖跨 producer 错配。

## CLI（MAIN 审读后运行；worker 不执行物理）

工作目录 /home/gbk/code/LeWorldModel/le-wm-main，模块 experiments.spectral_reachability.agent_observed_raw_coordinate_physical_20261028_v1。

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_observed_raw_coordinate_physical_20261028_v1 prepare \
  --fd-prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_observed_raw_coordinate_fd_20261028_v1_prepare_MAIN01.json \
  --fd-prepare-sha a1a87327d230af7e02e2379909305c9fbbfb8a05768ba85eb4fcbdac85f95ef5 \
  --fd-source-sha 83bf2420f5555f892ec1222bd158dde0634b096fa19f8a4bc626a2e07347970e \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_observed_raw_coordinate_physical_20261028_v1_prepare_MAIN01.json

.venv/bin/python -m experiments.spectral_reachability.agent_observed_raw_coordinate_physical_20261028_v1 verify \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_observed_raw_coordinate_physical_20261028_v1_prepare_MAIN01.json \
  --prepare-sha PHYSICAL_PREPARE_ACTUAL_SHA

.venv/bin/python -m experiments.spectral_reachability.agent_observed_raw_coordinate_physical_20261028_v1 run \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_observed_raw_coordinate_physical_20261028_v1_prepare_MAIN01.json \
  --prepare-sha PHYSICAL_PREPARE_ACTUAL_SHA \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_observed_raw_coordinate_physical_20261028_v1_MAIN01 \
  --authorization MAIN_OBSERVED_RAW164_PHYSICS_ONLY

.venv/bin/python -m experiments.spectral_reachability.agent_observed_raw_coordinate_physical_20261028_v1 consume \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_observed_raw_coordinate_physical_20261028_v1_MAIN01 \
  --report-sha PHYSICAL_REPORT_ACTUAL_SHA \
  --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_observed_raw_coordinate_physical_20261028_v1_consume_MAIN01.json
```

run 必须由 MAIN 在主机授权执行，不加载模型；其它三命令只读已有证据并独占写新 metadata。consume 返回自有 .consume，rows164、metric_breakdown164（index/panel/local_index/probe_id/status，缺失 outcome 为 null）、完整 counts/errors/bindings 与 physical_report_sha256。不伪装 FD 或旧 physical receipt。

## 回归与证据边界

开发首轮 RED 是缺新 module；初版6通过。随后真实 saved metadata smoke 捕获误认旧 plan 含 model_task、Nash baseline-only terminal context 的接缝；已按实际 binding/manifest 对齐。另一轮在测试 fixture 使用不存在的 base.digest 时报1 failed/14 passed，改为 physical own digest。上述是新 adapter 开发期错误，不解释历史 P1 负结果。

旧 RAM fixture 的32MiB cap 曾使 partial_native 缺终端闭包/full164成功路径触及 cap/floor；这是测试设施预算遗留。新 RamStore 使用本版1GiB cap 和增量 journal，生产 Store 同样按新预算，不改旧 fixture/source。

冻结前回归：主跑27 passed /281.76s，包含完整164成功 RAM-native lifecycle+saved consume、10824 PNG及reset328精确census；新增两项 baseline_ref/terminal_context 注入回归另跑2 passed、27 deselected /16.98s，最终收集29项，全部覆盖通过（不是声称一次29项总跑）。同一最终 source 的实际 FD prepare material() 全链只读 build 也独立1 passed /67.18s；主跑再次覆盖此项。未写 physical prepare、未运行真实环境。覆盖 baseline/history 差异、两 codec、164 budget、cp5 fail-before-suffix、失败全分母与 resealed mutation、实际 saved plan_jobs/native assemble/完整 prepare build。RAM Env 是 NumPy ABI fixture，不是物理收益/独立样本证据；saved build 也不是生产 run。

当前 worker 未发起真实环境、CUDA 或 producer 实验；MAIN 负责审读、prepare/运行/strict。所有科学/部署/P1 标志固定 false。

