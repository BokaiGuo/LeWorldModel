# 已知四分支 native-default overlay 成对 fidelity pilot v1

2026-10-27。只新增本 prefix 的 renderer、CUDA source、两份 tests 和本文；不修改任何旧 seal。
本交付只实现、执行 RAM mock tests；**没有 prepare、物理、神经、网络、训练或新的研究结果**。
MAIN 独占物理/CUDA、审代码并执行。不是训练 overlay 认证，不是独立新根，不开放 P1/P2 科学门。

## 先冻结的合同与两项边界

唯一实际来源：`results/p1_constructive_plan_physical_20261027_v1_main01/report.json`，
SHA256 `d205850e95f639b649fba9b5c6ee8059f2507431753a538b1ce30eccbd234300`。
固定顺序 root0 GN、root0 D2P、root1 GN、root1 D2P；不按 success、contact、Q 或未来 native 结果挑分支/帧。

原 physical source SHA `9dea58dfc6418dd01bf9ab28de6d712189ca1154a5cf29b165efa4b8ef65ea96`。
新 renderer **不调用**旧 `material/verify/load_source`，不开 sourcepack/certificate/witness。
从外部 SHA 绑定的旧 report → execution_binding → prepare → 完整 hash-chain journal，提取原 reset seed、
实际 `placement_request.requested/original_goal`、goal attachment 和 literal commands；replay **原实际 placement**，
不重新采样几何，也不从 future endpoint 重建动作。它不是重跑 source 构造/证书验证。
旧 prepare 里 privileged 文件 SHA 仅作已有元数据，不重新打开或 hash 其 payload。
只读其中明确绑定的 `s{slot}_blind_inputs.npz` 的 **goal_rgb entry**，用于与旧 recipe 完全相同的 task goal attachment。
该 RGB 是原批准任务，不用其构造新动作；不读取 witness/future_states。

两项不能过度声称：

- 原 artifact 没有 goal_pose 运行时 receipt。封存 native 源码 `c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2`
  的 DEFAULT_VARIATIONS 只采 agent/block，goal 默认 `[256,256,pi/4]`、render_goal=1。
  每次 capture 与 pair 前后核对此默认值及 variation 值；**不赋值修复默认、不改变 marker 对应的 goal_pose**。
  这是“源码派生默认 + 新运行时核验”，不伪称旧运行已记录 pose。它也不是 customtaskgoal marker。
- 固定 4×2=8 次 H1 只能测 factual，不能再测 factual-v-zero/mask action dependence。
  报 factual/persistence、predicted latent change、raw outgoing L2 和完整 normalized token；
  `action_dependence_measured=false`、counterfactual control=null。不能拿 predicted change 当动作依赖证明。
  如需真实 F-v-Z/M，必须另经 MAIN 批准 NEW 协议/调用预算，不能在本 v1 暗增调用。

## Renderer：原轨迹全重现，只有渲染 flag 配对

每分支原样执行两次 reset（原 seed，再 seed0+literal placement）、10 个 raw-zero warmup，
原 task goal attachment，再原封存 raw10×2 commands。goal attachment 与旧 recipe 相同：设置 task goal_state/_goal，
**不设置 goal_pose**；这不是 pairing 干预。两 view 在每个 pair 内 goal_state、_goal、goal_pose 均一致。

完整 off 保存 `warmup0..10 + plan0..10`，22/分支=88 张。每张先 durable 写 PNG，再对旧 receipt 的
state、observation_state、原 snapshot、pixel SHA、PNG **文件 SHA**与 phase/step/name 全字典精确比较。
每个实际 transition 另核 state/reward/terminated/truncated；不改成功判定、不移位、不插补 early endpoint。
失败图和最后返回 transition 先保存；step 调用本身若抛错，没有虚构 post-step observation。

固定 warmup5、warmup10、plan5(prefix5)、plan10(terminal10) 各一个 pair：
同 env、先 off 后 on，仅 on render 期间 `with_target=True`，finally 恢复 False。
off 本来就在88张中，因此额外16张 on，总104张 PNG，selected 是16pairs/32frames。
不去色、不 inpaint、不替换 default、无 pair 内 space.step/reset/setter/goal bind。

扩展 snapshot 对全部 space bodies 的 position/velocity/force/angle/angular velocity/torque/CoG，
space gravity/damping/iterations/collision 参数、原 state/snapshot、contacts/latest_action、body RNG、
goal_state/goal_pose 原浮点字节及 _goal RGB hash 做完全一致比较；每个 off/on render 后即核。
浮点字节包含 signed zero；不是 tolerance 比较。
renderer 的 screen、surface、shape.color 缓存不作为物理 snapshot；native renderer 本来会写这些渲染缓存。
**不声称序列化了不可见 Pymunk solver cache**。可见动态 snapshot 不变 + pair 内 step/setter 禁止 +
后续全 off 轨迹精确重现共同约束干预；不把 Python 可见 snapshot 叫完整引擎 checkpoint。

预算为真实 hook receipts，不把 reset 当零物理：

| 操作 | 完整期望/cap |
|---|---:|
| reset / 内部 goal render / setter | 8 / 8 / 16 |
| public step / space.step | 80 / 816 = 2×8 + 10×80 |
| explicit render / goal bind | 104 / 4 |
| selected pairs / images | 16 / 32 |
| neural / AD / training | 0 / 0 / 0 |

每次操作先 durable attempt、再 dispatch/completion；每次 reset 创建的新 space 都 hook。
同分支失败不重试，当前 run 停止，不跳到另一分支替补；四行固定分母和已保存 pair/incident 保留。
旧源码 seal、default pose 无法验证、off 不精确、pair 有变化或预算异常时 status=partial_or_failed。
seal 在 prepare 无法验证则启动前拒绝，不制造“已运行 partial”。运行末尾重验全部来源。

## CUDA：实际三观测历史，own-view label firewall

renderer 完整成功且得到 MAIN 的外部 report SHA 后，NEW CUDA prepare 重新验证来源，
解压完整单流 gzip 操作 journal，复算计数/nesting、全部88 off receipts/文件与16pairs；拒绝 partial。

每分支每 view 独立使用：

| 内容 | 固定来源 |
|---|---|
| latent states3 | 该 view 的 warmup5、warmup10、prefix5 |
| between packet0 | 真正 warmup5→10 的5条 raw zero |
| between packet1 | 该原 arm 已实际执行的 prefix5 raw commands |
| outgoing | 同 arm 原 sealed suffix5 raw commands |
| evaluator label | 同 view 的 actual terminal10 native192 |

不是复用原 warmup0/5/10 task history；不将 terminal10 放模型输入，不跨 view 混标签。
32 张固定逐张 encoder+existing projector，无去重，native float32[192] 单张存 NPY。
8 次 H1 均用原 TorchBackend.step 与 model.predict；shape 接线 predictor=(1,3,192)、AE=(1,3,10)，
native LeWM.predict 内部先 flatten，再 pred_proj=(3,192)，记录实际 method/pred_proj receipts（不是猜成1×3×192）。
模型前后 tensor hash、source/models/PNG seals、CUDA batch1/MATH-SDP/TF32false、no-grad/0AD 均保留。
原 public mean/std float32，time-major reshape，tile mean/std；无额外÷100、无 fit 或新 Q。
publicnormalizer 的训练 lineage 仍 unknown。

完整预算32 encoder/projector、8 predict/predictor/AE/pred_proj，输入 latent tokens24、AE packet tokens24、
pred_proj token samples24；AD/训练/优化/物理=0。model.predict 与内部 predictor 不相加冒充16次推理。
预测有完整输入/输出 gzip JSON；latent 有 NPY/file/rawbytes SHA。每项日志 sync-flush，最终 gzip finish；
调用中失败没有完成 receipt，保留 partial，不自动重试。模型/源码末尾再验。

每 view 报 mean((pred−actual_terminal)^2)、persistence mean((actual_prefix−actual_terminal)^2)、
predicted-change MSE 和 factual-minus-persistence。全部8分母列出，不设 scientific pass threshold。
overlay 也改变 target latent 坐标；off/on 各自标签上的误差变化不等于动作动力学改善，更不是训练合同因果解释。
后续 saved 复算可直接取 NPY + per-view fidelity.json.gz，不要求新建独立审计支线。

## CLI：仅 MAIN 执行 prepare、物理和 CUDA

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。下列路径均 NEW，不覆盖旧产物。
`RENDER_PREP_SHA`、`PAIR_REPORT_SHA`、`CUDA_PREP_SHA` 必须由 MAIN 获得实际完整64位值，禁止字面占位启动。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_live_overlay_fidelity_20261027_v1.py tests/test_p1_constructive_live_overlay_fidelity_20261027_v1_cuda.py

.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_live_overlay_fidelity_20261027_v1 prepare --physical-report results/p1_constructive_plan_physical_20261027_v1_main01/report.json --physical-sha d205850e95f639b649fba9b5c6ee8059f2507431753a538b1ce30eccbd234300 --output results/p1_constructive_live_overlay_fidelity_20261027_v1_prepare01.json

.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_live_overlay_fidelity_20261027_v1 verify --prepared results/p1_constructive_live_overlay_fidelity_20261027_v1_prepare01.json --prepare-sha "$RENDER_PREP_SHA"

timeout 650s .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_live_overlay_fidelity_20261027_v1 run --prepared results/p1_constructive_live_overlay_fidelity_20261027_v1_prepare01.json --prepare-sha "$RENDER_PREP_SHA" --output results/p1_constructive_live_overlay_fidelity_20261027_v1_main01 --authorize-physics

sha256sum results/p1_constructive_live_overlay_fidelity_20261027_v1_main01/report.json

.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_live_overlay_fidelity_20261027_v1_cuda prepare --paired-report results/p1_constructive_live_overlay_fidelity_20261027_v1_main01/report.json --paired-sha "$PAIR_REPORT_SHA" --output results/p1_constructive_live_overlay_fidelity_20261027_v1_cuda_prepare01.json

.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_live_overlay_fidelity_20261027_v1_cuda verify --prepared results/p1_constructive_live_overlay_fidelity_20261027_v1_cuda_prepare01.json --prepare-sha "$CUDA_PREP_SHA"

timeout 650s .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_live_overlay_fidelity_20261027_v1_cuda run --prepared results/p1_constructive_live_overlay_fidelity_20261027_v1_cuda_prepare01.json --prepare-sha "$CUDA_PREP_SHA" --output results/p1_constructive_live_overlay_fidelity_20261027_v1_cuda_main01
```

两 stage 各8MiB总输出 cap（含 partial+final，final预留256KiB）、200MiB floor；prepare各<256KiB。
renderer wall600s；CUDA wall600s/GPU allocated≤3GiB/reserved≤4GiB；MAIN外部 watchdog650s。
小于floor/cap gate拒绝，不删除旧研究证据；没有下载或 CPU neural fallback。
mock 不创建 temp/prepare/results/pyc/cache 文件。mock 通过不等于 native replay/goal_pose/model contract 已实际通过。

实现冻结前实际执行上述双文件 RAM pytest：**48 passed in 0.44s**。
覆盖全 replay kernel、首个 off mismatch 保存后停止、pair 变异/隐藏 physics 拒绝、default pose拒绝、
原资源/命令/goal seal、32帧join、own-view label firewall、8次全分母与 flattened pred_proj receipts。
没有实际调用 prepare、环境或模型；本交付不另开独立审计支线。

本实现使用 experiment-code skill 的先合同、JSON持久化、失败显式保留原则；用户限定 mock-only、无训练/作图，
优先于通用模板。仍待 MAIN 完整源码审查、真实 off 重现/default-pose/pair gate、实际 CUDA账本与误差。
