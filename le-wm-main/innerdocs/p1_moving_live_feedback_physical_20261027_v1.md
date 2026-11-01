# Moving live O/I/R continuous driver v1

本实现依据外层 `docs/D2P_MOVING_OBSERVED_FEEDBACK_PROTOCOL_20261027.md`。
交付仅 source + RAM mock tests；MAIN 独占 prepare、实际 HOST CUDA 与物理。
不表示已部署反馈，不表示 P1/P4/独立确认通过；不产生新的独立根。

## 固定接口与执行

固定顺序为 slot0 GN、slot0 D2P、slot1 GN、slot1 D2P，每层 O/I/R；
固定 12 rows，独立任务仍为 2。每次 fresh metered constructor，原两个 reset/placement，
真实非零 warmup10，原 goal 绑定，固定 prefix5，本次 step5 RGB 编码，seal 后 step6–10。
无中途 reset/restore、旧 prefix 图替代、结果过滤或失败 fallback。

driver `Handler.encode_observation(request, artifact_sink, *, model, device, check_limits)`
自行读取本次 PNG 并执行原 transform → frozen encoder → existing projector。
原 encoder 的 `Ledger` 无 stage 字段，这里只按实际 encoder/projector 事件入账。
`Handler.compile_suffix` 和 `verify_suffix_seal` 只委托冻结 suffix：
`p1_moving_live_suffix_cuda_20261027_v1.py`，SHA
`b421af2a74bf8af5639f043ae3a06790895f24157880e67b556695b82bb348f6`。
不要求该 suffix 提供 encoder，不修改其 globals/闭包。

run 入口只加载一次原模型，共享 `runtime={model, device, check_limits}` 给全部编码和 suffix。
prepare 保存 `suffix.modelbinding()` 原始完整对象及其 source_closure/model_files；
run 将同一个绑定装入每个 task。normalizer、goal、identity192 Q 均不变。
I/R context 为 `[原z5,原z10,imagined_or_own_observed_prefix]`；history 是真实 warmup 后5 +
实际 prefix5，经原 float32 subtract/divide 后 time-major reshape(2,10)，没有二次 /100。
I 请求仅带实际编码的 hashes，不含其 latent/path；R 带本 episode 实际 latent。
packet 精确白名单和 digest 与冻结 suffix 一致；verify 返回 raw5，不是布尔值。

状态/snapshot/goal 原生标签仅用于守卫和评价，未传入 compiler。
历史 physical report 仅在全部控制执行/封存后进入 final evaluator，用于核原 prefix；
不从历史结果产生动作或筛选模式。O/I/R 的新 prefix PNG/state/snapshot/raw 必须同层精确一致。

## 资源与失败语义

新增 neural cap：24 encoder +24 projector；40 predict API，各40 predictor/action_encoder/pred_proj；
其中包含8 AD-forward、8 Jacobian192×10、8 SVD、8 jacobian API、8内部 grad API、1536数学 VJP 方向。
这些是不同层级 receipt，不可把 API 与 module 相加当总 forward，也不宣称总 FLOPs 公平。
I/R 每次固定五阶段 fresh H1 ridge，必须派发新 ridge，不按候选代价 fallback；O 不编译。
旧完整 GN/D2P、原编码/source 获取成本是继承已付成本，不在新增账内也不称免费。

物理 cap：constructor12/reset36/set_state72/goal_render36/goal_setter48/goal_bind12/
public240/physics2688/explicit_render288。正常无 constructor reset 时实账应为
constructor12/reset24/set_state48/goal_render24/goal_setter36/goal_bind12/
public240/physics2448/explicit_render264。
sealed `moving.attach_goal` **调用** `_set_goal_state`，所以 goal_setter = reset + goal_bind。
constructor 在原生初始化前由局部 subclass 计量，复制 EnvSpec、不改 registry/global；
每个新 space 在 setter 首次 integration 前包装，不能只计 public step。
额外 constructor reset 计入实际账，不预填发生次数。未知 registry/native source 拒绝。

全部操作 attempt/return durable，失败不退款、不补步、保全部12分母。
合法 native stop 不算工程错误；缺少编码/后续 checkpoint 保缺失，不凑齐神经 cap。
全12完整时才要求完整 neural census；partial 按实际 reset/public/render 数验账。
native_success 从实际 `eval_state`、reward、terminated 一致性绑定，完成10步不是成功。
清理失败保 secondary，工程异常保 first error。无 retry/CPU fallback/训练。

600s 全 job 从 run 入口开始，含来源复验、import、模型加载和执行；
`wall_seconds` 报到 final checks，`execution_wall_seconds` 单列 execute scope，不能混称。
物理 dispatch 亦检查同一全 job deadline；显存 allocated3GiB/reserved4GiB。
输出64MiB recursive（包括 suffix namespace 的 J/原预测/receipt），普通写保256KiB final reserve，
每写及 dispatch 保200MiB；prepare256KiB。cap 是工程拒绝边界，不保证真实全账必能完成。
没有运行 teacher 或额外神经诊断。

## MAIN manifest 精确字段

所有路径使用绝对路径；以下 `<...>` 必须由 MAIN 填真实 existing 路径/SHA，不代表 pending 成功。
schema 严格不接受额外字段。

```json
{
  "schema": "p1_moving_live_feedback_physical_20261027_v1.selection",
  "sourcepack": "<absolute original sourcepack.json>",
  "sourcepack_sha256": "c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6",
  "source_prepared": "<absolute original source prepare>",
  "source_prepare_sha256": "<external actual SHA256>",
  "planner_output": "/home/gbk/code/LeWorldModel/le-wm-main/results/moving_history_planners_cuda_20261027_v1_cuda_main01",
  "planner_report_sha256": "cd8432c06288577d25d9d9b92c7eb3673de592499ee64cae93238156f3f19aa0",
  "encoder_report": "/home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/report.json",
  "encoder_report_sha256": "e7fb554327b3bb5a96190f6291f8450b2b0723cd981817715cfb7e50aa266bf6",
  "historical_physical": "/home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_moving_history_physical_20261027_v1_main01/report.json",
  "historical_physical_sha256": "a5322dc4f483e82b5ed9d7b3996cf8a63845261d65774f90a4ae341773016aaf",
  "backend_source": "/home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/p1_moving_live_suffix_cuda_20261027_v1.py",
  "backend_sha256": "b421af2a74bf8af5639f043ae3a06790895f24157880e67b556695b82bb348f6"
}
```

prepare schema `.prepare` 包含 manifest/externalSHA、固定 layers、完整 contract、bindings、
modelbinding、backend_source/backend_sha256。verify 重建整个 prepare 并逐项比对；不运行模型/物理。

run 输出 `.report`：fixed_denominator12、independent_tasks2、固定 rows、actual counts、errors、
model_final_check、prefix_lineage、journal_sha256/chain_tail、两 scope wall。
每 row 保 prefix/source/task 身份、全部 checkpoint、step5/10 编码 receipt、suffix request/dispatch seal、
原生状态标签和 actual_Q5/Q10；缺失保缺失。`summary.native_success` 不是 completed10。
每个 backend namespace 内保存原始数组/预处理张量/receipt；driver 根下保存请求、响应和命令 seal。
startup exception 独立保存 `.startup_failure`，保全部12 not_started，不伪称模型复核成功。

## 命令（MAIN 执行，本文未 prepare/运行）

从 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p1_moving_live_feedback_physical_20261027_v1.py
.venv/bin/python -m experiments.spectral_reachability.p1_moving_live_feedback_physical_20261027_v1 prepare --manifest <absolute-selection.json> --manifest-sha <SHA> --output <NEW-absolute-prepare.json>
.venv/bin/python -m experiments.spectral_reachability.p1_moving_live_feedback_physical_20261027_v1 verify --prepared <absolute-prepare.json> --prepare-sha <SHA>
.venv/bin/python -m experiments.spectral_reachability.p1_moving_live_feedback_physical_20261027_v1 run --prepared <absolute-prepare.json> --prepare-sha <SHA> --output <NEW-absolute-directory> --authorize-cuda-and-physics
```

RAM tests 使用真实 Handler、真实 visual Ledger、冻结 suffix 的 validate_packet 和原 attach_goal；
仅张量/模型模块/环境 integration 为人工数组 mock，绝不算实际 CUDA/物理证据。
覆盖 run→单次 load→同一 model/runtime→12连续分支、I隔离、R自身编码、seal前后时序、
完整预算、constructor隐式reset、earlystop、source漂移、失败无fallback、fsync/operation pending。
experiment-code 技能用于实现/回归结构；未按其通用模板加训练、下载、图片或额外实验。
