# Observed RGB history producer v1 — MAIN-only physics

新文件：`experiments/spectral_reachability/p1_observed_history_source_20261027_v1.py`
与同 stem tests。没有修改旧 generator/metadata/direct/diagnostic 或其证据。
本实现线程只执行 NumPy/fake/metadata 检查，不执行真实 reset/render/physics/GPU。

## 固定来源、重建、动作

prepare 绑定 calibration2 原 request 文件 SHA、sourcepack 文件 SHA、hash-chain
journal、两份选中 NPZ、原 source closure，以及已实际通过的 reset_goal_audit report
文件 SHA（必须 diagnostic_completed/all_reproduction_checks=true，无主/次错误，且同源）。
还绑定 producer/test/diagnostic/normalizer 的源码 SHA。run 前重构整个 prepare；run 后
再次逐文件核验 bindings。没有重新扫描 exclusions，没有 outcome 调参或重新选根。

仅原选中两根，顺序固定：parent root seeds 202761010/011；原选中
proposal/attempt 分别 0/1、0/0，即 random reset seeds 202781011、202781014。
每根先 random reset，再从其返回的原 goal 和当前 native polygons 重新运行原
`sample_state(goal, slot, proposal, attempt, plan)`，以 placement seed=0 和重新生成的
requested state/goal reset。**saved poststate 从不作 reset 输入**，仅作精确重建比较。
原两次 reset receipts、selected state/goal/RGB/实际 goalRGB 必须精确重现，否则失败。
保持 goalRGB 为原 placement reset 内实际捕获的同一帧，不重新绘制理想 goal。

随后固定执行10个 float32 raw `[0,0]` public actions（两包，每包5步）；实际 RGB
分别来自 at0/5/10 的显式 render。零动作不是零物理；relative PushT 下零命令也可产生
制动/接触运动。每一 public step 都记录返回 state、terminal/truncated 和 goal；
不读取 reward 来选择根。bootstrap terminal flags 是源构建安全信息，不是 downstream
future task label。任何 terminal/truncated 立即停该根，禁止继续、补采或替换。

at0/at10 沿用原预声明 source domain（canvas/contact/nonterminal），另检查 native
`eval_state` 非终止；其他步检查 canvas/native 非终止。无新科学准入阈值。
原根失败仍保留 denominator 两根；安全拒绝后可继续另一个预声明原根，技术异常则中止
余下未启动根。不能为了凑满20步越过终止继续执行。

## 预算、durability、失败

成功总计：4 reset、20 public step、6 explicit render；reset 内部8 `_set_state`
和4 goal render。已封存 PushT dt=.01/control_hz=10：每 public step 10个内部 physics
steps，每 reset两个，所以完整运行208内部 physics steps（8 reset内 + 200 warmup内）。
通过实例级临时 hooks 实际记录 space.step attempted/completed，不仅按完整调用推算；
reset 产生新 space 时重新安装 hook，所有 hook 均恢复。无全局猴补、无改变 physics 参数。
失败可能完成少于以上预算，或在未完成 reset/step 中已完成部分 physics 子步；这些实际
计数照实保留，不能写成固定208。successful exact-budget gate 检查全部七类资源。

每次外部调用前/后均写 hash-chain journal 并 flush/fsync。每次 at0/5/10 frame
另立即写 `checkpoint_{cid}_{step}.npz`（RGB/state/goal/goalRGB/hidden-velocity snapshot），
fsync 后再继续 public step；不只持有内存 RGB/hash。checkpoint 不是可准入来源。
成功输出 `source_{cid}.npz`；失败已观察帧输出独立 `partial_{cid}.npz` schema，loader
拒绝 partial。partial raw_action_history 仅保存实际完成的 n 步 `(n,2)`，normalized
history 仅含已完成的五步整包，不把未执行的 zero suffix 冒充观察。逐步 state/flags、
失败根与未启动根留在 journal/sourcepack。
first error 不被 cleanup/hash/report errors 覆盖；secondary errors 单列。
close、hook restore、final binding、journal hash/report 尽力完成。磁盘不可写时不能承诺
最后报告一定存在：stdout 返回失败，已落盘 journal/checkpoints 仍是部分证据，不可冒充完成。

准备/运行需200MiB+8MiB可用空间；总输出硬上限8MiB，写前检查200MiB floor。
无覆盖、resume、下载、删除。若已有输出名，必须另选 NEW 路径。

## 独立桥 schema（不是旧 causal-padding 合同）

sourcepack schema 为 `p1_observed_history_source_20261027_v1.sourcepack`；成功状态
`observed_calibration_source_completed`。consumer 必须先绑定 sourcepack/prepare SHA，
确认无 first/secondary errors、durable journal terminal、全部 rows completed、counts
精确成功预算，再验证每个 file SHA，调用 `load_observed(path, prepared)`。
不能从失败 sourcepack 单取某个 complete NPZ 冒充整个来源池成功。

NPZ 禁止 pickle，包含：

| 字段 | 含义 |
|---|---|
| metadata_json | 新 schema/prepare_id/parent source+attempt/新 root SHA/calibration-only |
| rgb_history | uint8 `(3,224,224,3)`，真正 at0/5/10，不要求图像一定不同 |
| observed_states | float `(3,7)`，同三时刻状态 |
| observed_public_step_indices | `[0,5,10]` |
| state / goal | **最终 at10 observed state** / 原 requested goal |
| goal_rgb | 原 placement reset 的实际同帧目标RGB，精确数组比较 |
| raw_action_history | float32 `(10,2)`，十个已实际执行的zero commands，非padding |
| normalized_action_history | float32 `(2,10)`，两条 between-state tokens，非零假设 |

新 root SHA 为 float64 contiguous concatenate(final state7, original goal7) 的 SHA256；
parent SHA 保留但不替代新 root identity。normalizer 仅读取/绑定已审计 public constants
AST，`training_provenance_verified=false`；不编造训练 provenance。

此文件不调用任何 encoder。未来 MAIN 新桥可将三张实际 RGB 和 goalRGB 各编码为
native projected CLS192，以 encoded at0/5/10 组成 root_context latent_history，以
这里两条 normalized tokens 组成 action_history，再从最终 at10 开始 direct screening。
raw actions 同时保留，桥应精确重算 normalization。旧 `screen.encode_source` 的
legacy loader/schema **不会直接接受此 NPZ**；禁止伪造旧 plan/history 字段绕过。
global freshness 始终 unknown/calibration-only；warmup 后根不因新SHA变成fresh confirmation。
这只填真实 observed-history 来源缺口，不提供 Q/goal任务准入、plateau、two-stage witness、
confirmation 阈值或真实效能证明。bootstrap信息与 privileged state 只用于来源校验/计账，
不能作为未来 neural planner 输入。

## MAIN prepare/run

在 `/home/gbk/code/LeWorldModel/le-wm-main`；第一条仅 metadata prepare，不构造环境。
三个外部输入 SHA 已从真实文件读取，非内部 plan SHA：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p1_observed_history_source_20261027_v1 --request results/p1_conditional_calibration2_main_20261027_v1_request.json --request-sha256 a53a6671e67eb6a79d0ccb6614bf6d5dfb4b6417ec3eb3b548ab63a41fb67d68 --report results/p1_conditional_calibration2_main_20261027_v1/sourcepack.json --report-sha256 728f14cb83c4fcd0ea4f5348bad2bff5b5dd372ab0185b6e6b3a26dbfda58057 --reset-audit-report results/p1_source_reset_goal_audit_20261027_v1_main01/report.json --reset-audit-sha256 2f6b7c4ad8b54a3157f1ac2f514698b96a5cce1550a6c81a4f57d206df7ebe24 --prepare results/p1_observed_history_source_20261027_v1_prepare01.json
sha256sum results/p1_observed_history_source_20261027_v1_prepare01.json
```

仅 MAIN 授权执行（**真实 physics**；`PREPARE_FILE_SHA` 用上条输出替换）：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p1_observed_history_source_20261027_v1 --run-main results/p1_observed_history_source_20261027_v1_prepare01.json --prepare-sha256 PREPARE_FILE_SHA --authorize-physics --output results/p1_observed_history_source_20261027_v1_main01
```

mock/metadata 测试：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p1_observed_history_source_20261027_v1.py
```

测试覆盖：真实输入 metadata seal，无 native imports；fake完整208子步、原 recipe 与
poststate区别、actual RGB/checkpoint/history、terminal at0/途中、末根非法、部分失败physics、
primary/secondary errors、source hash漂移、授权/空间/独占输出及八种 loader 篡改。
测试通过不代表实际 producer 已成功，actual terminal 必须由 MAIN 运行后另审。
