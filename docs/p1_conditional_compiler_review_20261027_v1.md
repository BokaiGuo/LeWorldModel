# Conditional full10 / fixed-H2 compiler 独立只读审查

日期：2026-10-27。结论：**Q 代数链与声明的 66/11/12 预算未发现接线错误；失败留证与保存数组验证仍有可复现实错，暂不建议据此签发完整执行链通过结论。没有验证真实模型或物理效果。**

本次仅新增本文；未改实现、测试、共享文档、既有封存输入或结果；未另派代理，未 prepare 新真实协议，未运行神经模型、CUDA AD 或物理。审查采用 code-review 的实现/合同分轴方式，按任务要求不使用其并行代理步骤。以下行号对应本次快照，不代表后续并发修改版本。

## 1. 快照及合同边界

路径基准：`/home/gbk/code/LeWorldModel/`。

| 文件 | SHA256 |
|---|---|
| `le-wm-main/experiments/spectral_reachability/p3_goal_patch_consistent_compiler_20261027_v1.py` | `006398e5aaf0f6e901a01c2d6779a977c31875413b967fec552245332002396d` |
| `le-wm-main/tests/test_p3_goal_patch_consistent_compiler_20261027_v1.py`（最终复核版） | `356f5c77308e676b00d2b0be8a6c2f266ea7a64b1c684f50a8b037b417957e09` |
| `le-wm-main/experiments/spectral_reachability/pusht_full_packet_compiler_20261027.py` | `f359e7b21d1970f95f065dbee206d2dc9359c2eaff7ddbcd7611e07686c6cf2a` |
| `le-wm-main/experiments/spectral_reachability/p3p4_goal_patch_metric_20261027_v1.py` | `f40fca0ccdf5f8f51d68174a7feca21192031816163ba2cc9bce508febb18957` |

审查中实现 SHA 未变；测试由初始 `ac8c85ca19ce5eb2b8ba64b7dc967e044b216566db9ae03e8987457c36b837c4` 增加两项测试；innerdoc 从暂缺变为存在，已重新完整阅读。**最初缺 innerdoc 不列为最终缺陷。**

交付前 main 通知已冻结 ready、正在 verify、拟运行 host CUDA；本次重新读取字节确认 source SHA 如上，`le-wm-main/results/p3_goal_patch_consistent_compiler_20261027_v1_prepared/prepare.json` SHA 为 `2b32137d1e7facf79335b25c6e6acf976ed10162073b938bff1262b385ba1425`，与通知一致。本次没有执行该新 prepare 的完整 verify，也未替 main 启动/停止任何任务。R1/R2 针对该冻结源码仍成立，已优先回报；建议在要求完整失败留证的执行合同下先处理，再重新封存，不能更改冻结源后继续使用原 prepare。

外层 `docs/p3_goal_patch_consistent_compiler_spec_20261027_v1.md` 是四臂 metric×schedule、joint H2/H3 root-ridge+随机尾部方案。当前实现 `contract():29–50` 和新 innerdoc 明确选择两臂、fixed H2、per-prefix conditional suffix，继承 full-packet compiler 的 stage+own-reach selection。本文按用户指定的 conditional / 66-per-arm 合同审查，不将两套不同方案混账；当前实现**不能称已实现四臂 P3 dynamic 或该外层 spec 的完整方案**。innerdoc 关于 main 已选路线的文字是作者声明，本次未独立核验对话授权。

## 2. 可重现实错（按执行风险排序）

### R1 [P1] 收尾 I/O 或 hook-cleanup 异常会丢失首错/报告

位置：新实现 `execute_cuda():355–374,389–401`；继承的 `p3_horizon_residual_budget_20261027_v4.py:123–130` 为 hook install/restore。

两种独立触发已用现有 `fake_host(..., broken=True)` 复现，没有真实 torch/model：

1. 先发生 `fixture kernel failure`；随后让 `checked.digest` 仅在 `.events.jsonl` 路径抛 `OSError('fixture journal hash failure')`。`execute_cuda` 在外层 except 后的第 373 行直接再次 hash journal，没有保护。实际对外异常变成该 OSError，`report.json` 不存在；原 kernel error 仅留在 journal。全八槽、attempt/completed 与首错未能组成最终失败报告。即使第 366 行 hash 的异常先被 except 捕获，第 373 行也会再次触发。
2. 同样先发生 kernel error，再令 `NeuralCap.__exit__` 在原 restore 后抛 `RuntimeError('fixture hook cleanup failure')`。实际 report 的 `failure` 与唯一 `failures` 条目均为 cleanup error，原 kernel error 未被收集进去。Python exception context 或 journal 可能仍有首错，但报告没有保存 traceback/context 链，不能称首错和 cleanup error 均已保留。

这不是要求磁盘彻底耗尽后仍保证写盘；**可读 journal 的单次 hash/cleanup 失败也能触发**，在当前小磁盘余量下尤其不宜忽略。第 375–378 行正确避免 SourceDrift 覆盖已有 `failure`，但保护范围未覆盖其他收尾步骤。

建议验收：保留首异常及 traceback/exception chain；hook restore、model-state 检查、journal hash/read/decode 和 root-record 写入分别捕获并追加 secondary errors；在可写条件下尽力形成八槽 incomplete report，无法核验的成本显式 unknown，禁止自动重试。应补上上述两项 fault-injection 测试，不能只检查正常 kernel failure 路径。

### R2 [P1] nominal/J 的 durable 保存晚于可能失败的后续操作

位置：新实现 `compile_arm():115–117,124–127` 与 `TorchBackend.nominal_jacobian():181–188`。

根上先取得 `(nominal,J)`，再做两个 bank SVD，成功后才在第 117 行把数组挂入 journal。将第一次 `np.linalg.svd` mock 为抛错后，实际 journal 显示一个 **completed Jacobian**，但没有 `root_jacobian` 或 `root_nominal`。如果真实 AD 已成功、随后 NumPy SVD 出错，已付费得到的 J 不能从 artifacts 复核。

更早的窗口是 backend 在第 186 行成功获得 nominal、第 187 行 AD 失败：nominal 从未交给 durable journal。suffix 上也如此；其 prefix 已在第 124 行执行成功，但 prefix latent、suffix nominal/J 要等第 125 行整体返回后才写。此时发生 AD 失败，会有付费预测/AD intent，却没有已完成 prefix/nominal 数组。

这不等于数字账必然漏计：predict hooks 和 AD intent 仍能记录 attempted/completed。错误是 innerdoc 所称“随阶段写入”“所有 partial 保留”在上述确定窗口不成立。

建议验收：nominal、J、prefix 完成后各自立即保存，再进入下一可能失败步骤；backend 接口应允许在 AD 前 durable 保存 nominal。补测 nominal 后 AD 失败、J 后首/次 SVD 失败、prefix 后 suffix AD 失败，并断言数组与成本状态均可恢复。

### R3 [P2] `validate_compilation` 未验证 action history 的实际内容

位置：新实现 `validate_compilation():316–319`，以及 `selected_actions()` 对保存 context/journal 的相互一致性检查。

该 verifier 对 `context['action_history']` 只做有限值和 `[2,10]` 形状校验；没有核对它是否等于本候选两段 commands 经封存 mean/std 归一化得到的 tokens。与此相对，latent history 的三帧 chronology 有逐值检查。

实际复现：正常 algebra compile 后，选一个非 winner 候选，只把 `c['context']['action_history'][0][0]` 改为 `12345.`，其余动作、J、goal、score、latent/winner/replay 均不改；`validate_compilation(p,Q,r)` **仍返回成功**。其 saved-array 完整 context 核验承诺因而不成立。还未对 context layout 元数据作同等级校验。

界限：运行时第 136/154 行确实比较整个 context，可拒绝实际 replay history 的变化；exporter 也绑定 journal SHA 和重复保存的 context。**本发现不声称能仅改一个文件就绕过 exporter 的所有哈希**，也不是神经模型 replay 已知错误。它表明保存数组 verifier 不能独立识别所有候选的 action chronology 错误；多个 receipt 若一致保存错误 history，现有代数审计不足以发现。

建议验收：使用纯数组、封存 public mean/std，重建两段 time-major normalized tokens，与每候选 history 严格比较；校验 layout/context_length/latent_dim/action_dim。不需新神经调用或物理。补测非 winner history、winner history 连同重复 receipts、layout 元数据污染。

## 3. 核心数学与执行接线核对

下列为静态追踪和纯 mock/数组测试结论，不是 host CUDA 认证。

| 关注项 | 核对结果与边界 |
|---|---|
| Q → prefix spectral bank | `make_bank():65–67` 传当前 arm Q 给原 `spectral_bank`；原 helper 用 `Q@J` 做谱、用 `Q@(goal−nominal)` 做 root ridge。新旧 Q 生成不同动作的 mock 通过，不只是重评分旧 pool。 |
| Q → own-conditional suffix | 第 124–128 行在各 prefix 的 predicted 完整 context 上 fresh nominal/J，求 `bounded_ridge(Q@j,Q@(goal−n))`。不是复用根 J，也不是其他 prefix 的 conditional solve。 |
| Jacobian 定义 | `TorchBackend:184–187` 对 `terminal(tanh(u))` 在零 logits 求 `[192,10]` raw native terminal J；J 本身不预乘 Q，metric 在其后施加，未发现漏 Q 或重复 Q。prefix 固定为条件，非 joint H2 端到端 Jacobian。 |
| score / objective | `cost():60–62` 为 `/192`，stage、terminal diagnostic 同 Q/goal0；选优用 `stage+own_reach_value`，stable index tie。原 `compile_arm` 同此目标。**不是 nonlinear terminal argmin**，重放验证非线性输出不等于据非线性终点优化选优。 |
| 正则尺度 | 原 ridge 最小化未除以 192 的 `||Kd−r||²+.002||d||²`，仅 receipt 的 cost/penalty 除以 192；未发现 Gram/rhs 除 192 而 damping 不变的陷阱。 |
| logits / raw | 每 packet 的 cap 为 `.75*sqrt(5)`，RADII 为 `.25*sqrt(5)` 与 cap；先 float32 logits，再 NumPy tanh，再 `[5,2]` time-major。无额外 `/100`、raw clipping 或二次 tanh。是两个独立 packet-ball，不是 joint-H ball。 |
| 与旧 compiler 的 tanh 差异 | 旧版在 CUDA torch 中 tanh，新版在 CPU NumPy 中 tanh 后把 raw 放到 CUDA；数值逐 bit 等价不能据此推断。新版独立 replay 使用其同一 raw commands，innerdoc 已披露此差异，不列作未知实错。 |
| 真实 action normalization | `compiler.make_transition:95–102` 与 `CudaReplay` 均调用 `normalize_live_packet`，归一化 `(flatten(raw)−mean)/std`，per-command stats 按 packet 顺序 repeat；未发现尺度接反。现有 LinearBackend 用 raw 假 history，并未实测真实 normalization。 |
| history / replay | 起始 `[3,192]` states、`[2,10]` between-actions；transition append token 后滚动为三 states/两 actions。所有候选独立 H2 replay，winner whole+segmented，canonical exact 比较整个 context，不是 endpoint allclose。真实 CUDA 下是否逐 bit 一致仍未验证；保存审计的 history 漏检见 R3。 |
| 192 维 projected CLS | `metric.compile_metric` 的原 goal-only object/nuisance projector 与原 goal0、原 192 维 context 分别封存继承；当前模块不取 patch-token mean、不训练 adapter、不编码 endpoint。shape 本身不是语义证明，语义来自冻结来源和模型路径；本次未重跑 encoder。 |
| 输入/active metric 身份 | `source_inputs:209–245` 按 root identity join 四根，核验原 goal0/order/model，分别保留 source input SHA、两 arm active metric IDs 和新 input SHA；Q 与 array-audit `.npy` 有 hash/数值双绑定。无旧 winner/candidate 导入新搜索。 |
| 封存失效 | `validate_prepared:266–278` 重新核验 contract/source_inputs/current source bytes/seals；execute 前后调用；source drift 会导致 incomplete，export 也重新 validate。fixture source drift 测试与真实 `source_inputs(PROJECT)` 只读校验通过。未创建真实新 prepare，不认证某个新 prepare 的可运行性。 |

`metric.compile_action_response` 是 `[192,10H]` joint response 的数组桥；本实现没有调用它，而是调用原 full-packet H1 helper 11 次，符合其自述 conditional 分解。不能拿另一路 H2/H3 或 FD JVP 数值替代这 88 个 fresh conditional Jacobian 的真实认证。

## 4. 预算和失败计账

| 项目 | 每 arm | 每 root 两 arm | 四 root 八 arm |
|---|---:|---:|---:|
| nominal / AD-forward（根） | 2 | 4 | 16 |
| 十候选：prefix + suffix nominal + AD-forward + optimized suffix + 独立 H2 replay | 60 | 120 | 480 |
| winner H2 whole + 两个 segmented packets | 4 | 8 | 32 |
| predict attempts / action_encoder attempts（分别计） | 66 / 66 | 132 / 132 | 528 / 528 |
| fresh J materializations / jacobian API | 11 | 22 | 88 |
| SVD：bank 2 + suffix 10 | 12 | 24 | 96 |
| 数学 scalar-output VJP 方向 | 2112 | 4224 | 16896 |
| planner visual encoder | 0 | 0 | 0 |

Mock 完成路径实得上述计数，未发现 66/11/12 的 off-by-one。`with_svd_receipts` 包裹实际 NumPy SVD 调用，失败后 finally 恢复，不是仅按公式虚报两次。每 candidate、含 zero/重复候选，仍执行并付费。预算只限制 predict 和 action_encoder 各自样本数，非 FLOPs；AD forward 已计 predict，backward engine 调用数未知。

`NeuralCap.dispatch:193–206` 在 callback 前加 event，predict/action_encoder 各自 cap，异常不退款；66 后的下一次在 dispatch 前拒绝。正常 kernel-error fixture 保留 incomplete+7 blocked slots，并拒绝 export。R1/R2 是这些保证在二次失败/数组保存窗口上的实际缺口，不能被上述正常路径测试覆盖。

继承 metric 构造 80 encoder、80 projector、8 SVD 与旧 context 成本单列；export 重建 saved-array bank/solves 额外 96 SVD，不计新神经 forward。新旧 Q 不同，不直接相减 scalar score 当外部效用改善。独立新根为零，不能由 matched 四 calibration roots 推出泛化/物理成功。

## 5. 可复现验证

以下从 `/home/gbk/code/LeWorldModel/le-wm-main` 执行；禁 bytecode/cache，不加载神经模型。真实 sealed-input 检查仅读数组与哈希。最初同 stem 23 tests 通过；并发新增两测试后，最终三个文件联合重跑 **52 passed in 5.78s**（同 stem 25 项、两个参照文件合计 27 项）。与 main 通知的 29 joint tests 不必视为相同测试集合。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p3_goal_patch_consistent_compiler_20261027_v1.py tests/test_pusht_full_packet_compiler_20261027.py tests/test_p3p4_goal_patch_metric_20261027_v1.py
```

复现 R2/R3 的最小内存脚本（以同一环境前缀运行 `.venv/bin/python -`；无需新增源码文件）：

```python
import runpy
import numpy as np
n = runpy.run_path('tests/test_p3_goal_patch_consistent_compiler_20261027_v1.py')
m = n['m']; p = n['packet'](); q = np.eye(192)
r = m.compile_arm(p, q, n['LinearBackend'](), lambda e: None)
c = next(c for c in r['candidates'] if c['index'] != r['winner']['index'])
c['context']['action_history'][0][0] = 12345.
m.validate_compilation(p, q, r)  # 实际未抛错：R3

events = []; original = np.linalg.svd
def fail(*a, **k): raise RuntimeError('SVD fixture')
try:
    np.linalg.svd = fail
    try: m.compile_arm(p, q, n['LinearBackend'](), events.append)
    except RuntimeError: pass
finally:
    np.linalg.svd = original
assert any(e['kind'] == 'jacobian' and e['completed'] for e in events)
assert not any('root_jacobian' in e or 'root_nominal' in e for e in events)
```

R1 可复现 fault-injection 核心如下。`fake_host` 中 torch/CUDA/model/backend 都是 SimpleNamespace/代数 stand-in；临时目录使用 `/tmp`，本次结束后自动清除自身临时目录。

```python
import runpy, sys, tempfile
from pathlib import Path
import pytest
n = runpy.run_path('tests/test_p3_goal_patch_consistent_compiler_20261027_v1.py')
m = n['m']
for mode in ('hash', 'cleanup'):
    with tempfile.TemporaryDirectory(prefix='p1-review-', dir='/tmp') as d:
        with pytest.MonkeyPatch.context() as patch:
            root = Path(d); n['fake_host'](patch, root, broken=True)
            checked = sys.modules['experiments.spectral_reachability.run_pusht_d2p_budget_guard_cuda']
            if mode == 'hash':
                original_digest = checked.digest
                def fail_hash(path):
                    if str(path).endswith('.events.jsonl'):
                        raise OSError('fixture journal hash failure')
                    return original_digest(path)
                patch.setattr(checked, 'digest', fail_hash)
            else:
                original_exit = m.NeuralCap.__exit__
                def fail_cleanup(self, *exc):
                    original_exit(self, *exc)
                    raise RuntimeError('fixture hook cleanup failure')
                patch.setattr(m.NeuralCap, '__exit__', fail_cleanup)
            out = root/'execution'
            try: m.execute_cuda(root/'prepared', 'sha', out)
            except Exception as exc: print(mode, type(exc).__name__, str(exc))
            print('report exists:', (out/'report.json').exists())
            if (out/'report.json').exists():
                print(m.io.decode((out/'report.json').read_bytes())['failures'])
```

本次实际输出：hash 分支抛 `OSError fixture journal hash failure`、report 不存在；cleanup 分支 report 存在，但 failures 只有 `fixture hook cleanup failure`。另以 AD-failure backend 复现了 nominal 已完成但无数组保存，只留下 Jacobian intent/error。

## 6. 总结与准入边界

实现/协议实错：R1、R2 为 P1，R3 为 P2；没有报告未经复现的 Q/尺度错误。纯测试通过不撤销这些额外 fault-injection 结果。源码风格没有另列可维护性建议，以免掩盖本次要求的执行实错。

可支持的结论仅为：当前代码结构上将 active Q 接入 fresh prefix bank、各自 conditional suffix、stage/own-reach/terminal，并有 model-free 匹配账；保存证据和异常收尾仍需修补与回归。没有确认该 compiler 的真实 CUDA AD、exact replay、实际行为质量或新 goal-patch Q 的方法成功；P1/P3/P4 admission 均应保持 false。
