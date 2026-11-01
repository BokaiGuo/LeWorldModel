# P3 在线 fixed/signal：未来执行接线与拒绝测试

## 已有资产与新增接线

现有 `p3_matched_order_feedback_host_v2_codex_20261029` 已实现 fixed/signal 四槽在线仿真、真实观察反馈、来源约束、分槽失败及保存；其独立 saved consumer 核完整张量、动作、观察与资源事件。已有预算合同 consumer 检查已计量向量、shape 与原生指标。因此本次不改旧 runner、不再写一套模型或环境执行器，而是新增 `p3_feedback_online_binding_20261030_v1.py`，将原 host 与事前预算合同接成以下链：

1. `plan`：CPU 重建合同和原 prepare/源码闭包，固定未来 run、strict receipt、分析三条直接位于 results 的唯一新路径，保存接线源码 SHA、合同 SHA、prepare SHA 和四槽。路径预检不是文件系统预留；输出仍要求排他创建。
2. `run`：先核计划 SHA、接线源码、合同与原 prepare，检查三条路径未占用、固定显式授权令牌和操作者确认、非空外部协调引用；之后调用**原** `host.run`。传入 host 的 coordination 包含计划 SHA、外部记录引用和“操作者声明、未验证”标签。
3. `consume`：核输出 report/completion SHA、其 coordination 中精确的计划 SHA 和引用、原 prepare，再调用**原**严格 saved consumer，排他保存其完整或 partial receipt；不重试运行。
4. `inspect`：核指定 receipt SHA 与同一 run-plan 绑定，调用既有预算合同 `inspect`；输出仍标记外部时间戳未验证、`scientific_pass=false`。partial receipt 保留分母和 unknown，不能升级为 complete。

原 lifecycle 的单槽失败后继续后续固定槽语义不变；任何 setup/slot/cleanup/停止都按原 runner 保留，不补槽。原 runner 若在创建 store 前失败，或文件系统无法写入终态，可能没有 report/completion；不能凭本接线补造 receipt。`consume` 已成功保存 receipt 而后续分析失败时，可直接用 receipt 的真实 SHA 再调用 `inspect`；不能重跑 `consume` 覆盖它。

## 本次唯一计划与授权边界

CPU-only 实际来源重建通过，已独占创建 [计划文件](../le-wm-main/results/p3_feedback_online_binding_20261030_v1_plan_ONLINE_BIND01.json)，SHA256：

`4820933cd4e64e3dc56073eb018212527f463088801de9a3c0e5a27ec1687802`

它绑定原 prepare SHA `b6c8dcf650d4e752f6306ed4b6f9434861c224ced51c7c8de25c0e7ca6d323f2` 与预算合同 SHA `5781d953d05477e633659b3e9bb58d630ea59447720964af4e8678829f565b4f`。计划路径当前未被占用并非锁定；启动前和原 runner 仍会复查。本计划仅针对未来运行，不给已完成的 RUN_MAIN01 倒签时间。

**本次未启动 GPU、模型或仿真，也未获得未来运行授权。** 真正运行前，负责人需独立审读完整合同和本计划，在外部可信不可变记录中将两份文件及 SHA 固定并证明先于运行；另外完成 GPU、环境、渲染/显示、时间窗口、存储及失败处理的真实协调。`--reference` 只保存引用，`--acknowledgment` 和 `--authorization` 只是人为声明／程序开关，不验证批准人、时间戳、资源租约或外部记录真实性；即使调用成功，`external_prerun_timestamp_verified=false`。

原计划的两个根均已知，不是 fresh。共同的完整计量向量、shape 多重集与成功的原严格消费，均不等于 FLOPs、wall、能源或全部未监测计算相同，也不证明科学通过。数值 ledger 的 wall 可能包含期间交错的物理和编码工作，不能解释为纯模型计算时长；完整单槽 wall、GPU 时间、FLOPs 保留缺失。

## 未来运行顺序（仅取得独立协调后）

在 `/home/gbk/code/LeWorldModel/le-wm-main` 中，以真实独立记录引用替换 `REPLACE_WITH_REAL_RECORD`。先重新确认路径没有被占用、计划及合同哈希和源码闭包不漂移、GPU 与仿真资源获批准。任何漂移须新版本重新冻结和独立时间戳；不得修改既有封存文件。以下 run 会实际使用 GPU 和 PushT 环境，本次**没有执行**：

```sh
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m experiments.spectral_reachability.p3_feedback_online_binding_20261030_v1 run \
  --plan results/p3_feedback_online_binding_20261030_v1_plan_ONLINE_BIND01.json \
  --plan-sha 4820933cd4e64e3dc56073eb018212527f463088801de9a3c0e5a27ec1687802 \
  --reference REPLACE_WITH_REAL_RECORD \
  --authorization COORDINATED_MATCHED4_HOST_CUDA_PHYSICS \
  --acknowledgment I_CONFIRMED_EXTERNAL_PRERUN_RECORD_AND_RESOURCE_COORDINATION
```

保留实际输出的 report SHA、completion SHA、退出状态与全部产物。存在这两个文件后，另起 CPU 进程：

```sh
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m experiments.spectral_reachability.p3_feedback_online_binding_20261030_v1 consume \
  --plan results/p3_feedback_online_binding_20261030_v1_plan_ONLINE_BIND01.json \
  --plan-sha 4820933cd4e64e3dc56073eb018212527f463088801de9a3c0e5a27ec1687802 \
  --reference REPLACE_WITH_REAL_RECORD --report-sha ACTUAL_REPORT_SHA --completion-sha ACTUAL_COMPLETION_SHA
```

再用上步打印的 `ACTUAL_RECEIPT_SHA`：

```sh
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m experiments.spectral_reachability.p3_feedback_online_binding_20261030_v1 inspect \
  --plan results/p3_feedback_online_binding_20261030_v1_plan_ONLINE_BIND01.json \
  --plan-sha 4820933cd4e64e3dc56073eb018212527f463088801de9a3c0e5a27ec1687802 \
  --reference REPLACE_WITH_REAL_RECORD --receipt-sha ACTUAL_RECEIPT_SHA
```

使用相同引用；未知 SHA 不能预填。成功分析的 `complete_saved_measured_work_only` 表示输入可作保存分析，不意味着 shape 必然匹配；分别看 `measured_vector_match` 和 `hook_shape_multiset_match`。不以假回调、CLI 退出码或完整候选 report 替代 strict receipt。

## 验证

CPU、`CUDA_VISIBLE_DEVICES=''`：新接线测试、预算合同测试、原 host 和联合故障测试共 **27 passed**。新测试用替身验证端到端参数接线、未授权和空确认不派发、路径占用与 SHA/源码漂移拒绝、错误引用与旧运行冒充拒绝、partial receipt 不升级及不重试、输出不可覆盖；原联合故障测试覆盖真实保存 consumer 对动作位串、根、提前停止、观测帧与峰值超限注入的拒绝。替身和合成事件只是软件验证，不是实际在线运行或科学证据。
