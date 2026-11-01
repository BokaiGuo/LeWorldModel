# P2 原27轴共同资源交集：可执行准入内核，而非公平标签

只新增同 stem 源码/tests/本文。P3 cp15 RGB 三件套继续冻结。先完整读了 `innerdocs/agent_baseline_next_gap_20261029_v1.md`：common-resource 六槽搜索、物理、RGB和三份独立summary已有完整实测，不重复runner、汇总器、manifest或根扫描。本轮不打开任何实验结果选cap，不执行旧consume/fullstrict/NN/physics，不训练、不删除GRASP、不改任何旧源或gate。

## 1. 确切缺口与本件作用

已存在的1902 API/sample共同上限与450s共同搜索scope，不等于实际费用相等；composition自然完成较少调用必须保留，不能padding。原数学J请求与192行VJP、GRASP scalar backward、AD primal是不同资源，不能折成同一个“梯度次数”。

旧 `agent_p2_common_resource_grasp_cuda_20261029_v1.resource_vector()` 汇总了资源，但不是原27轴统一 predispatch 准入门：`GAccount` 与 spectral `wall.Control` 用各自原字段；例如 `Native.observe().enter()` 构造 `inputs=[self.snap(x)...]` 后才发 module_attempt，returned也在 `snap(out)` 后发出；`wall.HostBackend.meter_event()` 接已生成事件。因此不能只包一层旧事件sink、反推旧数据或将 module_return 当作独立 returned/captured 认证。

本件实际实现的是纯stdlib `Ledger`：完整unit的27轴上界＋最终双replay预留同时通过才准入；每个typed callback和module阶段按硬门实际扣账并先送durable sink；明确区分模块attempt/dispatched/returned/captured，返回/抓取失败保留真实前缀，不退款。它能调用真实注入callback，但本轮只注入RAM函数，不含任何生产模型/算法外壳。`verify_journal`只复核完整hashchain与资源加法，不能冒充算法/神经/梯度认证。

## 2. 结果盲冻结来源

源码 `PINS` 只读取以下三份明确文件，每份上限2MiB，逐字节SHA核验；不import其模块，不执行foreign code、不读取root/results。旧next_gap只作阅读定位，不把可持续更新的文档当生产依赖。

| 文件（相对 `/home/gbk/code/LeWorldModel/le-wm-main`） | SHA256 / 接点 |
| --- | --- |
| `experiments/spectral_reachability/agent_p2_common_resource_grasp_cuda_20261029_v1.py` | ea763d96eee1040652a3ea02badcb5e8e37b5781530da721f73d2a63d9cb147d；contract/phases/GAccount/Native.observe/resource_vector |
| `experiments/spectral_reachability/agent_p2_compute_matched_protocol_20261029_v2.py` | f17101d1b4b1d8a2c32bcd0218bda9708a2c992c1747a5cfe863e89fb0fff2ee；schedule/unit_cost/Ledger |
| `experiments/spectral_reachability/agent_p2_six_target_controller_cuda_20261029_v1.py` | 075a5c09b8c57051b3e7bc3eb3acac7b824f0c1a983151ec3f8ee10c1c813a5c；原AXES精确27名 |

27名与原AXES由RAM测试AST literal读取逐名相等，不借六target实验的算法/结果。预算数只来自已事前封存1902/450、原joint/composition的42种子/84局部unit、原GRASP532joint＋100sync与6个final forward；没有读胜率、实际wall或Q调数。

## 3. “交集”精确定义和硬预算

对每个root×method**分别**约束 `used[k]+unit_envelope[k]+final_reserve[k] <= common_cap[k]`，全部可计数坐标同时成立；不是取三种方法usage的逐项最小值（那会把J和backward均压到0，实际删掉算法），也不是把未用J兑换predict。FLOPs保持null，不假设一个汇率。

共同cap（原27轴一项不删）：

| 轴 | 每根每方法硬cap |
| --- | ---: |
| predict_API / predictor_samples | 各1902 |
| AE_attempts / AE_dispatched / AE_returned / AE_captured | 各1902 |
| AE_rows / AE_tokens | 1902 / 5706 |
| predictor_attempts / predictor_dispatched / predictor_returned / predictor_captured | 各1902 |
| predictor_context_tokens | 5706 |
| lowproj_attempts / lowproj_dispatched / lowproj_returned / lowproj_captured | 各1902 |
| lowproj_rows | 5706 |
| candidate_cost_calls / candidate_trajectories | 各1902（保守source envelope，不强求耗尽） |
| mean_paid_replays / replay_trajectories | 0 / 2 |
| AD_primal / J / VJP / backward | 1896 / 85 / 16320 / 632 |
| FLOPs | null，不能作为已测cap |

AE原native输入为 `[1,3,10]`，并非只输入一个new packet；predictor两个输入均 `[1,3,192]`；lowproj输入 `[3,192]`。`module(...,shapes=...)`在任何模块attempt或callback前拒绝batch2/其它ABI；未来host接线必须从实际tensor提取，不能把传入的shape字面值当真实CUDA证书。

这三方法输出incumbent/last-complete action，不是CEM mean，故不能偷换 `mean_paid_replays` 的语义：该轴为0，但whole+segmented两条实际重放仍分别付完整3predict，总6计入1902，replay_trajectories=2。score要求显式scope：root_full_path、lifted_loss或serial_terminal。GRASP lifted loss计数不得谎称真实root全路径；候选轨迹数量也不是统一费用单位。

共同side硬cap同时保留CPU/优化器差别：root_SVD2、root_ridge1、Adam_step632、virtual_noise532、Jacobian_elements_requested489600。它们不是擅加进原27名或隐藏成“免费”。CPU capture真实bytes/seconds分别记录；capture interval从capture-intent持久化后开始至callback返回/同步clock，包含该callback及同步，不能推广为纯CPU kernel时间。capture/CPU/SVD及search journal/fsync都包含450s搜索scope；没有独立capture byte硬cap，因此仍需生产output/capture安全合同，不能宣称27轴穷尽实际机器费用。engine_backward_calls/independent_JVP_calls依旧null，不能填0。

固定六槽顺序：root0 joint/composition/grasp，root1 grasp/composition/joint，不按结果改序/删槽。同task/model/context/goal/Q/normalize/raw support的生产身份仍须在新runner复用旧 `join_tasks()` 明确验证；本内核不生成新root或认证任何task。

## 4. 有限schedule和停止算法

`schedule()`忠实保留原label顺序；`envelope()`复算unit工作上界。自然完成计划的predict/API samples依次1902、1062、1902，AD primal255、129、1896，J85、85、0，scalar backward0、0、632；这只是源码算术，不是本轮执行数。**三方法的自然消耗不同不构成错误，也不能改成相同quota掩盖差别。**

1. `Ledger(...,clock=...,sink=...)`要求显式同步clock和durable sink，无生产默认。每臂独立450s，原loader/初始saved验证/最终model hash/完整strict另外计时；搜索含nominal、bank、优化、CPU、capture、journal和最终重放。资源预留不保证wall有余量。
2. `admit(label)`在完整unit边界检查全部坐标，非replay额外预留6forward对应API/sample/三个模块全四阶段/token/row及2replay trajectories。拒绝时不扣未执行工作。不能跳unit、重选alpha或按结果改变GRASP phase。
3. typed `call('J',horizon,dimension)`先扣J及192数学VJP，J元素单列；AD primal只在实际J内predict入口扣，不从J成功反推。GRASP predict在优化unit里扣AD primal，scalar backward单独实际入口扣。不得用632backward或16320数学行称实际相同AD工作。
4. `module()`分别持久化attempt、dispatch，再执行callback；原kernel返回只加returned。之后独立capture-intent/实际capture/hash才加captured。已进入操作失败不退款；durable sink失败后禁止继续，新汇总used=null、local observed值另列，不能当持久化零费用。
5. elapsed>=450不派新操作；已进入kernel可能返回并如实记returned/overrun，禁止额外capture/replay尾段。不强行抢占kernel。返回时越wall的unit不得commit。
6. `commit()`要求该unit完整forward/模块四阶段、J/VJP/AD/backward、replay及side CPU/Adam/noise工作已返回；candidate评分只要求不超上界，不为凑数补评分。empty/partial unit不commit；失败保留actual prefix。这里的commit是**预算单元**完整，不认证模型动作或算法正确性。
7. 首个坐标拒绝/墙钟/operation failure/transport failure停止优化；仅在完整unit边界发生resource拒绝且wall尚余时可进入已预留replay。无replay则计划missing；不能拿last-complete动作或初始零向量直接假装有已付重放。最后完整参数/partial参数的真实数学选择和实际whole/seg等值，仍归未来新runner+fullstrict，内核不发部署许可。

`Journal(path,cap_bytes=...)`是排他、fsync、字节有界单文件测试/接线运输；没有新增生产输出配额或自动扩旧1536MiB上限。`verify_journal()`认证传输hashchain与资源加法，不独立证明调用栈、算法、神经IO、GPU身份、原J方程或wall实测。未来own strict必须按上述typed source序列重建这些，不能把此函数当完整strict替代品。

## 5. 哪些方法还没有公平准入、下一精确接线

| 方法 | 当前具备 | 仍不具备；因此不能声称 |
| --- | --- | --- |
| joint | 原冻结优化/账本/结果；本新硬盒和unit成本 | 新host实际27阶段接线、真实完整梯度/费用重建；不能称等实际费用 |
| composition | 原冻结优化/账本/结果；本新硬盒和unit成本 | 同上；自然较少predict保留，不能padding后称公平 |
| GRASP | 原532+100数学/Adam负零修复/原strict；本新硬盒、backward/noise/Adam门 | 新host分阶段capture接线和新预算strict；不能被排除或称官方750复现 |

三者都不是“禁跑工程”：MAIN可在审定下一最小接线后执行known2校准。它们现在不具备的是更强公平结论，而非被新造科学理由禁止运行。单根费用不同、AD内部未知、hardwareclock不稳定时，最多比较**共同可测资源盒与同scope wall限制下的响应**；actual equality和equal-FLOPs均false。P1/P2科学机制、fresh确认仍无新增证据。

最短下一实现顺序（本轮不扩成新runner）：

1. NEW host backend继承原输入/模型source闭包，针对 `wall.HostBackend.meter_event/cpu` 与 `common.Native.observe/step/grasp_optimize` 增加真正pre-snap预算入口，把returned和captured拆开；不得只接旧event sink或patch模块globals。原三算法及source形状/精度/动作support不变。新backend用本件Ledger调用，所有clock采样在相同CUDA同步口径。
2. 原spectral bank/jacobian与GRASP scalar backward/Adam/noise绑定typed入口，SVD/ridge进入side门；保留原完整source task和全部model IO/gradient/Adam witness。source callback可执行但预算账不认证其中hidden神经工作，实际host hooks是硬前置。
3. 新own consumer重建原27阶段、side、wall/pending/last-complete/最终双重放、模型beforeafter；复用已pin纯Adam算术，不从旧results回填27轴。数组/metadata输出预算另外source-first证明，MAIN批准后才生产。不得调用旧consume后将结果换schema认证为本件运行。

## 6. 现在可执行的 MAIN 命令

仅源码和RAM/临时盘，不读取生产结果、不跑NN或fullstrict。从 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_common_resource_intersection_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_common_resource_intersection_20261029_v1 verify-sources
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_common_resource_intersection_20261029_v1 spec
```

没有生产prepare/run CLI，不伪称可用NN三臂runner；本件是可直接接入typed callbacks的硬预算内核，而非新blocked manifest。测试包含全部六槽临时盘完整unit→hashchain重算、原27名、三有限schedule上界、逐轴拒绝、非通兑AD门、reserve replay实际付费、failed capture与returned分开、450秒精确边界及inflight overrun、durable写失败禁止callback、batch2拒绝、empty unit禁止commit和链篡改。本件按experiment-code技能做实现/失败回归，不采其泛用训练或结果调参建议。
