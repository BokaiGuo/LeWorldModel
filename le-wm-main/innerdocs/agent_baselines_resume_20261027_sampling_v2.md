# 强基线可消费证据：真实 receipt adapter v2

本工作位承担实现，不派生后代，不接 MAIN 的探针方向预测任务。个人有界完成条件为新 adapter、证据矩阵、RAM 测试及真实 saved 集成输出齐全；不是 P1 或研究总目标完成。采用 experiment-code 的固定输入/失败边界和 citation-check 的来源/主张分离；按用户范围不训练、不画新实验图、不联网核 DOI、不执行神经或物理。所有旧封存文件保持不变。

## 实际修复的不可消费接口

原 `moving_prediction_matched_sampling_comparison_20261027_v1` 排除了 actual receipts 中的 `environment_close`，另找 `cleanup_intent/cleanup_returned`。真实 physical v1 的 close 与其他操作一样使用 `operation_dispatch_intent/operation_returned/operation_failed`，resource、operation_index、status 均在 receipt 内。因此旧 comparison 的 RAM 测试通过并不意味着它能消费最终真实 producer。

新独占 `agent_baselines_resume_20261027_sampling_v2.py` 从旧 comparison 分叉保留来源、选取、命令、PNG、真实7维指标及 reference 合同，**替换** operation receipt 状态机，而非把真实 journal 改装成旧事件。检查连续整数 operation_index、pending→returned/failed 状态、严格嵌套 LIFO、dispatch/return 身份全字段一致、全部 actual receipts 按 index 与 journal 终态一致、各资源真实 attempted/completed 和 cap。close 纳入同一真实账。旧 cleanup ABI 明确拒绝；partial 的已返回/失败操作可以保留，无法由完整 journal 验证的 pending 不发布成可信比较。GN/D2P 旧 reference 仍按其真实 legacy 合同读取，单独固定原 helper SHA，不改旧 parser/global。

固定两根×两臂通过 root/goal/blind/task/selection 身份 join，不靠结果数组位置。所有已执行 warmup/plan raw 命令、88 PNG、guard、state/snapshot、checkpoint、原目标、物理选择冻结及完整分母保持原严格检查。planner 仅调用其 saved `consume`，不调用 build/run/optimize；RGBv2 仅调用其 saved `consume`，全 artifact、模块 receipt、PNG/NPY/Q 由其冻结实现复核，不执行 encoder。所有调用源在 import 前验 seal。

## 矩阵不是重复数字表

`result.evidence_matrix` 是机器可消费三层结构：

- `author_reported_cells`：完整既有238cells，236 numeric、2 null。每项保留作者原值/单位/±原义、版本、原 row/column/literal、HTML SHA、PDF SHA/页（PlanWAM 为 unknown）、table/subtable、协议原句、任务/数据/checkpoint/观察/动作/horizon/反馈/预算/指标/样本字段；`local_actual=null`。
- `prior_local_actual`：pinned 六基线 assembler 的12实际行，modelQ/native success/真实Q/预算与 task、blind、commands 身份；`author_reported=null`。这是已有 assembler 的密码学固定快照，并非本模块重新执行其全部上游消费链。所有实现明确为 local variant，不升级为官方论文复现。
- `current_prediction_matched_local_actual`：本次真正修通的四行，包含独立重新消费的 planner、physical 和 RGBv2。与 GN/D2P 的8项描述性比较只有原 root/source/goal/history-task/horizon/metric 均相同时发布，资源仍 `budget_match=false`。

没有把作者均值复制为本地成绩。四论文 `local_official_paper_reproductions` 均 null，跨论文可计算提升项为0，所有提升值 null。未知不是0；PlanWAM 无 PDF 页不从 HTML 顺序猜页码。

固定来源及关键边界：

| 作者版本 | 原表/页 | 不能转换成本地成功率的原因 |
|---|---|---|
| GRASP 2602.00475v1 | Tables1/3/4，p7/8/8 | learned-world-model success；Table1 时间只统计成功trial；环境/H各自超参 sweep，Table4时间分母仍unknown |
| Hi-LeWM 2607.12547v2 | Tables3/4 p13、预算Table5 p14 | 新训练高层+冻结低层，staged/online及d均需保留；通常3seeds×50episodes，不能当本地两根 |
| LeWM 2603.19312v3 | Table10 p26 | solver 的完整预算及±定义未绑定；不能从Table5借统计语义；本地使用LeWM不等于复现该表 |
| PlanWAM 2610.11382v1 | HTML S4.T1/T2/T3，PDF页unknown | 驾驶EPDMS/PDMS/RC/HDScore，四摄像头、另训练模块，不是PushT成功 |

原 sealed 238-cell 导出 SHA `52e687811e6d6f728ab3eda60836ea05eff11f36da2bb4af975d1dd19ef380f1`。本模块重核其引用的原 HTML 和三 PDF 字节，运行前另实际调用已有 `d2p_baseline_reported_values_20261027_v1.audit()`，离线原文重建238cells与导出完全相同；网络/神经/物理调用均0。未重新确认服务器最新版本/DOI。早期20261026文档的“未取得”是历史状态，不覆盖这批后来已获取的固定原文。

## 真实 saved 运行

显式 binding 为 `results/agent_baselines_resume_20261027_sampling_v2_binding.json`，SHA `99598ee5142c8c89cde50258d3855a8ce527f17d8a7dec8176e82631804091db`。它包含全路径/外部SHA，physical `8ce8634eddcd3e2e5b25d5b55572d80d06da1fd5d3513ee06aecec89001efa41`，RGBv2 `63c48c16b91d16b8f20c5f6ba08010d366544558f6e7eb9dd7a9542a5e50fa2f`，planner `60872ad114700731d82d293a616a4dede54a53b7be6514ad0cdf231d838d3ed8`。另含真实 prepare、execution_binding、manifest、journal、source/test/doc、sourcepack，以及原 GN/D2P handoff 到原 CUDA report 的独立绑定，不能把 handoff SHA 当 CUDA SHA。

实际输出 `results/agent_baselines_resume_20261027_sampling_v2_actual01.json`，SHA `c72cb6d96db7ae5cb6527e610145145d66b76c23d038b301efad047fc87ce5ee`，约1.5MiB（≤2MiB）。全4 terminal10/full horizon、0 engineering failure、0 earlystop、0/4 native flags，非4独立根。完整账：construct4/close4/reset8/set_state16/goal_render8/goal_bind4/public_step80/physics_step816/explicit_render88；各 attempted=completed。public/physics 是重叠轴，不相加成“总步数”。

planner 每组144 predict API、144 batch1 predictor samples、144 action_encoder/pred_proj；4组合计576各轴，0AD/J/VJP/SVD、新编码0；继承原8 encoder/projector只记一次。RGBv2 为12 checkpoint+2原goal＝14 encoder、14 projector，不能误标12；本模块新增调用0。MPPI臂虽保留历史tag `kernel_faithful_mppi_nominal`，**此特定144预测合同派发的是72已评分样本全局best，不是最终nominal**；只能称 local nominal-guided sampled-winner variant，不是官方MPPI。与CEM只匹配预测API和样本数，不是等FLOPs/总算力。

实际数值，均完整保留在JSON：

| root/arm | 选中modelQ | terminal真实Q | native距离 | 物体XY距离 | native成功 |
|---|---:|---:|---:|---:|---|
| 0 CEM | .4520115111 | .5307681357 | 79.98825211 | 46.91390997 | false |
| 0 local MPPI variant | .6192752895 | .4074763050 | 138.70832782 | 16.19464790 | false |
| 1 CEM | .0937338773 | .1381070809 | 380.81380829 | 22.97742928 | false |
| 1 local MPPI variant | .7034724913 | .9060070147 | 140.57613253 | 101.23103965 | false |

native距离/负reward包含速度，**success无速度条件**：agent+object四维XY误差阈值及原native角阈值另判。wrapped角/物体XY诊断单列，不默改环境。这四terminal未见negative native angle或native/wrapped success不一致。modelQ、actualQ、native距离及物体pose排序不能互换；不从0/4推出理论成立/失败。

首次集成失败发生在最终矩阵构建：PlanWAM 的 `pdf_sha256="unknown"` 被truthy判断当成有效PDF，导致KeyError。失败未产生actual输出，修为仅对前三篇取已认证PDF；这是运行接线错误，不是CUDA/科学负结果。第二次完整saved执行exit0，以上actual01为修复后唯一发布结果。所有旧source/result均未修改。

## 测试、命令与冻结

source SHA `54f9984c4471099f3dc5c66b72d76298fa6cd6bb8ef653baa3bbddd0b0498502`；tests SHA `fc77050d0aac2112a2817fbc6518965ba5d04e6986f1b73ce3416e26e16554c4`。doc SHA在交付消息另列，避免自引用。37项RAM/metadata tests，无tmp/新fixture磁盘写入；涵盖close、嵌套错误/重复序号/错状态/身份/遗漏返回/失败计账、RGB root/selection/分母、作者与本地分栏、页表版本、wrapped角不变性及未知保null。实际saved运行是集成证据，不是这些mock合成的“actual”。

cwd `/home/gbk/code/LeWorldModel/le-wm-main`，MAIN可复核：

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_baselines_resume_20261027_sampling_v2_test.py
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_baselines_resume_20261027_sampling_v2 --binding /home/gbk/code/LeWorldModel/le-wm-main/results/agent_baselines_resume_20261027_sampling_v2_binding.json --binding-sha 99598ee5142c8c89cde50258d3855a8ce527f17d8a7dec8176e82631804091db --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_baselines_resume_20261027_sampling_v2_MAIN02.json
```

第二条只做saved消费；新输出必须不存在，exclusive-create不覆盖，逐写遵守2MiB cap及200MiB floor，无prepare/神经/物理子命令。实际运行磁盘约60GiB可用。无需重跑旧神经/物理作业修接口。

剩余依赖具体为：官方Hi-LeWM高低层checkpoint/坐标/normalizer配对尚无本地可执行结果；官方GRASP不由本地bounded proxy替代；LeWM Table10逐solver资源及统计绑定缺失；PlanWAM与操纵根没有可直接匹配指标；当前只有两known roots，无独立根P1证据。真实8项GN/D2P reference的source/horizon/metric匹配成立，但预算不匹配，仅描述性差值，reference RGB未在本合同消费则相对actualQ仍null。本次修通证据接口，不解除这些科学前置条件，P1/P3/独立确认/等总算力均false。
