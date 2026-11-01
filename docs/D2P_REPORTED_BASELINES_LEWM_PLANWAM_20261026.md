# LeWM / PlanWAM 原文报告与当前P0/P1的比较边界

核验日期：2026-10-26。状态：partial。本文只报告实际读到的一手来源，不将附件数值、摘要概括或内部计划升级为已核验实验表。`paper_reported`不是`local_reproduction`；`context_only`不是同模型、同任务或等预算。

## 来源、版本与覆盖

| 工作 | 已确认身份/来源 | 已读范围与缺失 |
| --- | --- | --- |
| LeWM | Lucas Maes、Quentin Le Lidec、Damien Scieur、Yann LeCun、Randall Balestriero，*LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels*；[官方v1摘要](https://arxiv.org/abs/2603.19312v1)、[v1 HTML](https://arxiv.org/html/2603.19312v1)、[v1 PDF](https://arxiv.org/pdf/2603.19312v1)；本地 `le-wm-main/README.md`明确链接该v1 | 官方摘要与HTML请求曾成功返回，但HTML工具输出截断，后续提取请求审批超时；没有完成结果表、附录Table 10和设置的逐项核读。不能把“请求成功”写成“表值已核验”。PDF页码未核实，不声称这是当前最新版本 |
| PlanWAM | Jinchang Xu等，*PlanWAM: Planning-Shaped Future Representations for End-to-End Autonomous Driving*；[v1 HTML](https://arxiv.org/html/2610.11382v1)、[v1摘要](https://arxiv.org/abs/2610.11382v1)、[v1 PDF](https://arxiv.org/pdf/2610.11382v1)；缓存原文头部标`arXiv:2610.11382v1 [cs.RO] 08 Oct 2026` | 优先读取已有原文转存 `/home/gbk/.local/share/arxiv-mcp/2610.11382v1.md`，核对§4.1–4.2、Tables 1–3、附录B/C。未重新确认服务器最新版本或PDF页码；下面表号来自实际正文，不猜页码 |

缓存原文可核对内容但不是本轮新下载的PDF，也不提供独立PDF字节真实性证明。全文未取得/未完整核读写`unverified`；已读原文没有定位字段写`unknown in inspected text`，均不等于作者从未提供。作者项目链接：[LeWM](https://le-wm.github.io/)、[PlanWAM](https://jinchan9.github.io/PlanWAM_project_page)。PlanWAM项目页本次审批超时，不能宣称其代码或权重可下载。

## 可转录的PlanWAM报告值（仅该论文自己的协议）

全部为上述v1原文报告，指标越高越好、表内0–100尺度；不是本地跑出的结果，不是Reacher/PushT成功率。

| 原表/位置 | 任务/split | 方法与指标 | 原报告值 | 统计范围/限制 |
| --- | --- | --- | ---: | --- |
| Table 1，§4.2，v1 HTML | NAVSIM-v2 `navtest` | PlanWAM EPDMS | 90.9 | 官方扩展proposal scorer；没有从这一个汇总值恢复seed方差或置信区间 |
| Table 2，§4.2，v1 HTML | NAVSIM-v1 `navtest` | PlanWAM PDMS | 93.8 | 非反应式4秒proposal simulation，不是连续RGB闭环成功率 |
| Table 3(a)，§4.2，v1 HTML | HUGSIM zero-shot，全部436 scenarios | PlanWAM Route Completion Avg. | 53.7 | 场景加权均值，不是四个难度的等权均值 |
| Table 3(b)，§4.2，v1 HTML | 同上 | PlanWAM HD-Score Avg. | 38.7 | 与RC不同指标，不互换为success rate |

原文Table 3完整难度分层保留如下，避免只选最有利任务；不与其他论文混排：

| 指标 | Easy | Medium | Hard | Extreme | 原表Avg. |
| --- | ---: | ---: | ---: | ---: | ---: |
| RC | 87.6 | 48.3 | 51.9 | 37.2 | 53.7 |
| HD-Score | 75.3 | 32.9 | 35.6 | 22.0 | 38.7 |

附录C.4/Table 13披露难度场景数80/157/96/103，总436；Avg.以这些场景数加权。HUGSIM来自nuScenes、Waymo、KITTI-360、PandaSet。表中一位小数已经舍入，不反推未舍入值或seed不确定性。

## PlanWAM逐项协议登记

| 必需字段 | 原文确认的信息 | 尚缺/不可推断 |
| --- | --- | --- |
| task/split | §4.1、附录C.1/Table 12：NAVSIM `navtrain`训练、`navtest`评价；navtrain 103,288 scenes（约85k train/18k validation），navtest 12,146；v2本论文用同split。HUGSIM 436公开场景，NAVSIM模型zero-shot，无HUGSIM特定微调 | 逐scene ID、完整manifest及本地重放绑定未取得 |
| checkpoint/训练 | DINOv2 ViT-S基础权重冻结，但rank-32 LoRA及规划相关模块训练；posterior→prior两阶段。Stage 1/2 AdamW学习率1e-4/1e-5；16×A800，每卡batch8 | 不属于整个模型完全冻结的training-free D2P；发布权重文件、revision、digest、许可与可获取性未核实 |
| observation | 4历史frames×4摄像头（front/left/right/rear）；方法包含ego feature；训练posterior可看未来，部署只保留history-conditioned prior | 不可简写为单张RGB；不把训练未来输入说成部署oracle。具体本地sensor/state schema未核实 |
| action/output | 规划器生成64条候选trajectory，每条8 waypoints，在同一4秒时域上评分选择 | 非Reacher torque或PushT推杆动作；坐标单位、低层tracking/control适配未核实，不强映射到本地action block |
| horizon/反馈 | future posterior训练观察未来1/2/3/4秒；trajectory horizon=4秒；NAVSIM非反应proposal评价，HUGSIM闭环 | 历史采样间隔、闭环执行chunk与重规划频率未逐项确认；8 waypoints不自动推出本地rollout tokens |
| budget | K=64，历史压缩128 tokens、未来latent256 tokens；训练硬件有披露 | 测试predict calls/samples、实际backward、总FLOPs、wall-clock硬件/重复测时未完整核实。候选64不等于本地64次模型调用 |
| metric | NAVSIM-v1 PDMS、v2 EPDMS；附录C.2–C.4给出定义；HUGSIM RC为完成路线比例，HD-Score结合RC、碰撞、可行驶区域、TTC和comfort | 不是物体pose distance，不是非贪心/能力改变机制指标 |
| seed/统计 | Table 3按436 scenarios评价，分难度场景加权 | 检查的原文中未定位可转录seed列表/多训练seed误差条；不能将场景数充当独立训练种子，不能补SD/SEM/CI |
| 表页/URL | v1 HTML §4 Tables 1–3，附录C Tables 12–13；URL见来源表 | PDF页码unverified；不构造未经核实的`#page=N`链接 |

## LeWM：数值仍不能从附件直接升级

新附件声称Appendix Table 10含CEM/SGD/RMSProp/Adam结果及PushT goal offset/执行预算。由于本轮未逐项核读该表，这些均为`attachment_claim_unverified`，不转录为`paper_reported`，也不据此判定某优化器在当前P0/P1胜负。官方摘要的“约15M参数”“最高48×规划加速”只能作为摘要概括，不能替代具体任务/硬件/时域/计时表，不纳入性能主表。

| LeWM字段 | 本轮状态 |
| --- | --- |
| 唯一身份/版本/官方URL | 已核验v1身份及README链接；最新版本未核实 |
| Table 10及其他规划表的数值、PDF页码、误差条定义 | unverified，未录入已核验数值 |
| 原论文task/split/checkpoint revision/observation/action | 实验表级别绑定unverified；本地存在权重不证明与原表完全同一revision |
| goal offset、预测horizon、执行预算、反馈频率 | unverified；不按附件填写 |
| 候选/迭代/seed/metric/success predicate | unverified；不从本地默认配置反推原论文设置 |
| 官方资产线索 | README链接[LeWM HF collection](https://huggingface.co/collections/quentinll/lewm)；本地`/home/gbk/.stable_worldmodel/checkpoints/{reacher,pusht,cube,tworoom}/lewm/`可见config/README，未加载权重、未进行大文件哈希，也未认证原论文到这些文件的精确溯源 |

## 与当前P0/P1的可比性

本轮读取 `le-wm-main/results/reacher_d2p_p0_predict_sample_matched_calibration2_v1_20261026_cuda/report.json`：`calibration_only`，history_len=3、future_horizon=2、action_block=5、context_support=uniform、`p1_unlocked=false`。独立计数收据为每根D2P/CEM 536/536 predict samples，但calls为122/8；这不证明总算力匹配。当前P1要求非贪心前缀、真实未来能力改变、后缀物理收益以及独立outcome，不是单纯轨迹评分。

- LeWM是原生世界模型/规划器对照的直接候选，但原论文报告值在split、checkpoint、动作/观察、指标及预算未对齐前为`insufficient_metadata`；本地同协议CEM/GD/GN仍需自己的证据。
- PlanWAM已核验的值全部为`paper_reported/context_only`：自动驾驶任务、训练方式、4摄像头历史、trajectory output、PDMS/EPDMS/HD-Score均不对应当前P0/P1。可用于“planning-shaped representation”相关定位，不能计算D2P相对提升率、宣称同预算胜利或代替P1机制实验。
- 新附件“预算路线P0–P3”只表示工作优先级，不替换原研究阶段P0–P5；整理Published Results不等于原研究P0通过，提前实现组合也不能跳过原P1。

## 实际核验与阻塞

用research-lookup的一手来源优先原则区分原文、官方项目页和附件线索；未使用记忆补数。网络均经普通权限尝试及正常审批，不通过其他工具绕过权限。LeWM摘要/一次HTML返回成功；后续全文提取、PlanWAM项目访问多次自动审批超时。PlanWAM数值来自已有原文缓存，不是假称本轮在线PDF核验。补齐LeWM表值与所有PDF页码仍需可读原文或成功网络访问。

本任务新增仅本文件；未运行模型、训练、GPU、pytest或任何新物理实验。来源核验不构成研究完成。
