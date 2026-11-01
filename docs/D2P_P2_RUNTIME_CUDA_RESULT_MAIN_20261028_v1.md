# P2 runtime guard：MAIN 实际 CUDA 工程结果

MAIN 完整审读最终 source/tests/innerdoc（包括新 metadata writer）；独立联合测试 56 passed（36 release +20 guard），此前 native/guard 联合回归 77 passed。saved wrapper verify 退出 0。审读冻结引用见 `D2P_P2_RUNTIME_MAIN_REVIEW_20261028_v1.md`，SHA256 `52599f94ff9e60db7673310521a1536b9d24fe033bcca8be6206e814946918c8`。

实际主机 CUDA 命令执行终结、退出 0，`le-wm-main/results/agent_p2_runtime_guard_20261028_v1_known_MAIN01/report.json` SHA256 `0cfbe37a7783c6df331305e95e3c058a9b006f72bb1ca5b59d24fae047ee23a0`。报告 `known_root_completed`，error=null、secondary_errors=[]、model_unchanged=true、source_unchanged=true。四个根报告均 complete=true、pending_operations={}。

## 实际分轴（先按 producer 保存记录，待独立 consumer 全量对账）

每根 332 predict API /332 samples /996 context tokens，332 action_encoder；664 个 predict+AE 原入口记录。全四根合计 1328 predict，不能把不同资源相加叫 FLOPs。每根 1 个 192×20 Jacobian、40 个 192×10 Jacobian，82 份导数数组收据。失败分母固定四根／八臂／480 候选；没有新 physics、encoder 或训练。

参数 SHA 前后 `2819d25722be6aeaf6f124f43097633cde51337a4ba83d06b2edc6b4729630cb`，buffer SHA 前后 `e5253360c642186a30b274edaf707227edf6dad55b2aadbce7fa5853e8bdf51e`。实际 backward engine API、kernel profiler、FLOPs、同步设备时间及历史 acquisition cost 仍未知，不以 0 填补。

## 模型内评分（不是物理结果）

| 原固定 root slot | joint H2，无组合 | conditional suffix composition |
| --- | ---: | ---: |
| 0 | 1.39115063 | 1.32249796 |
| 1 | 0.77597554 | 0.89478084 |
| 2 | 1.20924887 | 1.07658399 |
| 3 | 0.89583112 | 1.02917531 |

组合臂两根模型 Q 更好、两根更差，不挑根、不改候选或预算。相同 predict cap 不是全资源等预算；两臂候选数和导数工作不同。只读 saved consumer 正在实现以重算候选、数组、journal 与实际资源对应关系；本记录不冒充该 consumer 已通过。

这是已知根工程执行。独立 P1、freshness、P2 物理收益、全资源等预算以及原 C2/C3 贡献均未关闭，P1/P2_admission=false 保持不变。

## MAIN 严格 saved 消费已实际完成

MAIN 完整审读最终 consumer/source/tests/innerdoc，核对三个冻结 SHA，实际联合回归 **75 passed in 59.41s**。随后以真实 report/prepare/MAIN review SHA 运行新 CLI，退出 0；持久回执 `le-wm-main/results/agent_p2_guard_saved_consume_20261028_v1_MAIN02.json`，SHA256 `c7d183f5b18d1bfacd81f855025efdbbb403f97e5442650970080766db17088b`。

四根八臂 480 候选、1328 predict/AE、3984 context tokens 均完成独立逐事件解释与原完整数组/候选/动作见证重算；每根 123 paid groups、867 primitives、82 数组。packet ledger 与实际 AE→predict 返回逐条对应，不重复加账。

两臂只有预测前沿匹配，完整资源仍不相同；backward-engine、kernel、FLOPs、同步设备时间、历史 acquisition 保持 null。该回执没有新神经或物理调用，`physical_execution_authorized=false`；不能把它当独立物理复现、P2 增量成功或全资源等预算证明。
