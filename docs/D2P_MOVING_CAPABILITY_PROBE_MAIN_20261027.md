# 移动历史 PushT：真实能力探针与评分近似审计

## 当前结论

主机实际执行固定40分支并完成saved分析和独立复算。D2P前缀会改变后续真实动作响应，但**改变不等于朝目标改善**。两个已知校准根之间效果不一致，不能称独立P1通过、完整C2成立或理论不可行。此次不训练、无新神经调用，不涉及神经CPU降级。

## 实际证据链

路径以下均相对`/home/gbk/code/LeWorldModel/le-wm-main/`。

| 产物 | SHA256 |
| --- | --- |
| `results/moving_overlay_capability_probe_physical_20261027_v1_prepare_MAIN01.json` | `a73f841da07edaf98d6684246ddaf685d9c1860abb1d988aa7fcc93fa2e9a12b` |
| `results/moving_overlay_capability_probe_physical_20261027_v1_MAIN01/report.json` | `e075d3b75c7b5a16c47d8c76c846eaa2ac4d8ebd202abd18037a7242792b076a` |
| `results/moving_overlay_capability_probe_analysis_20261027_v1_MAIN01.json` | `19661bb010662dcfc7c9397723ecb085080fe18a7de069bdaac1f7c59e0e7557` |
| `results/audit_moving_capability_MAIN_20261027_v1_MAIN01.json` | `e4f668ce10f4389019f04d4fa507a6efdfa66158fc7a1609feeb74d23420d43d` |
| `results/audit_moving_compiler_surrogate_MAIN_20261027_v1_MAIN01.json` | `f4c2f2de9022eca48d1a94b78bf18338a2d410c19f639c04ddee0791e5a0f720` |

真实物理状态`capability_probes_recorded`，40/40完整、errors=[]、inputs_final_verified=true，369.964127秒。实际完成constructor/close各40、reset80、public_step800、physics_step8160、set_state160、goal_setter120、goal_render80、goal_bind40、explicit_render2640。所有attempted=completed。journal SHA `e7457cbfaecd1a9de7641dafa48f32deb1925879c75b144faf45e3f63c0b1142`，tail `f42975b62296ba845e5a6800af6d35ac5c4e8488ddef506632465746ebcf6182`。

固定2 roots×off/on×GN/D2P八个prefix5锚点，每个测试zero/±x/±y各5步。raw幅度取float32(.1)，不据结果选方向/幅度。每次fresh环境重放原reset、非零warmup10、prefix5及完整off/on/off渲染顺序；51项历史/前缀图像文件与像素SHA、state7/body bits/goal/renderer严格匹配后才允许probe。不是使用缺少block速度的native setter近似恢复suffix。隐藏solver/contact cache未序列化，不声称其逐位认证。

MAIN全文审读producer、分析器，实际31/53测试通过；新独立复算器5测试通过，新全候选评分审计6测试通过，合计11项针对性测试通过。独立复算不导入producer、模型或物理模块，直接用保存trajectory核32个objectXY/agentXY/jointXY/angle矩阵、奇异值、全部zero-relative有限响应和任务投影；同时重算journal链与所有资源计数。它交叉验证分析，不以测试或复算代替真实执行证据。

## 真实响应的关键区别

以下`task response norm`为沿**原root→goal物体位置方向**投影的二维中心secant行范数，单位px/raw_command。此范数无符号，不能单独表示正向控制收益。固定±动作有限响应使用endpoint−zero endpoint，保留所有负号。

| root | rendering | GN task response norm | D2P task response norm | D2P固定探针的正向有限响应 |
| --- | --- | ---: | ---: | --- |
| 0 | off | 113.37966 | 27.61054 | 无；+y为−5.52211px，其余非zero探针为0 |
| 0 | on | 0 | 23.38953 | 无；+y为−4.67791px，其余非zero探针为0 |
| 1 | off | 0 | 112.76928 | −x为+12.32577px，+y为+17.76072px |
| 1 | on | 8.15394 | 111.60468 | −x为+12.68572px，+y为+17.58360px，−y为+0.13158px |

完整五探针原始数值均在分析JSON，不按“最好探针”汇总研究成功。root0的D2P objectXY矩阵只有一个非零奇异值，root1的D2P有两个非零奇异值；这只是固定尺度有限响应，不是全局可控性、局部微分rank证书或因果泛化。

GN root0/off与D2P root1还存在zero动作漂移，已保留并用真实zero endpoint对照。不能把惯性漂移全部计作动作控制收益。不同单位的位置与角度分开，不拼接成无单位总谱。独立根数仍为2个已知校准任务，不是40个样本。

## 线性评分与非线性预测：发现范围有限

MAIN重新检查实际compiler选法：`objective=stage_cost+own_reach_value`；own_reach_value为有界ridge线性残差与动作惩罚的均值。虽然每个候选已保存完整非线性终态预测，但选择仍按线性代理，符合原声明，不是新确认的接线bug。

新saved审计逐个复算全部四库40候选的stage、`Q(goal-nominal-Ju)`、ridge惩罚、非线性终态代价及原winner。结果：仅root0/off的排序会改变。原index6的非线性对应总代价为1.44165311967；同库index2为1.39979062105，差0.04186249862。其余3库对应winner不变。因此线性代理失配是可研究的近似问题，**不足以单独解释全部失败**。

本审计不释放新动作、不执行诊断analog winner、不重选根、不改旧报告；更低模型代价也不是更好物理结果。若验证新选择规则，必须新建事前固定协议，覆盖全部固定库/根而非只执行该有利case；仍需真实任务轴与强基线，而非只看latent。

## 对下一步的约束

1. 下一项有判别力的主机CUDA诊断应预测同一固定探针库，区分“模型是否预测到真实响应方向”与“评分是否选择任务有用方向”，必要时比较真实teacher前缀与open-loop预测前缀；不拟合新readout、Q或统计。
2. unsigned谱强度与有符号任务可达收益分开。当前原模型距离/物理方向关系仍未闭合，不能仅增大rank/响应norm就开放P1。
3. fresh根历史排重认证、强基线公平预算、完整P1机制、P2组合增量、C3调度与三实验族仍是原scope；本次完成的是其真实机制定位证据，不是整套研究完成。
