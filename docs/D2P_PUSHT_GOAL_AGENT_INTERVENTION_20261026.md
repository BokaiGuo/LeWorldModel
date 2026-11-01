# PushT 目标图像中的推杆干预：真实CUDA证据与候选修正

## 最新真实执行：仅修后缀度量没有恢复物理收益

候选已接入同一个Q的谱/DLS初始化与GN4后缀优化，保持原4个校准根×16个
固定前缀，全量与identity-GN4配对。主机CUDA完成3968预测、128真实分支、
11项完整性门；identity动作与历史完整度量GN4逐位相同，终态RGB hash相同。
无训练、无未来标签选动作，原失败根全部保留。此轮只改观察后后缀，尚不是
首段生成/stage/own-state全链新编译器，更不是独立P1。

独立审计复算全部64对，核对封存协议、源码、stage收据、动作/根身份及历史
baseline重放。nuisance相对identity的物理pose cost均差+2.930957，13对改善、
23对恶化、28对相同；四根均差分别+6.862024/+0.143690/+0.776308/+3.941808。
共同nuisance度量的实际编码成本均差−0.063998。不同各自度量的成本下降不
可当共同目标比较。64对不是64个独立根，不能据此声称独立泛化或理论否定。

这说明目标图像保真检查的正面结果不能代替物理控制证据；去推杆方向本身
没有修复当前短后缀。不得按物理收益删根或挑候选。审计还明确：未保存完整
Jacobian和每步参数，无法完整独立重算谱/GN方向；饱和动作不能逆恢复原
atanh正则参数。后续新运行必须补齐这些证据，而不是改写旧报告。

产物：

- `le-wm-main/results/pusht_d2p_nuisance_suffix_bank4x16_v1_20261026_cuda/report.json`
- `le-wm-main/results/pusht_d2p_nuisance_suffix_bank4x16_analysis_v1_20261026.json`

独立分析器16项测试通过；并行新增program scoring与ledger的集成回归共
129项通过。主机CUDA三项数值接线测试通过，验证投影同时控制Jacobian和
非线性目标、identity重现原GN、CPU输入在调用模型前拒绝。不是效能证据。

并行核账定位到旧P0计数存在调用/样本混用、final诊断未全量纳入等缺口；
新纯ledger已实现但尚未接入旧runner。历史强基线等预算主张不能由该ledger
事后认证。后续优先使用实际predict调用计数，而非旧公式推断；不放宽P1。

## 代码层面的任务合同差异

环境reset先把完整七维goal_state置入模拟器并render，因此目标RGB同时指定
推杆和物体。原物理评价`_pose_label`仅评价物体xy与朝向。两者对任务的约束
并不等价；这不是自动证明某个编码器错误，也不是模型理论不可行的证据。

本轮保持原4个已知校准根、原16动作前缀、三种后缀的全部固定预测不变，
只在目标渲染时施加推杆xy偏移(0,0)、(-32,0)、(32,0)、(0,-32)、(0,32)。
目标物体、角度、其余goal坐标不变，没有执行新未来、重优化动作或选择根。
这是目标合同的干预诊断，不是RGB-only部署的新控制器。

## 已完成主机CUDA审计

官方原冻结权重；20次goal encoder forward，0次新predict、0条新物理未来。
原目标图像逐位SHA与上游一致，原目标latent数值与动作库一致；9项完整性门
通过，包括模型参数/buffer/eval、权重/config/源码/动作库不变。

在4根×4个非零目标偏移的16种干预中，固定动作库的完整terminal目标argmin：

| 后缀类型 | 相对原目标的argmin改变次数 |
| --- | ---: |
| random8 one-shot | 0/16 |
| full192 one-shot | 4/16 |
| full192 GN4 | 2/16 |

以上argmin仅为固定库terminal-cost诊断，不是原重配置selector；不能宣称
原selector已改变或任务控制已改善。它确认同一物体任务的评分会受目标推杆
位置影响。独立保存数组分析重新计算960个完整坐标成本与全部argmin，一致。
这仍不能证明移除推杆依赖会恢复P1，或解释所有剩余失败。

产物：

- `le-wm-main/results/pusht_d2p_goal_agent_intervention4_v2_20261026_cuda/report.json`
- `le-wm-main/results/pusht_d2p_goal_agent_intervention4_analysis_v1_20261026.json`

首轮v1因新脚本误用了同名数组哈希辅助函数来检查文件，推理前严格停止；
系统`sha256sum`确认权重未变。修正为文件内容digest后完整v2通过，原失败目录
保留，不计为模型负结果。新增文件哈希接线回归防止复发。

## 候选修正已实现，未作为已验证控制器部署

新增`pusht_d2p_goal_nuisance_metric.py`：从明确提供的同任务goal图像集合的
编码差分做SVD，构造Q=I-NN^T，投影掉这个有限差分span。只优化临时任务
几何，不训练模型、adapter或readout，不读未来结果。不根据已知控制收益
选择rank；仅用float64标准数值rank容差。整个空间被移除时严格拒绝。

这个候选只能保证所提供有限goal集合的投影相同，不能保证所有推杆位置
不变性；更不能保证物体方向不会一并丢失。实现尚未接入整个谱生成/后缀/
stage/own-state目标链，尚无新物理收益证据，不能当作修复成功。

它需要显式goal图像集合，单张RGB不能凭空提供该集合。本轮模拟器坐标只用
于诊断渲染，不能偷偷作为部署时nuisance remover输入。任何正式方法必须
明确声明任务输入合同，并独立检查物体目标敏感性、全链一致性、强基线与
新根泛化。原P1保持关闭，原C1--C3及三组实验范围不变。

## 实际执行

### 后续已完成：物体敏感性与未见推杆位置（2026-10-26）

以原五张goal构造的投影保持不变，每根再渲染原图、物体x/y±16、角度±0.15，
以及推杆xy四个±32对角位置；4根共44次主机CUDA编码，0次新predict、0条新
物理未来。未根据结果改变rank或选择扰动，9项完整性门通过。所有原图重现
上游hash，物体/推杆扰动均在原五图构造之后才评估。

| 测试信号 | 投影后编码变化能量保留比例 min / median / max |
| --- | --- |
| 24个物体平移/旋转干预 | 0.726814 / 0.867725 / 0.976395 |
| 16个未见推杆对角位置 | 0.001522 / 0.002828 / 0.074523 |

独立分析不调用构造器的SVD投影结果计算残差，而是用原五goal差分的最小二乘
拟合重算44个残差能量；rank、Q对称/幂等、原五图差分消除和全部比例复核一致。
新增6项独立几何测试通过，与原度量测试合计12项。最新完整D2P相关回归
1626项通过（50.01秒），`git diff --check`通过。

这是继续验证候选的正面证据，不是控制信息完整保留的保证：最差物体信号
约27.3%能量也被移除；目标图像变化不同于真实后继动作响应。它既没有证明
Q后的模型预测排序正确，也没有证明物理收益。正式输入仍需要显式同任务goal
图像集合，尚不能声称单目标RGB部署。全链编译、实际后缀、非贪心与真实
能力/效能独立准入仍待完成，P1及依赖阶段保持关闭。

产物：

- `le-wm-main/results/pusht_d2p_nuisance_object_sensitivity4_v1_20261026_cuda/report.json`
- `le-wm-main/results/pusht_d2p_nuisance_object_sensitivity4_analysis_v1_20261026.json`

实际主机CUDA命令：

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/lewm-matplotlib .venv/bin/python -m experiments.spectral_reachability.audit_pusht_d2p_nuisance_object_sensitivity \
  --intervention results/pusht_d2p_goal_agent_intervention4_v2_20261026_cuda/report.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/pusht/lewm \
  --output results/pusht_d2p_nuisance_object_sensitivity4_v1_20261026_cuda

OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.analyze_pusht_d2p_nuisance_object_sensitivity \
  --report results/pusht_d2p_nuisance_object_sensitivity4_v1_20261026_cuda/report.json \
  --output results/pusht_d2p_nuisance_object_sensitivity4_analysis_v1_20261026.json
```

新增13项目标干预/有限span度量测试通过；完整D2P相关回归1620项通过
（49.77秒），`git diff --check`通过。测试不替代物体目标保真或物理效能证据。

在`le-wm-main`、主机CUDA执行：

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/lewm-matplotlib .venv/bin/python -m experiments.spectral_reachability.audit_pusht_d2p_goal_agent_intervention \
  --bank-report results/pusht_d2p_full_metric_bank4x16_20261026_cuda/report.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/pusht/lewm \
  --output results/pusht_d2p_goal_agent_intervention4_v2_20261026_cuda

OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.analyze_pusht_d2p_goal_agent_intervention \
  --report results/pusht_d2p_goal_agent_intervention4_v2_20261026_cuda/report.json \
  --output results/pusht_d2p_goal_agent_intervention4_analysis_v1_20261026.json
```
