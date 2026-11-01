# GRASP 官方源码精确合同与最小可接入核心

本轮只新增 GRASP 独占文件；不编辑旧 sealed 源码、配置、报告或外层 central 文档。
没有加载 checkpoint、运行神经模型/GPU/physics、训练、安装或下载大包。
paper-to-code 技能用于逐项算法到接口映射，TDD用于解析动力学反例，不执行技能中的训练/性能实验。

## 本次锁定的一手实现

| 仓库 | 固定 commit | 角色 |
|---|---|---|
| [michael-psenka/grasp](https://github.com/michael-psenka/grasp/tree/64f988aebf04ad9f674f9821bc6d2b9e22879105) | `64f988aebf04ad9f674f9821bc6d2b9e22879105` | DINO-WM 完整 planner 与最小 pseudocode |
| [michael-psenka/le-wm-grasp](https://github.com/michael-psenka/le-wm-grasp/tree/94be3c42aa2f0867c6cb1e38bda498b293b6921e) | `94be3c42aa2f0867c6cb1e38bda498b293b6921e` | 官方 README 链接的 LeWM/PreJEPA solver 适配 |
| [michael-psenka/jepa-wms-grasp](https://github.com/michael-psenka/jepa-wms-grasp/tree/2d81ea8d84d1f7c457fb1baf00621cff142afbe3) | `2d81ea8d84d1f7c457fb1baf00621cff142afbe3` | 官方 README 链接的 native JEPA-WMs GRASP planner |

身份链：已封存原论文2602.00475v1指向 `michaelpsenka.io/grasp`；仓库homepage对应该项目页，
GitHub owner姓名Michael Psenka；固定 README明确链接两个适配仓库。项目页直连TLS reset，
raw.githubusercontent也reset，因此使用官方GitHub只读API的固定commit/tree/blob。
每个blob解码后检查 `SHA1("blob <len>\0"+bytes)` 与commit目录记录相同，再封存SHA256。
仅取小源码/配置/README/LICENSE/元数据，不clone、不运行install脚本，不执行或import远端源码。
源文件保存为 `.source.txt`，不是可导入vendor包。单响应256KiB、持久化源合计2MiB上限，实际远低于该值。
仓库LICENSE是MIT，原notice完整保留于 `sources/LICENSE.source.txt`。

这是**当前源码版本**核验；尚未证明它就是论文Table1/3/4运行时commit，也未取得论文具体权重/配置/seed/trial绑定。
`paper_results_code_commit=unknown`；本轮不修改v3文献表来追认这种绑定。

## 不能混淆的三种实现

| 机制 | DINO完整 `planning/grasp.py` | 最小pseudocode及LeWM/native适配 |
|---|---|---|
| 虚状态 | cylinder或Gaussian初始化，优化选定中间状态 | 线性插值T−1个中间visual状态，起点/goal固定 |
| soft dynamics | channel sum、其它维度mean；另有非末步dense goal项、可选Lagrangian/regularizer | 每edge MSE的sum，加每edge预测到goal的MSE sum |
| detach | 默认state-input detach，`enable_state_gradient`可关掉 | lifted phase detach state input；action和目标侧虚状态仍有梯度 |
| 噪声 | 实际执行 `randn_like(z_state)*current_noise_scale` | joint Adam后仅给虚状态加Gaussian，不给动作加restart噪声 |
| sync | 0-based k%interval==0且k>0；有改善才接收，更新虚状态，拒绝可减LR | (k+1)%interval==0且k>0；只更新动作，重建joint Adam，不刷新虚状态 |
| serial gradient | sync穿过完整autoregressive链 | 同样保留完整state-chain梯度，不能沿用lifted detach |

重要源码细节：DINO完整planner虽然计算residual相关 `noise_scale`，最终实际噪声用的是
`current_noise_scale`，不能根据注释实现成residual-adaptive噪声。其sync最后返回的loss/rolled states
来自最后一次**pre-update forward**，而动作已optimizer.step；不能把它当作post-update verification。
LeWM/native适配的 `randn_like` 使用global RNG；构造器里的generator并不覆盖这条噪声路径。
新核心改为本地显式seed generator，是已声明的可复现性变体，不声称逐bit等价。

准确源码位置不靠本文硬记：`prepare/contract.json` 每条mechanism含固定commit URL、源文件SHA、
原文摘录与起止行、摘录SHA。包括以下原实现：

- DINO `_init_states_cylinder`、`compute_loss`、`plan`、`VisualWorldModel.rollout_encoded`。
- LeWM `GRASPSolver._jepa_predict_step`、`_jepa_full_rollout`、`solve`。
- native `GRASPPlanner._make_context`、`_single_step_rollout_flat`、`_compute_grasp_loss`、`_sync_step`、`plan`。

## 新代码实际实现什么

`p2_grasp_reference_core_20261027_v1.py` 是独立写的**官方H1适配机制子集**，不是内部GD改名：

1. 固定encoded start/goal；T−1个可优化虚状态；全零初始动作。
2. 每edge计算 `F(stopgrad(s_t),a_t)`，soft dynamics和dense goal shaping同时更新动作/虚状态。
3. joint Adam后给虚状态加常数std Gaussian。
4. 按官方适配索引周期执行fresh action-only Adam sync；梯度穿过serial rollout；重建joint Adam。
5. 返回最终动作（不是按旧outcome筛最佳），另做一次完整independent terminal verification，单独记账。

不实现DINO完整planner的cylinder/Lagrangian/parallel-topk/accept-reject、可选schedule decay或CEM sync，
也不假装这些只是无关基础设施。`official_equivalence=False`、`local_reproduction=False`。
输入必须是已有编码latent/goal及已经规范化的动作坐标；无encoder、normalizer、tanh/clamp或执行器。
生产入口只接受显式frozen eval CUDA适配，无CPU神经fallback。CPU仅允许测试显式声明的非Module解析函数。
无CUDA执行CLI，此prepare不授权启动CUDA。

最小接口已写，主线程可直接review而非只有文档：

- `LeWMH1(model, history_size=3, action_dim=10)`：单帧 `model.action_encoder` + `model.predict`；
  sync保留生成的latent/action历史。编码输入如官方PushT192维，须由未来runner显式检查，不能仅靠duck typing。
- `NativeH1(predictor, last_proprio=..., visual_shape=(1,16,16,384), action_dim=10, history_size=3)`：
  不pool patch、不拿LeWM CLS替代native；单edge使用最后真实encoded proprio，serial sync中传播预测proprio，
  native normalized10维action直接进AdaLN，不额外`/100`、重编码或双重标准化。

本地native `AdaLN_vit.py` 的Git blob与官方适配树一致；`prop_embedding.py`、`pusht_dset.py`不一致。
contract记录双方blob和本地SHA，不能据单个predictor一致宣称整个native仓库/数据处理一致。
新接口绕开有历史padding差异的旧wrapper，但必须在后续CUDA runner与官方适配unroll逐步验证后才能称数值等价。

## 真实H3历史是尚需主线程选择的实验合同，不是隐形默认

**官方适配把起点压成最后一帧H1**，native每条virtual edge也重新使用最后初始proprio。
本地已有P0/P2/P3输入则是三帧真实latent加两条真实历史action packet。新核心诚实复现H1-reset策略，
不能接上旧H3历史后沿用旧baseline公平性标签。

可实施的两条路径：

- 最小官方机制路径：先做新H1-reset encoded CUDA接口检查，所有对照臂也使用同H1起点、相同goal/动作域，
  成功后另设计独立paired任务；不能与旧H3结果直接配对。
- 若必须保持现有真实H3：另版本定义history-aware collocation。每个edge需明确虚状态涵盖哪些历史帧、
  历史actions是否参与梯度、native proprio如何传播/约束；sync仍用真实历史。它是明确扩展，不是已核实的官方H1等价。

本prepare不代替主线程选择，不绑定旧四根、不开旧outcome、不按结果修改公式。
也没有把“取最后一帧”的预处理变更藏在接口里面：类名、protocol、contract均显式H1。

## 冻结预算与实际剩余

新默认直接采用官方LeWM `config/eval/solver/grasp.yaml`：750 joint steps、lr_s=.1、lr_a=.005、
noise=.005、interval50、sync10steps/lr=.05。local rollout horizon显式T=3、seed20261027，不按outcome调参。
它不是旧动态预算调度的“147预算”对照，也不是论文最佳sweep的复现配置。

每root batch1计划开销：2250 lifted transition samples + 450 sync transition samples + 3独立verification
= **2703**；750+150=900 backward API calls。LeWM每transition还涉及action encoder；native直接predictor。
实际底层module调用、编码、失败、OOM与peak memory需未来CUDA runner hook记录。
当前receipt只计transition边界请求/完成和backward，不把估计当实际全pipeline成本。
编码/模型加载成本在核心之外，保持unknown。所有abort保留事件与`PlanningFailure.receipt`，不impute为成功。

仍缺：选择H1/H3合同；新root/task/feedback/goal与动作域/冻结权重绑定；动作无界与现有rawbox的合同决定；
native接口数值与action梯度验证；完整predict/encoder/backward/verification hooks；小CUDA memory预检；
独立paired任务和真实physical consumer。任何新action clamp/tanh必须成为明示变体，不默认继承官方算法标签。
这些是可执行前置，不影响本次源码核验，但足以阻止目前声称P0/P2强baseline已跑通或同全pipeline预算。

## 离线测试、prepare和封存

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_p2_grasp_official_contract_20261027_v1.py \
  tests/test_p2_grasp_reference_core_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m \
  experiments.spectral_reachability.p2_grasp_official_contract_20261027_v1 --prepare
sha256sum -c results/p2_grasp_official_contract_20261027_v1/prepare/SHA256SUMS
```

prepare目录固定exclusive，重复执行拒绝覆盖。生成`contract.json`、`prepare.json`、`SHA256SUMS`，
绑定新source/test/doc、全部官方小源码、paper原文及相关本地ABI源；零root、零model transition、零encoding、零physics。
prepare只import标准库和新核心dataclass，不import torch、不construct模型、不读权重。
测试仅解析动力学tensor algebra：反例验证state-input detach但目标/动作仍可求导，sync链式梯度，
虚状态噪声与seed、本地history/patch/proprio ABI、故障账本、独占输出、源SHA污染及固定repo版本。
先记录缺模块/缺函数失败后实现；测试不能充当真实模型等价或控制性能证据。

原始raw-source获取曾网络失败、后与成功blob获取路径冲突导致该批receipt未flush；
`retrieval_incident.json`诚实记录观察到的失败，不伪造HTTP回执或时间。最终权威是成功API receipts及git blob/SHA验证。
封存后请勿编辑；后续runner/真实H3扩展必须独占新版本。
