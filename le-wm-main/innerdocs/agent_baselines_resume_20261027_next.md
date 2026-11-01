# 下一项同反馈强基线：固定 observed-prefix CEM64

本实现仅新增 `agent_baselines_resume_20261027_next` stem，旧 sampling_v2、producer、论文、模型、结果和共享总账不改。使用 experiment-code 技能的固定输入/预算/错误持久化方法；按任务范围不训练、拟合、作图、下载、执行模型或物理。MAIN负责代码审读及主机CUDA/物理；没有派生后代。

## 覆盖delta，不重复已完成实验

上次实际 saved comparison SHA `c72cb6d96db7ae5cb6527e610145145d66b76c23d038b301efad047fc87ce5ee` 已消费原 moving 两根开环 CEM/local MPPI、四物理分支和RGBv2，并分栏238作者cells与本地结果。本轮不再重跑它或重抄无来源表。

`coverage` 新接口从该固定实际产物提取四论文完整238cells的作者值、literal/单位/±、version、table/page和原JSON pointer；本地GRASP bounded、LeWM上的CEM/MPPI/gradient结果另栏，所有两根都保留。每行完整协议/原HTML/PDF哈希/原文quotes仍可沿pointer回到上次实际矩阵；本轮新增的是**哪些算法已有哪种本地实现、协议哪些轴仍不匹配、下一次能补哪条接线**，不是新造作者成绩。无跨协议提升百分比、无官方复现升级。

| 方法 | 当前可消费内容 | 仍缺什么 |
|---|---|---|
| GRASP 2602.00475v1 | Tables1/3/4 p7/8/8，112cells；本地bounded两根 | 作者learned-model成功与本地native成功不同；DINO-WM、sweep、checkpoint/时域/完整预算未对齐；bounded不是官方完整复现 |
| Hi-LeWM 2607.12547v2 | Tables3/4 p13，106cells；已取正式loader/config文本 | 完整高层object未取得：v2真实probe `failed_preserved/Truncated range`；normalizer、嵌入低层和当前模型的坐标/参数配对未执行，不能把H10 final-goal fallback当learned generator |
| LeWM 2603.19312v3 | Table10 p26，8cells；本地原模型多solver真实结果 | 论文solver预算/误差语义/episode/checkpoint未匹配；目前原根开环sampling不提供相同观测prefix的后缀solver对照 |
| PlanWAM 2610.11382v1 | Tables1/2/3 HTML，12cells，PDF页unknown | 驾驶四相机/轨迹/EPDMS等非PushT动作与成功；另训练模块；不强造同协议成绩 |

Hi-LeWM本地asset失败报告 `results/p1_hilewm_bounded_object_fetch_20261027_v2_probe_main01/report.json` SHA `669bf41043dfae01a9bca183fe230d61baad29234c5cc73038c80409844e0b90`。本轮不重试网络，不拿partial当权重，也不把缺作者历史配对SHA当作禁止MAIN未来认证embedded-low的额外门槛。

## 选择的下一项：只补同观察的后缀solver

冻结旧 `moving_history_observed_suffix_cuda_20261027_v1` 输入构建器，源码SHA `3e72cef9375c5fa5b010593e2285e8eeb8d6b51b71310b70663b7f7406a7eb56`。它绑定实际原RGB8 report `7ec03c0a6a6e8411ff0dd0115044a872d099106edc95e4529b97c38c2b9b8f71`，全分母为slot0/1×GN/D2P已封存prefix5。只加载role=prefix5的原NPY，不读取terminal10 latent来优化。旧终点文件仅为输入闭包验hash；**不使用其数值、成功或pose决定候选**。

每branch新context严格为原latent_history[1:]加实际prefix192；动作history来自原warmup最后5与本branch实际prefix5，经原float32 mean/std算术拼为两个10维tokens。保留原goal、identity192 Q、packet logit球半径 `.75*sqrt(5)`、原raw native units，无第二次/100。每候选H1只从同一个实际观察context预测5步。

这补的是**控制prefix不变时的反馈CEM后缀基线**，不是完整CEM-MPC policy、更不是官方GRASP/Hi-LeWM。未来物理重放原prefix并验证actual prefix精确一致后执行新suffix，可与已完成bounded-ridge suffix物理逐prefix配对；不准继承旧endpoint。此前原根fullH2开环CEM不重复，已有live D2P反馈也不重复。只有两个known roots，freshness未认证，不称独立确认或P1通过。

## 在任何新CUDA前固定的算法与资源

每branch：NumPy PCG64；seed=`20261027+777+source_slot*101+prefix_arm_index*10007`。4轮×16 Gaussian候选=64；初始logit mean0/std1；每个10维proposal独立径向投影至原球。按本轮Q-cost稳定排序取top4，更新**投影后**logits的mean/std（ddof0），std裁至[.05,1]；无smoothing、incumbent注入、临时额外样本或按结果调参数。projection改变proposal分布，明确不声称exact density correction。

raw=`tanh(projected_logits.astype(float32)).reshape(5,2)`；物理派发前不再次裁剪。保存所有requested/projected bank、更新前后mean/std、64个原始cost、完整输出context。选全64的stable argmin，ties最早index。选择先封存，再对该winner独立重算一次H1；完整context及cost须MATH逐位相等，不相等就失败保留，不换赢家、不重试。old suffix不评价、不参与动态回退。

预算每branch65 predict API/65 batch1 predictor samples、65 action_encoder、65 pred_proj；四branch各轴合计260。195模块/branch，780模块总计。0AD/J/VJP/SVD/solve、新image encoding/physics/训练。AD=0由no_grad和无导数API执行路径排除，**未安装独立engine AD hook**；不把它说成已观测的额外AD-hook收据。每个预测3history tokens，共780 AE context rows及780 pred_proj rows；并非780 predictor samples。继承RGB8 encoder/projector和原source/planner/physical成本另记一次，不当免费。

相同的是prefix、目标、输入、动作支持、物理反馈时点和评分；**65 vs旧ridge5 predict不等预算**，旧ridge另有1J/192VJP/1SVD，不把各资源凑成等FLOPs。未来若要等算力结论，须独立预注册matched主轴/预算曲线，不能事后改本版本。

## 运行证据与失败保护

MAIN run只在显式authorization下先verify、再开独占输出；CUDA preflight后加载原模型，冻结所有参数，前后tensor hash必须与原模型一致。MATH/noTF32/noautocast，无CPU fallback、无新图像编码；encoder/projector调用设为禁止。每次AE/predictor/pred_proj保存原精度float32全部输入输出（不是只存终态），纯消费者重建非零tokens/真实context/AE→predictor→pred_proj→successor桥。每次candidate intent先落journal，模块attempt先于输入保存，return后记录数组receipt；错误时保留已尝试candidate、部分数组、模块attempt/return及四分母。执行首错和final model/source/journal/hash/report次错分列，无自动重试或清理。

输出≤16MiB，预留4MiB final report，磁盘floor200MiB；prepare≤2MiB（四个原192×192 Q显式保存），不复制上游报告/PNG/73MB资产。所有写入exclusive+fsync；模型权重仅读hash/MAIN CUDA加载，不下载。若完整native trace触达cap则失败保留，不能删数组放行。实际大小/prepare SHA以本轮最终交付消息为准；未有实际CUDA report不得声称planner执行成功。

## 下游consume合同

`consume` 输出schema=`agent_baselines_resume_20261027_next.consume`，status=`observed_CEM_commands_verified_not_admitted`，errors=[]，固定4 plans。

每plan：source_slot、arm=`observed_prefix_CEM64`、prefix_arm（原GN/D2P）、branch_id、parent_task_id、blind_sha256、parent_selection_sha256、physical_prefix_selection_digest、observed_context_sha256、原prefix NPY外部SHA、prefix_commands[5,2]、suffix_commands[5,2]、commands[10,2]、selected_index、model_Q、selection_sha256、whole_run_report_sha256。稳定键为(source_slot,prefix_arm,arm)，不能仅用slot/arm合并两种prefix。

NEW物理consumer尚未实现，是实际任务收益的明确前置。它须原recipe/seed重建root，不reset(savedpoststate)，原goal/RGB精确绑定；warmup和selected分账、每步RGB guard、prefix5像素/state身份精确匹配原观察、独立新terminal10，完整attempt/return/失败分母和close。native成功位置用agent+object前四维及角阈值，object-only仅诊断；reward/state7距离含速度，success不含速度。不得用modelQ或旧endpoint填真实收益。

## CLI（MAIN独占CUDA）

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_baselines_resume_20261027_next.py
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_baselines_resume_20261027_next coverage --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_baselines_resume_20261027_next_coverage_MAIN02.json
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_baselines_resume_20261027_next prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_baselines_resume_20261027_next_prepare_MAIN02.json
```

使用prepare命令实际返回SHA执行verify；只有MAIN审读通过才执行run，下面的`ACTUAL_PREPARE_SHA`须填真实返回值，不是默认占位seal：

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_baselines_resume_20261027_next verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_baselines_resume_20261027_next_prepare_MAIN02.json --prepare-sha ACTUAL_PREPARE_SHA
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_baselines_resume_20261027_next run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_baselines_resume_20261027_next_prepare_MAIN02.json --prepare-sha ACTUAL_PREPARE_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_baselines_resume_20261027_next_cuda_MAIN01 --authorization MAIN_OBSERVED_PREFIX_CEM64_CUDA_ONLY
```

待真实run后，`consume --output <实际run目录> --report-sha <实际report SHA>`，不预测未来SHA。coverage/prepare/verify及RAM mocks不执行神经/物理，也不算科学通过。本有界个人goal只到实现/测试/实际metadata/准备交付，不包含实际CUDA/物理或研究准入。
