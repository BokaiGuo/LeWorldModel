# P3：预声明有效对照及实际等预测预算验证

结论：原配额退化已被新命名的rank-reserve工程对照消除；本固定队列没有观察到模型或真实任务收益。
这不是旧比例调度的追认修复，也不是科学P3通过。P1、freshness、P3和compute/wall优势全部保持false。

上一轮 `le-wm-main/innerdocs/chat_p3_20261027_delivery.md` 是当时仅有队列/saved缺失评分的历史快照，原样保留。
本文件与实际run/consume/accounting是后续完成状态，不覆写早期结果。

## 1. 碰撞链与预声明

完整读README和指定五份innerdocs；实际调用旧scheduler ownconsume认证原report、prepare、模型/源码/数据绑定、bank、journal、候选与回放。
旧w=[162.32405169996156,156.2867146609139]，p=0.5094744711674455。
旧连续配额约[10.0947,9.9053]，half-up后D/S与U/N全[10,10]，不是零信号或漏派发。
另已只读核实原comparison-RGB status为saved_comparison_RGB_joined_not_admitted，
joint-horizon report为cuda_completed且P1/P3=false；不将其1066predict或mixed-H结果混进本fixed-H2预算。

新规则先封存：总20blocks、各根保底5、每block3候选。D把余10给严格高w根，S反向；精确平权含双零均分并记录碰撞。
固定U/D_rank/S_rank/N顺序，固定root/goal/Q/bank与索引前缀，原zero/GN/D2P锚点保留。
rank-only会放大微小差异并丢失幅度信息，这是限制，不称理论上最优。
只根据模型内冻结信号分配，不根据模型候选质量或真实outcome改权重、温度、根、预算、候选顺序。
执行时重新计算全图joint J；信号需精确复现已封存工程信号，否则失败保留，不改队列重跑。

## 2. 已实际执行的有效干预

| 策略 | root0/root1 blocks | 实际候选数 | 独占predict API | 实际predictor batch samples |
| --- | --- | --- | ---: | ---: |
| U | 10/10 | 30/30 | 128 | 128 |
| D_rank | 15/5 | 45/15 | 128 | 128 |
| S_rank | 5/15 | 15/45 | 128 | 128 |
| N | 10/10 | 30/30 | 128 | 128 |

每臂120 search predict+8 whole/segmented独立回放；没有去重或退款。
两根U/D/S实际消费的候选前缀均三者不同，不只是配额receipt不同；N与U刻意相同以披露信号开销。
所有候选包括新tail都实际执行；全部成本、稳定argmin、独立回放和真实520个全context/module链由ownconsume重建通过。
运行后无改参数、加预算、换根、去anchor或重试。

## 3. 零收益完整保留

两根四臂winner仍分别为index1和index2；每根literal commands均相同。
模型Q各臂分别为0.6003295827228584、0.1660668051225332；D-U/D-S/D-N逐根和均值均0。

真实物理使用本轮producer的8个selection seals，独立构造/复现moving warmup后执行全部8分支，未复用旧终态。
8/8完整10步，0/8 native success；每根四臂真实终态完全相同。
两根native distance分别105.2180881741788、508.4131455293894；对象位置误差分别39.788662801735335、36.78867149910163。
预声明native_distance/native_success/wrapped_allxy_success/object_position_error/wrapped angle的D-U/D-S/D-N逐根差均0。
不能由此推断总体调度理论无效；这是两个known roots、有效资源干预但相同winner下的工程零效应。
不将8分支当8个独立根，不报告p值/CI/泛化收益。

## 4. 完整资源轴

新CUDA总520predict API、520 predictor batch samples、1560 context tokens；action_encoder也520调用，pred_proj是1560 flatten rows，不能混作predictor samples。
共享只付一次：8predict/8batch samples/24tokens、2J、384数学VJP、384grad API、384raw backward callbacks、2SVD。
引擎内部backward/kernel/FLOPs未知；saved复算自身的数组SVD/hash/I/O另有开销，不等于模型调用。
新visual encoder/projector=0；历史8encoder/8projector、288anchor predict和完整历史AD receipts在prepare/accounting里保留。
独立部署逻辑视图（非额外实跑）：U/D/S各136predict+共享AD/SVD，N128predict不读信号；没有独立部署wall实测。
所以这里只有等预测API与实际样本预算，不是等总计算或等FLOPs。

CUDA实际wall 28.22814436489716s，peak allocated141664768 bytes/reserved144703488 bytes；全部低于预声明600s/3GiB/4GiB限额。
每资源嵌套inclusive墙钟和每arm实际scope span已保存，不能相加当总耗时或当FLOPs。
物理dispatch lifecycle wall 11.419749494059943s；prepare/saved验证/accounting有自己的CPU/I/O开销，不冒充含所有阶段端到端wall。
物理实测construct/close8、reset16、set_state32、goal_render16、goal_bind8、public_step160、physics_step1632、explicit_render/PNG176。
本新run约11MiB CUDA+3MiB physical；未放宽16MiB cap/200MiB floor。

## 5. 交付与证据定位

- 规则：`docs/chat_p3_20261027_prereg.md`；seal：`le-wm-main/results/chat_p3_20261027_queue_seal.json`。
- 实现：`le-wm-main/experiments/spectral_reachability/chat_p3_20261027_{scheduler,cuda,physical,accounting}.py`。
- 四个独占测试文件共50 passed（6.10s）；只覆盖RAM合同，不冒充CUDA/物理真实性。
- 实际CUDA：`le-wm-main/results/chat_p3_20261027_cuda_run01/`；ownconsume：`chat_p3_20261027_cuda_consume01.json`。
- 实际physical：`le-wm-main/results/chat_p3_20261027_physical_run01/`；ownconsume：`chat_p3_20261027_physical_consume01.json`。
- 完整配对/分层资源报告：`le-wm-main/results/chat_p3_20261027_accounting01.json`，含全输入/源码/模型/artifact SHA闭包。
- 可校验交付SHA清单：`docs/chat_p3_20261027_delivery.sha256`（从仓库根执行`sha256sum -c`）。

关键SHA256：

| 产物 | SHA256 |
| --- | --- |
| queue seal | 2f39483bceb87e638f74b2da170db8cf7ddbf6a1050df4137bbbcf9dde37ed78 |
| CUDA prepare | 72a02d66c99c9037cd55828bf67342514ec110a3310ced194477ef6e34c432f1 |
| CUDA actual report | 6df856e36b145162e6a8d5ce5f141880a49afe88f43d3cc88f8b98ad790b6d08 |
| CUDA ownconsume | e50c07141c3b41e7026d291fc2cf0eb95b15ff79f080c34632793ff74ff002f7 |
| physical prepare | 724a8f1684e17c79dbc3705f55fc712b9c500e800cf019f854490cd58350c3c4 |
| physical actual report | d0f3b5b61dc66d1a87c2f940fe4035fa259f7f7d43c2161097500e047fa21cf5 |
| physical ownconsume | 60aa73d27f65219849a9ae61da2ccfc133341bd9d3ca7ec6a25a8e77dbf44e6f |
| accounting | 0dd4914498db991d1a5978d413ca71c64fdddd2212fce048bf4aeec3d67437e1 |

## 6. 完成核对与严格边界

已证：碰撞链、固定total/minimum/tie规则、先封存后评分、不同实际前缀、同root/goal/bank/预算、
可执行CUDA/物理队列与独立saved消费者、全部50测试、实际520账/样本及8物理终态、零收益不筛选、SHA完整交付。
只用主机CUDA运行神经模型（RTX5060Ti，原模型hash前后相同）；未训练、未CPU降级、未放宽任何误差/科学门槛。
全程无子智能体；仅独占新前缀文件/目录；旧源码、用户已有改动与所有封存结果保留；没有删除、推送或改写共享登记。

仍缺但不能由本工程goal消除的科学依赖：P1通过receipt、认证fresh的有序root/goal清单及输入闭包、独立fresh配对确认。
可选真实same-Q/RGB编码及其ownconsumer未做，其值为null；不能拿native pose或模型Q替代actual latent Q。
本goal完成“消除调度碰撞并交付预声明有效对照”，不代表取得正收益、完成P1或打开P3科学准入。

## 7. 不覆盖产物的复核命令

cwd=`/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/chat_p3_20261027_test_scheduler.py tests/chat_p3_20261027_test_cuda.py tests/chat_p3_20261027_test_physical.py tests/chat_p3_20261027_test_accounting.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.chat_p3_20261027_cuda consume --output results/chat_p3_20261027_cuda_run01 --sha256 6df856e36b145162e6a8d5ce5f141880a49afe88f43d3cc88f8b98ad790b6d08
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.chat_p3_20261027_physical consume --output results/chat_p3_20261027_physical_run01 --sha256 d0f3b5b61dc66d1a87c2f940fe4035fa259f7f7d43c2161097500e047fa21cf5
```

不建议自动重跑run；现有结果完整，重跑不是本交付所需，也不允许按零收益修改策略后覆盖。
