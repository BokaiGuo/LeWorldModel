# B8 physical independent saved audit

结论：程序/协议/绑定审核通过；不构成科学P1、stall、独立样本或物体任务成功准入。
本审计仅数组、JSON/gzip、PNG读取，无神经/物理执行，无tmp/prepare/result目录。
旧封存源码和结果未修改。新增test可打印紧凑机器JSON到stdout，无落盘报告。

## 固定证据

- B8 physical prepare SHA256:
  `e872fdbfef99ef47a00d48ea9263a590cc3f792dd9b9e3184cebdbb56aaad706`
- B8 actual `results/p1_b8_physical_20261027_v1_main01/report.json` SHA256:
  `e94a6cbf55dd9b408d350e690d17bfdb9ca693b2ef360a12ab904024f272cd7a`
- 旧actual `results/p1_witness_physical_20261027_v1_main01/report.json` SHA256:
  `48df27e54eca33159c44a834ce0be1a13723553dc9d64dde996b7b4ca9eaf747`
- 旧prepare SHA256:
  `e4fc7b7adeac3a23036196adaadfbbe69d8873f1e8649ebbf5fe9e8fb1eca0ff`

按root SHA+method配对，非JSON顺序猜配。root0=`02a13916...`，root1=`05844ca2...`。
旧物理GN/GD来自direct **B2=3**（不是B4）；下述物理收益是B8与该旧B3对比。
B4→B8模型损失来自同一实际B8模型轨迹的iteration4/8，不与旧B3混用。

## 审核范围/数字

新四臂：8 reset、40 public step、44 explicit RGB、12 PNG精确完成，无首错/次错。
独立重验全部journal hash-chain/首尾、各资源attempt-return事件分母、branch计划与收据、
原seed与sample_state placement请求、state/goal/root初态、逐命令raw精确、完整返回观测、
state7、official info和终止标志。fresh RGB/guard必须在下一命令dispatch前，所有11帧/臂
都有hash与完整guard历史。12PNG均224 RGB uint8，文件SHA、解码像素SHA、初态原RGB、
canonical agent pixels独立计数和0/5/10对应状态均核验。旧四个GN/GD臂另核40步/12PNG。

producer.verify只用于原source closure/task/gzip winner/full8J绑定认证；native/wrapped
误差、PNGS/轨迹比较由新test独立算，不调用producer.pose_diagnostics。模型原报告、
task SHA及winner digest核验，goal/Q/fullH2/normalizer/model语义与旧实际task逐字段精确相同。
完整绑定重建不重新执行模型、SVD或physics。

native angle负值步数0，native/wrapped success分歧0，terminated/truncated步数0。
四臂native与wrapped成功均 **0/4**。独立wrapped数值与producer atan2方式只存在浮点末位差，
使用2e-12 receipt容差；不是任何科学阈值。
reset隐含内部physics16/goal-render8已核，actual substeps仍unknown，不能冒称新实测。

## GN：两根与旧实际完全同轨迹

两根GN均：全部10条raw commands、完整trace（奖励/终止/obs/state7/contact/info/诊断）、
final state7、完整guard、全部11 fresh RGB hash，以及0/5/10 checkpoint states、
PNG文件SHA及pixel SHA逐值相等。不仅仅端点接近；新的task_contract/source语义仍不可
因为数值相同而重标成新增独立任务。

| root | 模型B4=B8 | 物体位置误差 | wrapped角误差(rad) | all-xy误差 |
|---|---:|---:|---:|---:|
| 0 | 1.1557897108702389 | 9.167073971082 | 3.017656615692 | 170.862167446648 |
| 1 | 1.9680302719088025 | 20.552924489553 | 1.804489538123 | 315.364254305093 |

GN0物体全部10步不变、无contact；GN1第4步contact，物体pose从第4至10步不同于初态。
GN1相对zero的物体位置误差 **增加6.861576637419**，角误差增加0.021976210547rad；
all-xy反而减少3.912221784429。因此不能用agent/all-xy改善掩盖物体误差恶化。

## GD8：agent改善，不是物体pose改善

| root | 模型B4→B8 | 旧B3→B8 all-xy | all-xy下降 | agent误差下降 |
|---|---|---|---:|---:|
| 0 | 1.241928755217→1.210409211497 | 176.102883079185→118.033391125431 | 58.069491953753 | 58.187252328821 |
| 1 | 1.799042981327→1.716405365543 | 257.991421263615→219.972978546428 | 38.018442717187 | 38.081388818301 |

GD8两根命令/实际轨迹不同于旧GD。但两根**每一步**物体x/y/angle与初态逐值相同，
10步无contact，物体pose相对旧GD/zero没有收益：

- root0物体位置误差9.167073971082、角误差3.017656615692rad。
- root1物体位置误差13.691347852134、角误差1.782513327576rad。

相对zero，GD8 all-xy下降124.184278087821 /99.303497543094。
native state7距离相对旧GD下降47.951650541676 /30.285192702582，但该距离混合位置、角度
和agent速度，不是物体pose收益，不能据此声称物体任务成功。没有本审计新增的actual-image
model cost；其他sampling RGB CUDA工作属于MAIN且不纳入本审核。

B8 GD模型仍在改善，不能把此前预算叫plateau/stall；GN4→8相等也不证明驻点/机制。
本批只有原两个calibration root family；independent_sample_increment=0，freshness unknown，
P1_admission/model_task_eligible/stall_certified=false。

## 复现

在le-wm-main，纯saved检查，不创建tmp/prepare/result：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p1_b8_physical_saved_audit_20261027_v1.py
# 可选stdout机器摘要，不落盘：
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -c 'import runpy; runpy.run_path("tests/test_p1_b8_physical_saved_audit_20261027_v1.py",run_name="__main__")'
```

6 passed。小型独立test与本报告合计远低于150KB；无结果资产复制。
