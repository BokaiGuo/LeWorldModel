# P2 frontier 真实物理：逐根预测／同Q／pose归因

2026-10-27；独立只读复算。只新增本文及独占tests，不改consumer、已封存源码、结果或共享进度。无下载、删除、提交、CUDA、physics、CPU神经调用或训练。这里只读取已完成的实际执行，不据outcome改Q、门槛、动作或根集合。

## 结论与success的含义

**8/8物理分支完整执行、8/8新终点编码完成；官方任务success为0/8，object-only success也是0/8。** `guarded_calibration_recorded`是完整校准记录，不是任务成功或P2准入。

conditional相对joint的pose在root0改善，root1/2/3损害，但三种损害不应归为同一原因：root1是模型与真实同Q都已预示的相对劣势；root2出现预测排序反转；root3同时有预测反转与真实Q—pose proxy错配。root0虽改善pose，也同时存在这两层失配，不能把其收益当作模型同Q排序正确的证据。这里是保存证据上的排序分解，不是对模型源码错误、因果机制或一般理论的证明。

## 输入与身份链

项目根为 `/home/gbk/code/LeWorldModel/le-wm-main`。固定输入：

| 输入 | SHA256 |
|---|---|
| `results/p2_composition_predict_frontier_physical_20261027_v1_main01/report.json` | `9144c3f25f2880af62d94dd88257918fa508b5ddbea51d98e9ccbc3d3b7532e2` |
| `results/p2_composition_predict_frontier_physical_20261027_v1_prepared01/prepare.json` | `ea9e30764c6f5a12e84e109987eedb3d425ed413d8af5cc66b884157035f37a2` |
| `results/p2_composition_predict_frontier_cuda_20261027_v1_main01/report.json` | `4023a0e3896a90407188d260a45dd334e0802ba17ecd59ef62ea931603eba173` |
| frozen consumer source | `281f6e8dd422bfa73ddff45ef0ca551ec7f6f1b23c35fca9835fde553f3f8110` |

按 `(root_sha256, method)` join consumer protocol、physical rows和encoding receipts；拒绝重复身份，不按数组位置配对。完整root身份为：

| root | root_sha256 |
|---|---|
| 0 | `9173218f71898716eac8fe0f3946b8e2cf20527ac71d84555c20c4e1b7fbf333` |
| 1 | `23b8de7837c39fb4e0fcffe7678732a9d840c57de7a0c8376e22076770ef6050` |
| 2 | `4afefa3be02a327aa69c22f38c17376d82340ba352aaf4aa29f95b070f5503ca` |
| 3 | `cf1b9bb913bf7815c9184d5de649b77bf09cc268dd80905a0feca9bd2a2edbf2` |

复算独立检查prepare/protocol字节、全部已绑定source/input当前哈希、planner report SHA、physical_report SHA、每臂journal/physical JSON/encoding JSON、全部88张PNG哈希与224×224 uint8 shape、RoyalBlue像素数、逐步原始命令、selection/metric身份与同根完整reset身份。未调用consumer执行器或其评分函数，也未重新编码图片。

## 逐根三层排序

所有cost越低越好。J=joint80、C=conditional40；Δ统一为C−J。Q始终是封存的同一root `source_Q`，`mean((Q@(z−goal0))**2)`，192维原始终点向量重新评分；不是拿不同Q比较。pose从保存物理state独立复算 `||object_xy−goal_xy|| +20*abs(atan2(sin Δθ, cos Δθ))`。

| root | 预测Q：J / C | 真实终点Q：J / C | Δ预测Q | Δ真实Q | Δpose | Δofficial distance |
|---|---|---|---:|---:|---:|---:|
| 0 | 1.391150635 / 1.322497960 | 1.680956094 / 1.744178515 | −0.068652674 | +0.063222421 | −70.717129670 | −91.433817539 |
| 1 | 0.775975544 / 0.894780844 | 0.895921772 / 0.907031882 | +0.118805300 | +0.011110111 | +42.863366495 | −55.933266233 |
| 2 | 1.209248872 / 1.076583990 | 1.227599034 / 1.247038926 | −0.132664882 | +0.019439892 | +15.783901578 | +125.362129638 |
| 3 | 0.895831125 / 1.029175309 | 1.717167150 / 1.344769875 | +0.133344184 | −0.372397276 | +47.447426780 | +90.574469063 |

| root | 预测Q偏好 | 真实Q偏好 | pose偏好 | 分解 |
|---|---|---|---|---|
| 0 | C | J | C | 预测反转＋proxy错配；最终pose改善并不验证预测Q排序 |
| 1 | J | J | J | **既无该对预测反转，也无该对Q／pose错配**；C劣势本来就被预测，不能事后称“本应获胜却失败” |
| 2 | C | J | J | **预测反转**；真实Q与pose方向一致，本对损害无需诉诸proxy错配 |
| 3 | J | C | J | **预测反转＋proxy错配**；模型预测J较优，但真实Q偏好C而物理pose仍偏好J，两层误差不能省略其一 |

因此预测Q→真实Q有3/4根排序反转（0/2/3），真实Q→pose有2/4根错配（0/3），预测Q→pose却有3/4根方向一致（0/1/3）。最后一个表面一致率可能由两层反向误差抵消而来，不能代替逐层验证。没有ties；本次按精确差值符号描述，不引入新科学margin或统计显著性阈值。root1实际Q差约0.01111，仍只是这一对保存样本的方向，不声称稳健泛化。

official distance是保存7-state全向量欧氏距离，包含agent及速度，不等于object pose。root1的C pose恶化42.86337却使official distance下降55.93327；不能把official改善改写成物体pose改善，更不能将距离下降当official_success。两臂所有official/object-only success仍全false。

## 实际完整性、成本与接触边界

全部8臂为同H2、各10 selected commands，attempts=returns=80；fresh endpoint均为full_H2，reset/context两臂逐值一致，无guard中止或执行failure。真实budget journal独立计数：

| 阶段 | env.step attempts / returns | env.reset attempts / returns |
|---|---:|---:|
| root collection | 0 / 0 | 32 / 32 |
| warmup | 120 / 120 | 8 / 8 |
| selected | 80 / 80 | 0 / 0 |
| 合计 | 200 / 200 | 40 / 40 |

合计240次公开调用产生480条attempt/terminal journal事件；这**不是480个env.step**。私有physics substeps/reset内部工作未被计量。encoder/projector各8 attempts=8 completed，predict禁止调用计数0；模型及source状态报告通过，并与当前绑定文件哈希一致。原数组复算不独立证明CUDA内核真正执行，也不重验tensor hash计算本身。

上游真实1328predict＋1328action_encoder、164Jacobian及其VJP/SVD开销沿用已封存账。joint80与conditional40可以在声明的predict-frontier上比较，但AD／候选数／总算力不同；不称等总预算。已有四根是重复使用的known-root calibration，不是独立确认队列。

**本审计没有接触连续性结论。** RGB guard仅证明采样图中RoyalBlue可见；终点agent-object几何贴近、分离或单个时刻接触均不能证明10步整轨迹接触。保存trace含离散state，但本次没有重建逐帧polygon间隙、contact manifold或内部碰撞冲量，故不能把任何损害归因为“全程无接触／全程接触／持续滑脱”。即使另做离散几何审计，也不能代替内部substep接触证据。

## 可执行检查与mock／实际／准入区分

新增 `/home/gbk/code/LeWorldModel/le-wm-main/tests/test_p2_frontier_physical_attribution_20261027_v1.py`：

- 7项小型mock测试覆盖两层符号分解、真实劣势不误报、ties、顺序置换、重复root拒绝、NaN拒绝；只是算术／接口证据。
- 1项实际saved-data集成测试独立核对上述8分支、逐步journal/88 PNG/编码数组、真实同Q和物理pose／official标签，并打印完整精度逐根归因；没有运行神经或physics。
- 实际运行合计 **8 passed**。数值一致检查容差仅用于浮点复算，不是新科学准入margin。所有P1–P5 admission仍false。

复核命令（无输出目录，不复制大资产）：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -s -p no:cacheprovider tests/test_p2_frontier_physical_attribution_20261027_v1.py
sha256sum tests/test_p2_frontier_physical_attribution_20261027_v1.py /home/gbk/code/LeWorldModel/docs/p2_frontier_physical_attribution_20261027_v1.md
```

剩余前置条件：正式P2仍需原P1依赖成立、预先冻结独立根／确认队列、足够覆盖的真实机制与任务成功证据，以及包含AD/历史构造/反馈编码等完整资源账的强baseline比较。若要解释具体接触机制，还需独立版本预先定义的轨迹几何／碰撞证据；当前不能凭末态补证。没有发现本轮足以授权修改sealed consumer或提出“修复即可成功”的具体接线错误，也不将本结果提升为一般理论否定。
