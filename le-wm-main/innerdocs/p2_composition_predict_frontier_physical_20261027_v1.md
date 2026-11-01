# P2 predict-frontier：独立 fresh physical consumer v1

仅NEW模块、tests与本文；不改任何sealed上游。main实际执行，agent只写代码、运行NumPy/PNG/constant-interface fake测试及只读saved-data验证。没有CPU神经、CUDA或physics运行，没有训练、重选、裁剪或准入。

## 唯一上游与身份

固定真实报告 `results/p2_composition_predict_frontier_cuda_20261027_v1_main01/report.json`，外部SHA256为 `4023a0e3896a90407188d260a45dd334e0802ba17ecd59ef62ea931603eba173`。消费NEW runner的 `selected_actions`：joint全80槽、conditional全40槽；不是旧40-slot exporter，不继承任何旧物理endpoint。独立导出JSON不是必要输入：consumer直接调用同一封存NEW exporter，验证真实report/execution/prepared、全部bank/journal/witness、winner、模型计数和AD开销。其saved-data SVD/solve验证属于额外CPU算术成本，不是新模型调用，也不算免费。

`planning_material`要求外部报告SHA精确匹配，导出后再次绑定字节；按 `(root_sha256, arm)` join，不依赖export数组序。四根×两臂、完整H2十个raw commands、source root完整latent/action context、原goal0/Q/metric SHA、raw域/normalization、模型policy、RGB来源逐一校验。逐个winner按全部edge的 `program_id` 重建完整saved witness；joint须一个10-command edge，conditional须两个5-command edges。独立float32 normalize重建H2 action history，不拿重复receipt相等替代token算术。保留新bank/row/provenance及1328predict、164J等真实上游资源；不把predict-frontier称为等总算力或P2通过。

复用已有source-only references仅用于初始四根/seed/三帧RGB/goal/context/evaluator合同；其endpoint不参与本模块。main提供的真实结果选择动作已固定，不读取新物理outcome调参或重排根。执行顺序固定root_index，并交替两臂先后。

## 物理与失败证据

四根两臂一次，共80 selected公开env.step；每臂fresh reset后15 warmup，共120 warmup，预计完整执行200次公开step。collector按现有源码只reset、不step，其实际reset attempts/returns和warmup reset分账；不能把80当全部physics，不能将模拟器私有substeps说成已计量。

`PhysicalMeter`逐公开调用记录attempt/return，collector step cap0、warmup全局120/每臂15、selected全局80/每臂10。失败attempt不退费；warmup严格核对实际15次返回。每臂初始state/goal/RGB/history同根一致；官方evaluator源码绑定。所有raw command前RoyalBlue official uint8 224×224 RGB guard，复用durable逐步ledger；PNG/journal fsync与文件SHA。发生guard失效或termination停止该路径，调用/完整性错误阻断后续分支并补blocked census。磁盘检查在render而非代理step内，未发真实step不能误计物理attempt。

partial保留尝试/返回及最后观测，但只有journal最后RGB对应实际返回、attempts==returns、状态可观测才是fresh endpoint。step/render异常的陈旧图不编码。branch/reset/evaluator、journal/hash/write、close及最终source/model错误分别保留，不覆盖首错。

先持久化全部 `physical_report.json`，main再显式 `--encode-terminals`：至多8个新鲜终点，各一次HOST CUDA encoder+existing projector；predict禁止，不进行AD/重编译。冻结eval模型、同一weights/config，前后tensor hash；每次encoder/projector预记attempt，完成才增completed，异常写terminal error event，失败journal错误另存。先保存完整192向量，再计算原Q分数，预测/实测不反馈选择器。只有8个完整fresh分支、两hook各8次attempt==completed、无失败且source/model未变才记录完整校准状态。

四条稳定根paired描述给conditional-minus-joint pose/official，仅同reset、同H2且双方无失败完整时算差；partial/blocked保留，不混入完整对比。所有P admission恒false。

## 小产物与主线程命令

prepare只复制静态小源码，协议保存8个selected witnesses和每根一份Q。上游约73MB报告/journal/bank和模型权重仅哈希引用，绝不拷贝。prepare估算严格小于20MiB，创建目录前要求可用空间≥200MiB keep +100MiB run reserve +估算prepare；不降低门槛、不自动删资产。执行启动仍要求300MiB；每次render另检查200MiB+16MiB。并发外部写入不在本线程控制内，main需统一调度。

预计prepare约5–15MiB（以只读size核验为准），physics/编码通常小PNG及192向量，预留100MiB而非声称实测体积。consumer不复制上游大JSON；所有失败产物保留。测试使用224×224 dummy RGB，无真实环境或神经调用；覆盖真实 `planning_material` 的只读实际源报告接线、全witness program_id重建、identity/partial/计账/编码失败。

main review源码后统一prepare；agent不prepare或执行：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p2_composition_predict_frontier_physical_20261027_v1.py
.venv/bin/python -m experiments.spectral_reachability.p2_composition_predict_frontier_physical_20261027_v1 --prepare --planner-output results/p2_composition_predict_frontier_cuda_20261027_v1_main01 --planner-report-sha256 4023a0e3896a90407188d260a45dd334e0802ba17ecd59ef62ea931603eba173 --output results/p2_composition_predict_frontier_physical_20261027_v1_prepared01
sha256sum results/p2_composition_predict_frontier_physical_20261027_v1_prepared01/prepare.json
.venv/bin/python -m experiments.spectral_reachability.p2_composition_predict_frontier_physical_20261027_v1 --verify --prepared results/p2_composition_predict_frontier_physical_20261027_v1_prepared01 --prepare-sha256 CONSUMER_PREPARE_FILE_SHA
.venv/bin/python -m experiments.spectral_reachability.p2_composition_predict_frontier_physical_20261027_v1 --execute-physics --prepared results/p2_composition_predict_frontier_physical_20261027_v1_prepared01 --prepare-sha256 CONSUMER_PREPARE_FILE_SHA --output results/p2_composition_predict_frontier_physical_20261027_v1_main01 --encode-terminals
```
