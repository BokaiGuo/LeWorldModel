# Matched-domain beam physical：独占纯saved consumer

只新增本stem source/tests/本文；已恢复并冻结253行physical-v1，绝不修改其三文件或MAIN产物。使用experiment-code固定源/预算、故障fixture流程；本次不创建后代、不运行模型、神经CPU、CUDA、物理或下载。不训练/拟合/调门槛，不按结果筛根。

## 独立入口

```python
from experiments.spectral_reachability import agent_kinodynamic_matched_physical_consume_20261028_v1 as c
receipt=c.consume(PHYSICAL_OUTPUT_ABSOLUTE, MAIN_EXTERNAL_PHYSICAL_REPORT_SHA)
```

先核physical source/tests/doc最终三SHA：

```text
631cb4e7894b2b4da0b96cca19af627a950133089e4ff571b4a1a6f1b9403a80
41c5b1db23123ff751f3847ba9efd2647843984403574e2e9a2e6895db3ebcbf
7b508cf7321cef1606f6a99bd3081565604a37317db76c7ceb80da92aa64588f
```

再核外部真实reportSHA，读取其physical prepare/prepareSHA，调用physical.verify重建原source/model/task/producer/256KiB以内prepare闭包及完整两根计划。这是纯saved own-consume，无模型加载；不把工程准备等于科学准入。producer报告只允许已封存新beam d4976aeca357b5324c74be8b2b7e3bfc254e82b834b3acfe1b0ab90a1ef4614c，不接旧rawbank或其他arm。

## 严格saved检查

- report新schema、固定2rows `(0,matched_domain_H2_beam20x6)/(1,同arm)`，不接受四臂旧schema或减少分母。完整/合法earlystop/工程partial都保留原行；不完整或损坏journal无法认证时拒绝，不填伪partial收据。
- execution_binding、run_started完整prepare、run_finished终态、journal全字节SHA、逐事件index/previous/eventSHA链与最终tail全部闭合。
- operation_dispatch_intent/returned/failed逐索引、原scope/selection identity、嵌套栈与全部actual receipts精确对应。intents不当attempts；可完整解释的returned/failed才重建actual账，与caps/counters严格相同；不存在缺return却伪报完成。
- warmup与own各原raw10×2、连续step、每个实际public return的state transition、每个原始命令、guard、真实0/5/10 checkpoint。完整44PNG每个都核文件SHA、224RGB、pixelSHA，不只查6个endpoint。invalid render的实际raw artifact也核SHA；目录严格白名单，不容额外/外国文件。
- 原真实7维goal/state绝不截5；独立重算原native distance为全7 norm、成功为前4维norm<20且原`min(absΔangle,2π-absΔangle)<π/9`。角度不暗加mod：Δ>2π会有negative native angle，wrapped与object诊断分列，不能以object-only成功替换原task成功。velocity进入native distance，object_pose仅[2:5]。
- 自身plan与warmup每一步原reward=-全7distance、terminated=原native_success、truncated布尔，末state/metrics/terminal与最后真实returned transition一致。warmup用placement_request中的原native goal，own用离线绑定构造goal；不把两个goal混用。
- 最后复核所有source/task/model/artifact/本consumer三文件SHA。返回全rows、实际checkpoints及png_path、counts/report/prepared/bindings_sha256/consumer_sources、fixed_branch_denominator=2、checkpoint_complete；后者只代表6个checkpoint齐，不代表科学准入。

step内部抛异常不保证存在故障后新frame；旧last_observation仅为最后已保存观测。该路径可能只有plan0及ownattempted1/ownreturned0；消费者不生成新state/PNG/reward，不把旧frame称故障现场。无重试/补帧/补步/重选/新预测。original native angle风险保持可见，不能“修复”封存env或success阈值。

## 命令与预算

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_kinodynamic_matched_physical_consume_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_kinodynamic_matched_physical_consume_20261028_v1 --output MAIN_ACTUAL_PHYSICAL_DIR --report-sha MAIN_ACTUAL_EXTERNAL_SHA
```

CLI只读并输出stdout，不创建MAIN产物/prepare/result；若MAIN要保存JSON须独立选NEW路径。输入沿物理32MiB全output上限与原metadata256KiB上限，consumer不创建大型tmp。没有实际physical report时仅RAM测试，不伪造reportSHA或声称actual consume通过。RAM BytesIO PNG fixture可验证文件/像素/chain与mutation拒绝，不能证明真实physics。

原两个known moving root不是独立样本，P1/scientific_admission/independent_confirmation/equal_total_compute恒false。仅原模型已选计划的描述性物理校准；P1/P2尚无科学放行。

## 本轮真实saved运行（没有新物理/神经）

只读MAIN已完成output `results/agent_kinodynamic_matched_physical_20261028_v1_MAIN01`，外部report SHA `01d5485208fe40a71a53ae20bb2920740894d6ef7c114c139a991f4707e5ebe7`，其prepare SHA `2430db7624f3d24ba80af78313b32ac2be168d45383a484f3aed91d5270fab43`。本consumer实际consume成功：2rows、6checkpoint、44真实PNG、218bindings，所有attempt/return计数相等（construct/close各2、reset4、setter8、goalrender4、goalbind2、public40、physics408、explicitrender44），native_successes=0。只输出stdout，未写MAIN结果或新研究证据文件。该结果不证明P1，也不是本worker运行了physics。

首次RAM端到端consumer测试发现两项fixture接线问题：重复newline，以及mock verify后的虚拟prepareSHA未提供给最终closure检查；均仅修本新tests，未放宽实际consumer门。旧physical-v1三文件始终保持上述冻结SHA。
