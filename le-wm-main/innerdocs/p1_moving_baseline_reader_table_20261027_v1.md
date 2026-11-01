# 六基线逐根读者表生成器 v1

仅新增本source/tests/innerdoc三文件，v3及其它sealed产物不改。实现worker只RAM/mock tests与CLI help，不执行真实表生成、ownconsume、artifact重算、神经、物理、训练、图表、论文或外部检索。MAIN负责实际运行及复算。experiment-code技能用于固定输入/预算/输出合同，用户限制优先，不启用其训练或图表模板。个人goal完成仅指实现交付，非P1/P2完成。

## 冻结合同与输入边界

仅从MAIN外部SHA批准的两个严格JSON输入：v3 manifest与v3 comparison。CLI强制两路径和两SHA；SHA格式、绝对路径、regular nonsymlink文件、重复JSON key、NaN/Infinity及文件大小约束继承冻结JSON工具。comparison.manifest必须逐字段等于传入manifest绝对路径/SHA；manifest自有schema通过v3.manifest_check，comparison schema/status及implementation binding必须是冻结v3。v3源码SHA锁 `506d4abc6e962e2374ccbc813c0d8b953275cecba9a5df741c5dc71c1a3b3477`。

本工具是研究结果展示assembler，不是第二次科学/源码/原artifact审计：不调用v3.run、不重建v3结果、不执行ownconsume/build/verify、不读取PNG/NPY、sourcepack/witness或大模型。依赖MAIN已经审读/运行/复算的v3证据。两JSON中声明的模型/物理/RGB来源、state和cohort必须相符，固定task ID/SHA不可换；只做展示所需的格式、接线、分母和布尔/数值合同检查，不声称这些检查独立证明原始实验真实。

已读Jason完整handoff `innerdocs/p1_moving_extended_baseline_rgb_huygens_handoff_20261027_v1.md`。其新8图合同是18 artifacts/56 events，不能冒充旧6事件接口；该原始复算由冻结v3和MAIN负责，本reader只接v3的实际RGB账，不再自行加载/编码图。MAIN已提供actual extended RGB report SHA `8e72ca50d43a5460330dafff8b07568e204ffb43fdd3e0b53d8b9d6906b929a9`、prepare `f094c5c28dbc6c157e6cb2ef014bd5825826dbe55a6fc614b101482055ed1411`，不以handoff文档本身充当actual authority。

## 六基线、两根与非等价声明

固定顺序：moving_gradient_open_loop、CEM、kernel_faithful_mppi_nominal、fixed_H2_kinodynamic_beam16x4、GRASP_H3_bounded、joint_H2_spectral_packet_mapped。每臂slot0、1，共12行，不按native/Q结果删根或臂，无均值替代、置信区间、显著性、总体成功率或排名。原两根仅known-root calibration，不是独立确认。可接受v3事先纳入的原GN/D2P额外两方法，但本读者表始终展示六baseline；所有输入cohort账仍保留，MD明确说明不是按结果过滤。

所有实现标本地、非官方复现、不等总算力/搜索域。nominal MPPI非官方强MPPI；GRASP bounded moving是机制扩展，H3为3历史帧，未来H2，不是完整官方等价；kino constant-packet离散库与D2P逐时刻logitball不同，全库枚举不能称自适应预算优势；SRA joint-bank后packet投影改变支持域。仅展示本地绑定结果，不引用任何未核外部数值、reported异协议成绩或SOTA。

## 输出与数值语义

NEW输出目录固定三个小文件，合计≤256KiB，逐写留200MiB，不覆盖既有目录：

- `reader_table.md`：中文逐根结果表、逐根模型资源表、物理cohort表、来源完整SHA及边界说明。
- `reader_rows.csv`：六臂×两根12行，完整float repr精度、task ID/SHA、model/physical/RGB状态、来源cohort ID与两输入SHA。model Q/model-root Q、actual root/prefix5/terminal10 Q、native布尔、full10布尔与own returned steps分列；每项带status。
- `reader_cohorts.csv`：每个模型invocation、每个物理cohort、每个RGB cohort各一行；另加唯一一行共享原encoding8/8。完整来源路径/SHA、整次latency及其状态，物理reset/set_state/goal_render/goal_bind/public_step/physics_step/explicit_render/environment_construct/close与RGB编码attempted/completed分列。没有分摊或重复累计。

missing/partial/unmeasured一律输出字面 `NA`，不是空串、0或false；状态列保留差别。只有真实actual布尔false才显示 `false`，只有实际meter或已冻结零合同才显示0。缺模型不显示modelQ，缺物理不显示native/full10，缺RGB不显示actual prefix/terminalQ；原root Q即便可算也仅沿用v3已经发布的来源状态，不用modelQ替代actualQ。native成功与Q改善、完整10步不同。各cell必须类型正确、有限、非负；counter必须非负整数且completed≤attempted。

模型资源逐根保留predict/action_encoder、predict API、backward、grad API、Jacobian操作、Jacobian API、SVD、bounded solve、pred_projector的attempted/completed，另列ADforward子集和mathematical reverse计数。未meter项NA，不因为算法相似补0。ADforward已包含于predict，不加第二次；predict API与predictor模块hook可能指同一调用；backward/grad/VJP/Jacobian计数可能重叠，不相加。GRASP reverse计数是scalar loss sweeps、SRA是输出VJP方向，不能直接当等额反向计算。若v3只保留raw predict_API计数，本reader按原meter转列，不推估其它API。

每根物理返回步数与整个物理cohort的public/internal physics调用不是同一字段；后者只能在cohort表显示一次，通过physical_cohort_id关联。不能将同cohort120 calls复制到六行再求和，也不能除以根/臂数发明单根实际meter。模型/物理整次latency不均摊；RGB没有独立整次latency就NA。共享继承原8encoder/8projector只一次；standalone各算法的8/8成本视图不能求和为共享编码，端点评价RGB另计。

## MAIN真实链与可执行命令

MAIN已报告v3实际综合12model/12physical/24RGB，六臂各0/2native；本worker只读核对schema/status/manifest指针，没有重跑综合或独立复算。读者表不得将该两根结果写成总体失败率/理论否定。

Python根 `/home/gbk/code/LeWorldModel/le-wm-main`，实际输入为：

- manifest `/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_baseline_comparison_20261027_v3_manifest_main01.json`，SHA `388143639950875317be2f5d25ebaeb8b6b9c1c409290c92763dc10ceef600d5`。
- comparison `/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_baseline_comparison_20261027_v3_main01/comparison.json`，SHA `96cf27b143bb1a839edf7bc1e4de02f4a97e433c6475c4c731c78af76300d32e`。

MAIN运行（本worker不实际生成）：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_baseline_reader_table_20261027_v1 --manifest /home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_baseline_comparison_20261027_v3_manifest_main01.json --manifest-sha 388143639950875317be2f5d25ebaeb8b6b9c1c409290c92763dc10ceef600d5 --comparison /home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_baseline_comparison_20261027_v3_main01/comparison.json --comparison-sha 96cf27b143bb1a839edf7bc1e4de02f4a97e433c6475c4c731c78af76300d32e --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_baseline_reader_table_20261027_v1_main01
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_moving_baseline_reader_table_20261027_v1.py
```

stdout返回各输出完整路径/SHA及字节数，P1_admission/equal_total_compute=false，不输出或改写新的scientific report。

## 实现测试范围

31项RAM tests，复用既有纯数组fixture定义、不读取实际研究artifact进行再综合、不生成tmp/prepare/result。覆盖12行完整分母、浮点精度/逐根保留、actual native=false与missing不同、missing/partial模型和物理NA、无RGB不填Q0、AD不重复/反向单位区分、cohort物理账仅一次、共享编码8/8仅一次、未知API NA、输入不变；错schema/task/source/manifest路径及SHA/实现绑定/分母/资源/NaN/布尔类型均拒绝；输出cap与写前200MiB floor拒绝。--help可用。科学准入/统计外推、独立根、官方等价和总算力公平性仍未解除。最终三文件SHA见交付消息。
