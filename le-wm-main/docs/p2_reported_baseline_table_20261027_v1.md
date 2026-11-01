# P2 reported-baseline table sidecar v1

本模块是已有核验文档的有界导出器，不是第二份文献 registry。
复用 `d2p_reported_baselines.ReportedBaselineRegistry` 的逐条校验及可比性分类；
所有数值从既有文档的指定表格单元格读取，保留原 token、行号、原行、SHA256。
不读取任何本地 calibration/outcome report，不产生新实验。

## 可用的表

`reported_only.json/csv/md`：24 个已核验论文表值。PlanWAM v1 的
NAVSIM-v2 EPDMS 90.9、NAVSIM-v1 PDMS 93.8，及 HUGSIM RC/HD-Score
各五列（四难度与原文 Avg.）；HWM v2 文中 DINO-WM / DINO-WM (hier.)
在 Push-T 三档 d 下的六值，以及 PLDM / PLDM (hier.) 在 Diverse Maze
三档 D 下的六值。不是原论文完整所有方法表，也不是根据胜负选择新增数值。
只导出旧核验文档已经确认的字段；未核验的其它原表方法不从缓存临时扩填。

`official_project_only.json/csv/md`：AW-0 四值、AVIC 网站十八值，合计22。
这是作者网页报告，不进入仅接受 paper_reported 的 registry，不升级为论文PDF表值。
AVIC/AVIC-R、Direct/MindJourney 和 MapGPT 对照分别保留；WM单位不映射为predict
samples，K tokens 不变换单位。AW-0 rubric score 不是二元成功率。

`reproduction_readiness.json/md`：19项外部方法/系统的精确来源与接入缺口。
GRASP、Hi-LeWM、LeWM、SAGE、SuperMimic、PRISM、IMWM、AP-CEM/AP-rank、
Frozen HWM Subgoal、CGS、Dual-WM、Slot-JEPA、HarnessPAI、WorldForge
不录入性能数值，不将 unknown 当0。AP附件值绝不转录。
`report.json` 还保留旧忠实性审计的9类本地对照要求原行，以及预算、训练、观察合同原行。
这些是已检查来源的缺口，不声称穷尽机器资产，也不声称旧审计后没有任何新接线。

## 语义与证据边界

- 外部性能与本地校准不混表，不排名、不求跨论文均值、不计算增益或配对显著性。
- 相对明确的Push-T环境reference，NAVSIM/HUGSIM/Maze为context_only；HWM Push-T为
  insufficient_metadata：相同任务名不能补齐checkpoint与协议。reference其它字段保持unknown，
  不构造一个虚假的完整本地合同。DINO-WM/PLDM与LeWM的backbone差异另列在接入缺口中。
  全部只作外部文献参照，不是本地复现或直接可比结果。
- PDF页码、checkpoint revision、seed数、不确定性缺失保持 unknown；原文表号与版本独立保留。
- PlanWAM的4秒trajectory不是4个LeWM预测token；registry的整数horizon保持unknown，
  原文秒数、64候选、训练LoRA和多摄像头合同通过原行保留，不推算模型调用数。
- HWM的d/D是任务距离，不是规划horizon。d=75的flat/high/low预算只以其明确适用范围
  原行与结构化level保留，不推广到d=25/50；Maze Table13/14预算仅绑定D=[13,16]，不代表总算力匹配。
- HUGSIM Avg.是80/157/96/103场景加权的原报告值，不重算未舍入均值，不把436当训练seed。
- HWM缓存头部日期与正文日期不一致的旧记录不被清除；锁定v2不声称最新版本。
- source seal只绑定所读文档/代码字节及第二次读取稳定性，不认证论文真实性或实验时序。
  九份核验文档的预期SHA写入导出器；任一变化即拒绝，不静默接受新文献数值。
  本次非新primary verification；既有原文核验记录才是转录依据。原始缓存只曾只读抽查，
  不被当成网络重新核验，不改变旧来源状态。

## 执行与冻结

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_p2_reported_baseline_table_20261027_v1.py tests/test_d2p_reported_baselines.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p2_reported_baseline_table_20261027_v1 --write
cd results/p2_reported_baseline_table_20261027_v1
sha256sum -c SHA256SUMS
```

`--write` 固定新独占目录；已存在即失败，不覆盖。退出0仅表示导出完成，P2_admission始终false。
每份来源上限2MB，只读固定allowlist，拒绝路径逃逸和source symlink。
未使用网络/下载/GPU/physics/神经CPU/训练；仅新增module/test/doc/results。
测试证明转录、分类、边界与封存行为，不是基线性能复现。

本次非神经联合回归：新增27项 + 现有registry56项 + baseline protocol v2的38项，
共121 passed（0.40s）。新增模块前测试按预期ImportError；预算scope、来源漂移和flat
不能继承hierarchical反馈合同均先出现失败回归，再实现通过。
