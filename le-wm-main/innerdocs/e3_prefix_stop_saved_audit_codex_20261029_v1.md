# E3 前缀早停：保存终态一致性增量审计

仅saved-only，不改变旧源码、结果、E3协议或科学门；不调用GPU/物理/神经。新代码
复用现有transport consumer与native saved semantics，再核对前缀早停终态的动作
float32位串、第一次stop、最终实际观察、状态、保存success与指标的一致性；拒绝
未来checkpoint/plan观察及任何反馈编码或数值事件。`terminated/truncated`与
`native_success`分开，不由早停推导成功；不独立重算native成功predicate。

## 此次限定资产盘点

在 `results/agent_e3_*` 已保存报告路径集合内，16个control5数值HOST receipt存在；
它们不证明实际物理反馈。当前发现7个不同格的物理HOST terminal：

| 格 | 保存terminal | 本次范围 |
|---|---|---|
| 00 | `completed20`，success=false | MAIN05与MAIN06是同一格两次运行，不计两格；历史MAIN06已有单独语义审计。 |
| 01、08、09 | `completed20`，success=false | 本次只抽取row与launcher状态，不重做full strict。 |
| 04、12、13 | `native_stopped_prefix`，第一控制步terminated，row success=true | 本次新增审计对三份真实保存出版物完成transport和所述终态连接检查；无控制5之后的反馈边界。 |
| 02、03、05、06、07、10、11、14、15 | 本次清单内未找到物理HOST终态 | 不推断它们失败或从control5数字receipt补造终态；这只是有限路径清单，不证明全仓绝无其他资产。 |

报告 `completed` 意味transport publication完整，并不等于完整任务运行或科学准入；
三早停格为原native提前终止，不能说三个执行失败，也不能把这些记录称完整反馈。
MAIN01–04为source/module census类partial工程失败；不代替任务失败结果。所有
上列HOST报告仍`full_native_strict=false`、`scientific_admission=false`。所有16格
均无本工具所需的完整科学验收回执，不能据此声称E3完成。

## 实际读取绑定与结果

相对 `le-wm-main/results` 的固定三份：

| 目录 `agent_e3_chunked_identity_host_20261029_v1_...` | report SHA256 | completion SHA256 |
|---|---|---|
| CELL04_MAIN01 | `7b9db580a2d4ff62268776701f53dbeb885c25a3ccabfde934337f305c802319` | `4894b09fb91f9a54ada82116c824d1fc42ad3d1e7fe4176314cb051fbd0d2dd4` |
| CELL12_MAIN01 | `83a7dcc994ea3c409f04128a7e2a06968af68eefcac40777ebc5ba6f4eafc517` | `2912cd833995ee2a617f7c42a59e2e06ef4213e0dca5bee38f50094cd8de157b` |
| CELL13_MAIN01 | `6dd7e26048fbfa0eb7d2accd6c10883acf38cdb282a7ba6718cbfde1294a5bf9` | `1cacd69f298eeb49ac425f8a58eb69f41106c26b4bc2b23165cbba0f2288e2ee` |

三个stdout-only audit均成功：returned_prefix=1，terminated=true，truncated=false，
recorded_native_success=true，feedback_boundaries_completed=0。其success仅交叉检查
保存row/metrics一致，并未从物理状态独立重算；模型/loader/source guard未重新
认证，未做physical/neural parity，也不能据此补出所有失败路径。transport consumer
核验指定report/completion、日志链和本出版物登记的文件；其通过不等同full strict。

调用模板（将三个实际SHA配对代入，路径使用绝对路径；只打印stdout）：

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.e3_prefix_stop_saved_audit_codex_20261029_v1 --output ABSOLUTE_OUTPUT_DIR --report-sha256 REPORT_SHA --completion-sha256 COMPLETION_SHA --episode cell04
```

本次新增测试与既有native semantic测试联合19 passed；fixture测试仅验证拒绝逻辑，
不能证明真实环境异常路径完整。未写生产receipt，未调用P3。未来应按原协议
处理剩余9格及source/model guard、完整失败语义和parity，须另有适当运行权限。
