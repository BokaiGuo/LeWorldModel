# 两个历史sourcepack schema窄投影

独占新增source/tests/本文；不改旧manifest、resolver、MAIN、P4 v1/v2或authority表。
依experiment-code技能固定三输入、预算与失败分母；不训练/调阈值/采集，不打开四预留根。

## 真实代码收益与接点

旧moving_root_exclusion_manifest_20261027_v1.patterns('sourcepack')只读取rows；build又固定
container='rows'并拒绝observed schema。因此conditional实际slots→proposals→attempts的8行
全部不进入identity_records，observed的metadata.parent_source/random_reset_seed/checkpoints也不接线。
本模块复用旧Projection的纯lexical skipping，但定义自己精确allowlist；不修改旧patterns/global。
仅导入metadata模块，无producer/model/environment，无checkpoint/NPZ读取。

conditional：2slot×2proposal×2attempt全8行保留，原root_seed、reset span、每次random/placement
seed、state7及原root-goal hash逐pointer投影，不读取selected/status/cost/goal。state缺失保null，
不删失败/未选；state有值时按旧state_identity重算float64 state SHA及wrapped pose key。
缺snapshot/RGB3/raw10仍identity_complete=false，根hash未重新定义/未读goal重算。

observed：全2行保留actual reset、metadata root/parent-root、parent_source seed/span及三个checkpoint
文件/SHA/RGB hash/public_step描述。own root_state/root_state_sha/snapshot/history/raw/root_seed仍null：
parent seed仅留parent_source，不能冒认own seed；不从last_state/placement_requested_state/父state
填自身root，不把checkpoint RGB声明直接升级为独立验证history身份，不建立maps_to。
缺checkpoint保partial描述及unknown，不压缩2行分母。

## 原历史阻断与接口限制

第三输入是已封存inventory618 resolution metadata，不重读其引用文件。完整携带618来源的
path/SHA/inherited_refs/blockers、authority_requests及原summary；原480路径960引用、16 reset-bound
refs一个不少，original_unknowns_discharged=0。任何分母缺失/重复或reset refs变化拒绝。
这是新projection schema，不是MAIN approval/证书或旧identity_metadata，original_manifest_compatible=false。
未来新manifest实现可将conditional.attempts和observed.rows接入明确metadata适配器，但必须
同时携带inherited原阻断；禁止丢掉inherited只喂成功records来清unknown。旧sealed结果不覆写。
上述unknown未解、MAIN范围/原运行依据未核，source_run_preconditions_met/coverage_approved/
freshness_certified/confirmed/P1/P4/physical_release恒false。CLI exit0只代表投影工程成功。

## 只读运行

固定输入与外部SHA在source.PINS：原conditional sourcepack20KiB、observed14KiB、capture约3.1MiB。
cap分别128KiB/128KiB/4MiB，输出2MiB。无目录发现、follow引用、参数替换源、写结果或网络入口。
输入前后SHA检查，输出绑定三输入SHA/path和source/旧projector/lexical源码SHA。

```bash
env OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_p4_fresh_schema_projection_20261028_v1_tests.py
env OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p4_fresh_schema_projection_20261028_v1
```

MAIN可直接运行第二条，stdout为完整投影，不创建结果、不触碰四确认槽。本任务的RAM fixture
仅验证schema/拒绝路径，不是历史原证据；actual saved调用只读上述三份明确旧资产。

## 本轮实际结果

新27项RAM tests通过（0.23s）。已实际运行上述saved-only CLI，pipeline退出0：
conditional为2slots/4proposals/8attempts，8个state7实际存在，reset202781010..017全保留；
observed为2rows/6checkpoint descriptors，own state均null；618sources/480path groups/
960unknown refs/16reset-bound refs全保留，discharged=0。status为schema_projected_historical_gate_BLOCKED，
source_run_preconditions_met/freshness_certified/physical_release均false。
未写真实报告文件（完整结果仅stdout），没有改旧gate或生成证书；MAIN可直接复跑只读入口。
