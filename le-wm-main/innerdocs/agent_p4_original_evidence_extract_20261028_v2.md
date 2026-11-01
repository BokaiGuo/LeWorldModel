# P4 async original evidence v2

独占新source/tests/doc，冻结v1不变。使用experiment-code的输入/回归记录原则；仅saved日志解析，无神经/物理/网络/子agent。复用v1的纯stdlib parser、traceback sanitizer和coverage loader，显式绑定v1源码SHA `5f2288bafdc03e4717dc3734a3e8970fc62649ec667276874a9480a0810aaea6`。

## 精确关联，不按顺序猜归属

只用coverage `results/agent_freshroots_resume_20261027_coverage.json` SHA `00a3fffeaaa8943e851d129b941a7f07feaa1db387117beb98110e4ec95f5eeb` 绑定的同两日志prefix（路径、bytes、SHA完整写入report.bound_logs，后缀不读）。A/E/H原dispatch行分别58279/58307/58363，直接返回行58284/58309/58367，其中有3/3/2个明确session_id。这些session只归属父call，不分配到父脚本的某个subprocess或task。

后续仅认可实际函数输入：top-level `text(await tools.write_stdin({session_id:LITERAL,...}));` 或直接function_call write_stdin JSON参数，和同call_id的返回。安全词法解析，不eval JS、不执行日志命令；字符串/注释/代码patch/循环/变量/条件语句内提到数字不是执行证据。多poll以及其他exec混在一父call时，terminal结果没有session_id便只留batch级，不能按output位置配对。唯一literal poll+唯一结果可由明确call-ID桥关联；返回明确session_id则逐字段关联。跨日志相同数字不默认同一executor namespace。

保存每session出生origin、poll input origin/span、poll-call返回origin，以及sanitized terminal exit/traceback/严格ledger候选。origin带日志prefix SHA、字节数、JSONL行/record byte span/record SHA；JSON-in-string另带text SHA与decoded pointer。traceback只保留File frame；ledger仅接受 `{call_id,attempts:[index,reset_kind,disposition,identity_sha256]}` 严格元数据形状作为候选，不认可exhaustive。stdout中的score/outcome/state数组不复制、不读取其中路径。

## MAIN命令（inner根）

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p4_original_evidence_extract_20261028_v2.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p4_original_evidence_extract_20261028_v2 extract --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p4_original_evidence_extract_20261028_v2_report_MAIN01.json
# ACTUAL_SHA为上条真实输出，不预填未来值。
.venv/bin/python -B -m experiments.spectral_reachability.agent_p4_original_evidence_extract_20261028_v2 consume --report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p4_original_evidence_extract_20261028_v2_report_MAIN01.json --sha ACTUAL_SHA
```

report与consume永远source_run_authorized=false、reset_blockers_cleared=0，不自签approval/certificate。不打开保留根、NPZ/sourcepack、stdout路径或当前模型/环境文件；不能将局部terminal恢复说成历史穷尽或freshness通过。new_neural_calls/new_physics_calls=0只描述本次extractor成本，不表示原历史作业未执行神经/物理。

## 实际saved关联及未闭合原因

真实prefix已验证：A父返回sessions71462/37004/43033，后续literal poll位于58298、同call_id返回58301；E父返回29686/39278/18657，poll58316→58319；H父返回42594/59270，poll58374→58378。三个poll调用自身call_id分别为 `call_5cb6ad88921e4250a6178bf0a6de28b6`、`call_eb4cd41fb234487cbfda479412914e79`、`call_30cc69f80fcb4be99c55c0e74cc3b7e7`。

8个session全部有明确literal poll输入。但poll-wrapper返回分别4/4/2条带exit_code的engine结果，共10条，均为exit0且没有session_id；其中夹有其他exec_command返回，不能把10条结果称8个poll已终结，更不能按位置确定哪个是P0/P1。report保留全部10条batch级terminal observation及origin，逐session terminal_observations仍空、reset_attempts仍null、subprocess_source_assignment仍null，具体reason为no_unique_terminal_result_session_join。未从这些结果恢复traceback或reset ledger候选（观察数0，不是历史不存在的证明）。没有读取新后缀或扩大到session全部生命周期，exhaustive/loaded-byte/approval仍未成立。

实际新27项回归通过（6.82s），包含wrong-session、return顺序/重复、父日志namespace、文件读取白名单、冻结v1依赖hash、冷进程禁止neural/physics/numpy导入、原生origin字节/JSON-in-string复取与真实saved extract→临时report→consume。另一次真实只读extract smoke确认8session/3batch/10未归属terminal/0joined terminal。早先联合回归44 passed in8.29s（当时v2为25项）；最终联合数以交付实跑为准。回归只检验提取接线，不是科学准入。MAIN正式results报告未由worker创建。
