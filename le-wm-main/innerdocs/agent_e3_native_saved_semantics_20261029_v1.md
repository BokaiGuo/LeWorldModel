# E3 native saved 语义：增量审计，不是 full strict

本独立文件只读现有出版物：先调用原 transport consumer 检查 report/completion 外部 SHA、日志哈希链、PNG/数组及资源账；再递归枚举 physical 记录内所有形状为 native body snapshot 的对象，逐个核 native7、双 body 的位置、速度、角度、力、扭矩、sleep 字段类型和 float64 bits。对保存的 warm10→original root→plan0 记录检查完整状态/body snapshot 精确相等，检查原 goal/root ID，cp5 和 boundary 10/15 encoder 使用完全相同的观察 receipt。动作分支检查连续派发、相应 return 的实际 raw bits、停止后不再派发，保留最多一个已派发未返回动作；失败、清理 intent/return/failure、launcher failure 和已完成 terminal row 只按实际记录核对，不虚构缺失事件。

本次真实 MAIN05：transport complete；32 observation、37 次 snapshot 出现（包含重复 receipt）、20/20 实际已记录控制派发/返回；warm10/root、plan0、cp5、encode10/15 链均匹配，cleanup return；新物理与神经执行均零。持久审计 `results/agent_e3_native_saved_semantics_20261029_v1_MAIN01.json` SHA `20470106ec6da1465171f6084d0124fbee4ecca210dda5ccbe4888281f06e67e`，真实 whole verify 成功。

这不认证捕获过程本身：日志未保存 goal bind 前后完整 renderer/goal RGB/RNG 对象的逐项状态；nested 身份不等于两个时刻动力学连续性或物理重演。失败语义仅重建所记下的派发/返回、异常和清理局部顺序；warmup 失败、中间 render/编码/源 guard 失败、transport 写失败和多重错误的全部可能路径尚未独立构造和完整消费，未记录的效果保留 unknown。fixture 是合成/已保存记录检验，不能证明物理发生。现有模型、loader、source guard、数值重放的完整检查仍由原 consumer 负责，本模块未重新认证这些；原 MAIN05 inspect 现在因原封存 P3 文件发生变化而 fail-closed，先前 inspect 的历史通过不等于当前复核通过。独立物理/神经 parity、fresh roots 与 scientific admission 均未证明；`full_saved_strict` 固定 false。

在 `le-wm-main` 下运行，只创建独占新文件：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.agent_e3_native_saved_semantics_20261029_v1 audit --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_chunked_identity_host_20261029_v1_MAIN05 --report-sha256 8ff1580f2c79482ecef10ec370367e0cc8f4b9b43e96ade0cbcea54913a5bb01 --completion-sha256 f12b2a20d46b7268ee8015e3848313d370bb6b095e9521ca3aef75033d990d28 --episode cell00 --saved /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_native_saved_semantics_20261029_v1_NEW.json
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.agent_e3_native_saved_semantics_20261029_v1 verify --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_chunked_identity_host_20261029_v1_MAIN05 --report-sha256 8ff1580f2c79482ecef10ec370367e0cc8f4b9b43e96ade0cbcea54913a5bb01 --completion-sha256 f12b2a20d46b7268ee8015e3848313d370bb6b095e9521ca3aef75033d990d28 --episode cell00 --saved /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_native_saved_semantics_20261029_v1_MAIN01.json --saved-sha256 20470106ec6da1465171f6084d0124fbee4ecca210dda5ccbe4888281f06e67e
```
