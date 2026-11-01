# P4 fresh certificate：具体上游需求与消费接口

独占source/tests/doc，旧audit、authority和results均保留。使用experiment-code的事前接口/输入记录原则；仅saved metadata与RAM测试，不执行模型/物理/网络、不打开四个保留根，不读取NPZ/sourcepack，不按outcome挑根。

## 已确认的具体缺口

读取并绑定v2 audit source SHA `3242a7d1d32c6667563588744bc153c6a5ec3f21d2274762ed07385239549092` 和真实report `results/agent_p4_fresh_certificate_audit_20261028_v2_report_MAIN01.json` SHA `15e3ece87e36eba469a0cc2188efbb7edaa94fa00f1562f9e6adc2052e4e4ae0`。report仍是 `blocked_no_legal_fresh_certificate`。

1. root/consumer manifest实际都没有 `history_certificate` 键（不是已经存在的显式null字段）；freshness/source_run均false，不能生成证书。
2. coverage中A–H的8来源、16继承引用、10已知dispatch：缺原运行reset尝试/重启全覆盖、原加载closure/穷尽包络及origin citation。B的3次不能只保留最后一次。次数unknown保留null，不按报告缺项推0。原authority提供两条可替代路线，不要求两条都交。
3. 历史scope：machines/worktrees/time_interval/unsaved_failures/origin_citation以及cutoff后delta处置；Oct27固定catalog不是全部后续接触史。新执行不能补旧历史。
4. exclusion approval.external_file=null、approved=false；必须由MAIN绑定具体manifest_content_sha256/cutoff_utc/coverage_scope。保留975 unresolved occurrences的全部record和准确JSON pointer，不删减、不称975独立run。
5. 历史前置闭合之后，独立root流水线还需pre-witness实际state7/wrapped pose/snapshot/RGB3/rawhistory/task-goal身份、历史和批内排重、access/disclosure与顺序证明。本次不访问保留槽或声明这些字段已有。

identity_inventory保全618条metadata记录的path/SHA/blockers/pointer；其614缺直接身份refs与closure保留960继承refs不是新独立run，也不能与975/16相加当总样本数。family_requirements准确转录legacy_zero/original_four/overlay/feedback的补证状态：original_four inventory_paths为空，overlay5条prepare、feedback1条prepare只算收到inventory，仍需MAIN范围裁定。所有引用路径只作为需求输出，不打开其内容。next_delta/sourcepack/prepare/NPZ等未读引用在other_audit_bindings_not_read明确分列，不伪称已核验。

两个construction NPZ只继承audit的report-level引用；本实现不打开/重核其成员，不能将它们变成fresh same-task/history certificate。inventory/closure只是metadata关联，不是实际根映射的认证。

## 输出与既有流水线对接

`request`生成机器可读需求，包含每项 `evidence={path,sha256,pointer}`、`field_present/observed/missing_fields/responsible`，完整16ref/10dispatch/975缺项分母。source内部PINS列明全部9个实际读取文件的路径/SHA；不会follow它们引用的原report、checkpoint、日志、NPZ或保留根。source/tests/doc也绑定SHA。

嵌入的 `authority_request` 直接调用已冻结 `agent_freshroot_authority_20261028_v1.requests`，不重造补证schema；其source SHA `f7a30fcad18483e1848170acdd1fe4bd96e7d08b38804b18f1cf2cf1087870e7` 事前核验。`authority-request`可单独导出原schema，独立流水线可用现有response协议与 `validate` 接收真实原始补证。validation即使成功也不代表MAIN批准/历史事实已认证。

`consume`重建当前固定输入闭包及需求清单，篡改/SHA漂移拒绝；正常返回 `blocked`，`source_run_authorized=false/reserved_root_access_authorized=false`。exit0仅表示blocked请求可解析、未篡改，不能当run gate通过。本版本无解锁分支；真实新authority获批需MAIN另封存后续版本，不原地改pins。

## MAIN命令（inner根）

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p4_certificate_requirements_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p4_certificate_requirements_20261028_v1 request --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p4_certificate_requirements_20261028_v1_request_MAIN01.json
# 将ACTUAL_REQUEST_SHA换成上条真实输出；没有未来SHA默认值。
.venv/bin/python -B -m experiments.spectral_reachability.agent_p4_certificate_requirements_20261028_v1 consume --request /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p4_certificate_requirements_20261028_v1_request_MAIN01.json --sha ACTUAL_REQUEST_SHA
.venv/bin/python -B -m experiments.spectral_reachability.agent_p4_certificate_requirements_20261028_v1 authority-request
```

最小外部下一步：取得A–H原运行方已有ledger或原加载closure+有限穷尽包络，再交MAIN裁定范围/真实性；若历史不可恢复，明确unknown，不造证书。没有收集新证据、签发approval/certificate或打开reserved roots的权限。RAM/schema测试不是freshness/P1/P4科学通过。

实际验证：本模块25项RAM/saved测试通过，联合既有authority38项共63 passed in2.35s。已实际完成临时文件CLI request→consume，结果blocked；旧authority对空response保留8个missing sources、0 cleared，未创建实际补证response/approval。所有真实读取只在固定9输入与自己三文件白名单；未follow任何原报告/NPZ/权重/日志/保留根引用。MAIN持久results输出尚未创建；上面命令由MAIN实际执行。
