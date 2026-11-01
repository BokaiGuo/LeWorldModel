# Six RGB roster：纯RAM接线核心

仅 `assemble(prepared,native_receipt,paid_report,get_paid_trace)` 和低层 `vectors`；get_paid_trace(i)返回真实trace字典与原数组key→ndarray。无文件读取、consumer、模型导入、图片解码、评分或writer。对应当前Pasteur原生six schema，固定24位置，plan0是真实cp0，earlystop未到达保null，actual-token whole/seg完整history逐byte核。不是完整认证器，输出明确upstream_full_authentication_performed=false、host_ready=false；调用方生产前仍须独立全量native/paid消费。

RAM回归命令：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_six_target_rgb_roster_20261029_v1.py
```

实际49passed/0.29s。人工构造的HOST字段只用于字典ABI mutation，未调用任何模型/环境，不是实际HOST evidence。覆盖全部24、早停0/1/5/7/10/14/15、whole/seg/实际raw signedzero、完整history、同时间frame/state/response和错误拒绝。不独立认证原生done/reward/模型公式/外部receipt SHA，也不发raw部署许可。由后续RGB adapter重建真实native/paid authority后使用。原source/tests/doc不修改。
