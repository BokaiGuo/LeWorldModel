# 官方 GitHub blob API 补充抓取

首阶段查询发现 HF 网络不可达、raw.githubusercontent.com reset，但 api.github.com 可达。保留首阶段源码/协议/响应不变，新增此 collector，经官方 Git tree 的 blob SHA 读取有限小文本。仍是只读查询，无模型/物理/transform fixture，不联系作者，不下载权重/数据，P1=false。

读取首阶段完整 report 和原始响应并验证 hash。对已选两项目 revision 抓取固定8个训练/评估文本；对两依赖 head 读取完整未截断 tree，匹配 pyproject 与 reader/transform 五个文件。每 blob ≤128KiB、API响应≤512KiB，最多32个GET，正文总限额继承首阶段已消耗字节，共≤8MiB。相同 blob 去重请求、保留全部 revision/path 边。校验 base64、size、Git SHA1(`blob <len>\0<content>`) 及 SHA256；不执行远程代码。

这些材料可证明公开 source-history 的具体合同，不能凭 head/上传时间推断 checkpoint 训练 run。无 checkpoint→commit/dataset/normalizer/camera witness 时继续 unknown。依赖 main 只作实现定位，不当作历史版本绑定。

```bash
.venv/bin/python -m pytest -q tests/test_p1_checkpoint_historical_contract_github_api_20261027_v1.py
.venv/bin/python -m experiments.spectral_reachability.p1_checkpoint_historical_contract_github_api_20261027_v1 --parent results/p1_checkpoint_historical_contract_lookup_20261027_v1_public_v1 --out-dir results/p1_checkpoint_historical_contract_github_api_20261027_v1_public_v1
```

首阶段和本补充都采用独占目录、执行前 source/protocol seal 和结束 hash 校验；不可修改已冻结 source。最终另写 review 文档，以 decoded 文件行号明确给出实证发现和未知边界。
