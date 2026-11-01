# 未来 P1 批次：外部时序绑定的 fail-closed 预检

只用于未来批次，不追认当前 spec。此工具不读取物理结果、不签名、不读取私钥、不运行 executor/NN/physics，也不自签可信时间。它核验字节哈希及两个外部 Ed25519 签名的**因果引用**，不是时间戳验证器或强制执行器。

## 外部职责与实际限制

MAIN 需在批次之前，通过独立可靠途径确定并封存外部 trust policy 的**实际文件 SHA**和两个不同角色公钥：独立 witness 与 launch authorizer。密钥身份、独立性、未泄露、可信时间戳/append-only 日志的信任与撤销由 MAIN / 外部治理核实；任意使用者自己生成 policy 和密钥再传 SHA 不能建立可信外部权威。工具只验证该调用指定的 policy 字节与密码学签名，输出始终 `trusted_timestamp_proven=false`、`dispatch_observed=false`、`outcome_blinding_proven=false`、`execution_authorized=false`。无正式可信 policy、凭据或外部时序记录必须 blocked，不能称前瞻通过。

固定 spec、physical prepare、executor source 原始字节及它们的 SHA，连同独占 run_id 和 policy SHA 交外部 witness，获得 commitment 签名；authorizer 的独立签名必须引用该 commitment payload SHA、同一 run_id、prepare/source/policy SHA。两个签名顺序只是逻辑引用，不证明真实签署时间、物理 dispatch 开始时间或分析者没有看 outcome。外部需要独立、不可倒签/回填的登记及审计证据，并确保实际启动顺序；已有执行器 `run_started` 不引用 spec，不经 MAIN 的受控启动与独立登记就可绕开本工具，故本工具无法在 executor 层强制 fail closed。要将此用作门控，MAIN 需在受控运行环境中要求每次启动前成功预检、校验唯一 run_id 和绑定的实际文件，并用外部记录关联真实 dispatch；禁止绕过直接调用旧 executor。未能提供该外部机制时不得宣称已实现强制时序绑定。

## 签名 ABI

所有 payload 以 Python `json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')` 精确编码，Ed25519 原始签名 base64。JSON 禁止重复键、额外字段。策略文件字段 `schema`（本 stem 加 `.external_trust_policy`）、`witness_public_key_pem`、`authorizer_public_key_pem`；必须是不同 Ed25519 公钥。以下所有 SHA 是小写十六进制 SHA256。

commitment envelope 为 `{payload,signature}`，payload 精确字段：`schema`（stem 加 `.commitment`）、`run_id`、`spec_sha256`、`prepare_sha256`、`executor_source_sha256`、`policy_sha256`。authorization envelope 同样 `{payload,signature}`，payload 精确字段 `schema`（stem 加 `.launch_authorization`）、同一 `run_id`、`commitment_sha256`（commitment payload 的 canonical 字节哈希，不是 envelope 哈希）、`prepare_sha256`、`executor_source_sha256`、`policy_sha256`、`purpose`（固定 `pre_dispatch_review_only`）。唯一 run_id 的跨批次防重及外部记录一致性须外部系统负责，本工具无全局账本。

CLI 仅在外部 policy、签名及所有明确 SHA 已获审查后使用；`check` 的输出是条件性预检，排他创建，不授权 dispatch；`verify` 重建并逐字核验。绝对路径示例：

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_agent_p1_external_temporal_binding_20261029_v1.py
.venv/bin/python experiments/spectral_reachability/agent_p1_external_temporal_binding_20261029_v1.py check --policy /ABS/POLICY --policy-sha POLICY_SHA --spec /ABS/SPEC --spec-sha SPEC_SHA --prepare /ABS/PREPARE --prepare-sha PREPARE_SHA --executor /ABS/EXECUTOR --executor-sha EXECUTOR_SHA --commitment /ABS/COMMITMENT --commitment-sha COMMITMENT_SHA --authorization /ABS/AUTHORIZATION --authorization-sha AUTHORIZATION_SHA --output /ABS/agent_p1_external_temporal_binding_20261029_v1_preflight.json
.venv/bin/python experiments/spectral_reachability/agent_p1_external_temporal_binding_20261029_v1.py verify --policy /ABS/POLICY --policy-sha POLICY_SHA --spec /ABS/SPEC --spec-sha SPEC_SHA --prepare /ABS/PREPARE --prepare-sha PREPARE_SHA --executor /ABS/EXECUTOR --executor-sha EXECUTOR_SHA --commitment /ABS/COMMITMENT --commitment-sha COMMITMENT_SHA --authorization /ABS/AUTHORIZATION --authorization-sha AUTHORIZATION_SHA --output /ABS/agent_p1_external_temporal_binding_20261029_v1_preflight.json
```

测试私钥仅临时生成用作虚构 fixture，不为真实研究提供权威。涉及任何真实批次，在 MAIN 指定外部 witness 和启动守门前，这只是可复核的前置组件而非可直接使用的完整时序证明。
