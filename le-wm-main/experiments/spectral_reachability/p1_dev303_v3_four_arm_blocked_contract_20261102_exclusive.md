# Dev303 v3 四臂前缀：阻断中的开发提案

本文件是事前开发提案，不是物理运行授权，不证明评分来源、账本、P1 或独立确认。几何候选不是谱程序。任何解封须另行制定并封存合同。

固定来源：模型报告 SHA-256 `e9bb449bcb44aff75df4061b867f6dacd9ebcb17202dea0c4b7fa0078fb6f15c`；事件 SHA-256 `5727e49cd70666c0b2aeed96447b97974672a3f2d35f925e1a951aa12b7ec35c`；动作合同 SHA-256 `fafd1d2de1d211af9166c6ff4966a61ecf29b83c78fda6e2e0a3bffaa7e44345`；模型运行合同 SHA-256 `213345b6680e7cd139768b50857180864a0d512213ba0da535ce77f5c3dfde35`。这些是本地观察的内容绑定，不是独立认证。

按顺序固定 selected=3、zero=4、task_greedy_direct=5、random=6；全部五步动作原值来自上述动作合同对应行，使用 float32 little-endian 字节哈希核验。task_greedy_direct 是既有 RGB direct-support 控制，不是 latent greedy。禁止基于物理 outcome 改选，允许不同角色动作相同。

拟议运行应为每臂独立新环境、相同 dev303 seed/reset/goal 和 10 步零动作 warmup，复用既有 0/5/10 帧与 reset 核验；每步记录 dispatch、实际动作、返回和帧，并将保存物按固定合同校验。发生异常、身份不符、终止或截断即停止该臂，保留部分记录，不补值、不替换、不重试。具体实现须先完成真实接口和来源闭包核验，不能将本提案当作已实现的执行器或保存核验器。

当前阻断：模型报告的 `score_provenance_verified=false`、`ledger_verified=false`。日志自身一致性不能构成独立 scorer 认证。必须先有可验证、与以上精确字节和冻结来源绑定的独立认证及其信任根，确认选择来源和账本，再单独核对 reset/帧/动作日志接口、写入安全与真实运行授权。现阶段不得运行物理。
