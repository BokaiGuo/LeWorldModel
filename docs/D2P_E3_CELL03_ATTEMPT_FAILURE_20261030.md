# E3 cell03 一次性运行的模块封存失败

主机 CUDA 一次性尝试 `le-wm-main/results/agent_e3_chunked_identity_host_20261029_v1_CELL03_MAIN01` 已出版 partial，report SHA256 `3f346697508d2254f7b57bf655ae53fdef1e8fa5f621f8a6c3bf4aaed8e428cb`，completion SHA256 `42e119fbbd50075e725ee2261e96d194b914e0507e44b7082f26a46d5b3abdf2`。

`events.jsonl` 的 `launcher_failure` 明确报告 `loaded module not presealed`，目标是 `e3_cell03_exclusive_host_codex_20261030_v1.py`。一次性 wrapper 在当前 Python 进程里直接调用原 runner，故 wrapper 自身进入模块清单，但已封存 authority 并未包含 wrapper。CUDA 模型加载和前置绑定事件已发生；模块审计在物理 episode 之前拒绝，本次不能作为物理或科学负结果。返回进程退出码为 0，但返回 JSON `complete=false`；不能以退出码代表成功。

这是新启动接线的工程失败，保留原 claim 和 partial，不覆盖、不删除、不将失败替换为成功。后续修复应在新版本中完整绑定实际加载模块，或使用经验证的新进程隔离方式，并增加覆盖真实模块 census 的测试；原一回合交接协议要求失败后先取得明确方向才能重试。已有的 CPU fixtures 未触及真实加载模块清单这一接缝。
