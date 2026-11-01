# E3 known16 其他 15 格：只读覆盖清单

该清单使用固定 resolver SHA
`b44699e6a10bb4c257ed8872dc86b99a37508d7022850f8b4fcd257bcd232b26`
核对 known2 × goal2 × horizon2 × domain2 的顺序与 16 格身份。
分母仍是 16；cell03 在本任务中排除，不读取其同名 production
目录或 claim，不把另外 15 格改称完整 E3。此工具没有运行入口，不创建
结果文件，不调用模型、CUDA 或物理环境，也不改变合同。

## 当前有界结果

范围是 `results` 下固定 resolver、每格固定 control5 路径、列出的
feedback authority 命名模式，以及 `agent_e3_chunked_identity_host_*`
输出目录；不保证全仓所有可能的别名或其他 runner 已被发现。

| 格 | control5 保存 receipt 哈希连接 | feedback authority 引用文件哈希连接 | online HOST 保存状态 |
|---|---|---|---|
| 00 | 是 | 是 | MAIN01–04 partial；MAIN05–06 completed20，均记录 10、15 节点；六次尝试只占一格 |
| 01 | 是 | 是 | completed20，记录 10、15 |
| 02 | 是 | 是 | 有界目录中未发现作业 |
| 04 | 是 | 是 | native_stopped_prefix，无反馈节点 |
| 05、06、07 | 是 | 未发现 | 未发现作业 |
| 08、09 | 是 | 是 | completed20，记录 10、15 |
| 10、11 | 是 | 未发现 | 未发现作业 |
| 12、13 | 是 | 是 | native_stopped_prefix，无反馈节点 |
| 14、15 | 是 | 未发现 | 未发现作业 |

共 15/15 保存 control5 receipt 与报告/完成文件的 SHA 连接；8/15
发现 feedback authority 且其四个引用文件哈希匹配；7/15 有 terminal
记录，其中 4 格有 10/15 节点记录，3 格前缀早停。不是所有 authority
都重新做了 source guard；receipt 及 authority 的 SHA 是现读快照，
不是本工具独立批准的外部 pin。数值 control5 不是实际在线反馈。

online 输出先使用原 transport consumer 对当前 report/completion
快照和 journal 做校验，再抽取记录的 cell、terminal 和节点。
因为 report/completion SHA 是现场计算而非独立提供的外部锚点，
它只证明当前出版物内部一致，不能证明历史不可篡改。节点仅为
记录，不是重新验证实际图像与数值/物理 parity。未重建完整 source
guard、native strict 或所有失败语义；早停不是自动失败，publication
completed 也不表示任务成功。cell00 partial 中可能已出现节点记录，
不能拿它们填补其他 attempt 或当作 terminal。

## CPU 复核

在 `le-wm-main` 执行以下只读命令：

```sh
env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.e3_other15_coverage_codex_20261030_v1
env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/pytest -q -p no:cacheprovider tests/test_e3_other15_coverage_codex_20261030_v1.py tests/test_agent_e3_transport_publication_20261029_v1.py
```

命名范围中的损坏或无法归属作业列入 `unassigned_or_invalid`，不因
缺失被视为任务失败。后续对尚无 authority、未发现终态或需要严格
复核的格，应先核对更广的已有资产和原合同，再按独立授权推进；
不得从清单推断可以启动新生产作业。
