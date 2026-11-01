# E3 非03格 online 前置与 cell02 安全交接

本次只读保存结果，不执行 GPU/物理、不改封存或生产合同、不访问 03
claim。输出保留固定 known16 分母及 16 行；03 行仅列 resolver 固定身份，
标记未检查。本文件和检查脚本不是生产运行授权。

## 逐格检查

在现有覆盖清单的限定路径范围内，无 online terminal 的非03格是
02、05、06、07、10、11、14、15。对这八格逐一调用原
`agent_e3_control5_authority_join.load`：原 prepare 的固定 SHA、
报告/body、完成与 receipt 的连接、model/precision、保存 row、
source rehash 和原先科学 flags 检查均通过。每格报告、完成、receipt
的摘要来自现读文件；除了 02，未在本检查中提供独立外部封存锚点，
也没有重跑其上游 strict consumer。不能将数值 control5 当 online。

| cell | resolver task | 已发现的 feedback authority | 运行前仍缺 |
|---|---|---|---|
| 02 | r0.g0.h3.dDxy | LOCAL01；原 prepare 重建和 host source guard 通过 | 独立运行授权、来源封存、独占启动/重复检查及 host GPU/native runtime |
| 05 | r0.g1.h1.dDx | 本清单未发现 | authority、同等来源/保存 setup 验证及后续授权 |
| 06 | r0.g1.h3.dDxy | 本清单未发现 | 同上 |
| 07 | r0.g1.h3.dDx | 本清单未发现 | 同上 |
| 10 | r1.g0.h3.dDxy | 本清单未发现 | 同上 |
| 11 | r1.g0.h3.dDx | 本清单未发现 | 同上 |
| 14 | r1.g1.h3.dDxy | 本清单未发现 | 同上 |
| 15 | r1.g1.h3.dDx | 本清单未发现 | 同上 |

未发现仅限既有清单的命名范围，准备或运行前必须扩大检查，不能推断
全仓不存在。04、12、13 有原生前缀早停 terminal，不填造后续反馈，
不能因缺少节点把它们自动当成失败或未经授权地重跑。

## cell02 来源验证

authority 是
`results/e3_cell02_feedback_prepare_codex_20261029_v1_authority_LOCAL01.json`，
本次复核 SHA
`596014e1c53b4db19ba3dedc40877d814883e9f07b3b794aec80d7a28d387518`。
原 cell02 `verify` 重建整份 authority 并执行原 `host.authority` 的
source guard；新增检查再次核对固定 cell/root、H3/Dxy、control5
float32 动作、saved reset/setup、cp5 像素、authority 稳定性。历史
control5 receipt 并非实际同环境反馈。此 SHA 为当前 LOCAL01 的复核
锚点，后续需由独立授权流程确认，不因脚本中的常量自动获得生产权。

CPU 新解释器集成测试以真实 authority 和原 `module_census.snapshot`
核验已加载来源，并注入未封存模块验证拒绝；没有加载实际模型，不证明
CUDA 加载后或 episode 后 census 完整。检查脚本本身没有加入原
authority 的 source bindings，不得在与 runner 同一解释器中调用
`runner.run`，否则可能重现 03 的 census 拒绝。

## 安全运行交接（不是启动指令）

1. 明确授权 cell02 known-root 的一次 HOST CUDA/native 运行；保持原
   16格任务、source guard、early-stop、预算和科学门。确认没有同格
   进行中或已有可用终态；检查应包含更广的路径、进程和其他调度者。
2. 另行设计并审批独占 attempt 身份、claim 和输出；不能覆盖、删除、
   改名旧证据，也不能把 token 字符串或存在 authority 当成授权。
   claim 应原子创建并持久化，关联已批准的 manifest、来源、输入和
   输出；任何启动失败保留 claim 和 partial，不自动重试。
3. 事先独立固定并批准父进程入口及其加载模块、预检、子进程精确启动
   字节、解释器/导入环境与来源的 manifest。这个只读脚本不构成此
   manifest；不要现场计算哈希后自行称为封存。子进程必须与父预检
   隔离，让原绑定和 before/after census 如实执行；包括验证检查与
   读取之间的源码替换风险。原 runner 和原 authority 不做修改。
4. 协调主机 GPU，核对进程、GPU、renderer/native 依赖、磁盘和来源
   漂移；原 Store 创建前要求至少 512 MiB cap + 200 MiB floor，原
   wall 上限 1800 秒。不能以本次 CPU 检查、磁盘快照或锁替代协调。
5. 对 partial/异常/信号/缺出版物使用非零或明确未知状态，保留所有
   证据；对实际完成结果独立验证原 report、completion、journal、
   saved/native/source 与物理/神经 joins。publication complete 不是
   task success、freshness、full strict 或科学准入。

当前没有为 cell02 创建或启用新的生产 launcher/claim。上述父子完整
封存、互斥与失败状态交接仍是未来实施工作，不应直接运行旧 runner
绕过它们。

## 只读复验

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.e3_non03_host_readiness_codex_20261030_v1
env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/pytest -q -p no:cacheprovider tests/test_e3_non03_host_readiness_codex_20261030_v1.py tests/test_e3_other15_coverage_codex_20261030_v1.py tests/test_e3_cell02_feedback_prepare_codex_20261029_v1.py
```

只读脚本汇总中任何 `saved_control5_join_failed`、新的 terminal、
未归属作业或 cell02 检查失败均需要人工处理；不要仅凭其进程退出码
判断运行资格。当前样本结果为 8 格 saved join 通过、02 source/setup
验证通过；不表示 online 已执行或其他七格 authority 已存在。
