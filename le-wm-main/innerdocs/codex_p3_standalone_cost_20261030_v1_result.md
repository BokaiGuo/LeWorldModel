# P3 独立策略全流程成本：部分运行

本轮采用 `codex_p3_standalone_cost_20261030_v1.md` 的六个独立进程合同，前向、Jacobian、VJP、梯度、SVD 和 metric 分轴上限与先前在线预算一致；每槽独立 CUDA 初始化及模型加载，外部父进程计整个被启动进程的 wall。模型状态、GPU UUID、来源清单及每槽 journal 保存。固定 prepare `results/codex_p3_standalone_cost_host_20261030_v1_PREP_01.json` SHA256 `3116f4596f43a773807c2dde316c002f50eb5f7114ef569c78225f5f17b2da93`。

实际批次 `results/codex_p3_standalone_cost_batch_20261030_v1_RUN_01/report.json` 状态 `partial_or_failed`；20/uniform 成功，随后的 20/signal 在进入固定 inode 合作锁处遇到 `BlockingIOError: [Errno 11]`，发生在创建该槽输出和 CUDA 初始化之前。批次按固定 no-retry 规则立即停止，后四槽未启动；失败事件保存在 `events.jsonl`。不得把 1/6 槽当作公平比较结果。设备稍后的空闲快照不能抹去当时的锁冲突或自动授权改变此批次重试规则。

唯一完成并由独立 saved consumer 验收的槽是 20/uniform：

- 外部进程 wall 19.950086 s，覆盖 Python 进程启动、来源验证、锁及子进程终结；不含批次预备和后续独立消费。
- GPU child 生命周期内 wall 7.485537 s；CUDA preflight 0.228761 s；同步模型加载 2.564097 s；算法含 journal 0.767680 s，其中搜索事件区间约 0.752626 s，结构信号未执行。区间范围不同且有嵌套，禁止直接相加作无重叠分解。
- 模型加载期间 `/proc/self/io` 进程级 rchar 差值 195854350 字节，read_bytes 差值 0；后者是进程实际块读取统计，不代表没有读取模型、不可推断模型权重大小，且 page cache 不受控。
- CUDA 观察峰值 allocated 106384896 字节、reserved 111149056 字节。两个根各 12 候选；action_encoder/predict/predictor/pred_proj 各 48，metric 24；Jacobian、VJP、grad API、raw callback 和 SVD 各 0。已完成数及严格 saved 语义核验见 `results/codex_p3_standalone_cost_host_20261030_v1_CONSUME_P20_uniform_01.json`；该单槽通过本合同的分轴上限不代表跨策略公平。

内核、主机、saved consumer、batch 和测试分别为 `experiments/spectral_reachability/codex_p3_standalone_cost_{kernel,host,saved,batch}_20261030_v1.py` 和 `tests/test_codex_p3_standalone_cost_20261030_v1.py`。该测试与原预算内核测试合计 13 passed，使用模拟 backend，不能当作实际成本。源冻结、记录、独占输出与失败不重试用于保护实验边界；完整动态代码闭包、全部非合作 GPU 活动、加载前缓存冷热、FLOPs、能耗、外部环境噪声和其他策略独立时间仍 unknown。科学 gate 未改变。
