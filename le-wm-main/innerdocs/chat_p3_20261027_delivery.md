# P3 独占交付：rank-reserve 工程对照

只新增 chat_p3_20261027_ 文件，未修改旧scheduler、结果或其他人的改动；无子智能体、神经训练、CPU神经降级、删除或推送。

已完整读取README及指定五份innerdocs，并实际调用旧scheduler自身consume核验原report SHA、prepare、源码/模型绑定、bank、520调用journal、全context replay与saved信号。
实际w=[162.32405169996156,156.2867146609139]，p=0.5094744711674455。
旧配额全10/10；旧winner各策略均[1,2]。权重非零但连续偏移小于half-up分辨率，
故同bank前缀造成执行碰撞，不是调度理论被否定。

新规则在任何新候选评分前封存，见外层 docs/chat_p3_20261027_prereg.md。
明确为新rank-only策略而非原比例公式的实现修复；放大微小差异/丢失幅度是限制。
真实saved权重形成U/N10/10、D15/5、S5/15；精确平权仍如实碰撞。
未通过筛根、温度、权重、预算或outcome搜索求有利结果。

## 实际验证

`tests/chat_p3_20261027_test_scheduler.py`：13 passed，纯标量队列fixture，非神经证据。
覆盖权重顺序、非有限/负/bool拒绝、平权、保底、总预算、前缀、N防火墙、失败不重试及缺失saved评分。
实际执行`chat_p3_20261027_scheduler seal`后，再执行saved_validation；后者重新构建并认证外部seal，独立消费旧saved候选。
U/N缓存覆盖全部队列；D/S各缺15项尾候选，paired_gain=null。
没有用旧winner宣称新15/5调度收益为零，也未为缺失尾部补模型成本或真实outcome。

封存SHA256：

| 文件 | SHA256 |
| --- | --- |
| results/chat_p3_20261027_queue_seal.json | 2f39483bceb87e638f74b2da170db8cf7ddbf6a1050df4137bbbcf9dde37ed78 |
| results/chat_p3_20261027_saved_validation.json | 8731d333c52ceb2eba143cd816ec96c278505d70d11e4843636eae0a7fb0ec98 |
| docs/chat_p3_20261027_prereg.md (外层) | 5e5391b569795fb735fd385ab80144dfecb90a19321232cb0378d174d3820c3d |
| experiments/spectral_reachability/chat_p3_20261027_scheduler.py | 9cbc3be52fe1d8c738ec192f5486cd108f6307a45b6c8d7cc8b313ebf0560aa3 |
| experiments/spectral_reachability/chat_p3_20261027_saved_validation.py | 6822121c885814200fe011fd0b1bd17b58aa6fc3c14641f2e827a66ec6497918 |
| tests/chat_p3_20261027_test_scheduler.py | 0641a664e9e17c7029e6c38a85b7520621a725df32967008fdf49d181f35ce2a |

## 成本与未完成项

新saved工作predict API=0、predictor samples=0、编码=0、AD=0、模型SVD=0；saved核验自身有数组SVD和wall开销，不是零总成本。
saved_validation JSON记录本次wall；继承旧共享8predict/2J/384方向/2SVD、历史8encoder/8projector和288anchor predict，不能免单。
旧predictor samples未在新程序内独立重数，明确unknown，不能直接用API次数替代。
新完整执行设计仍每臂120 search+8 replay、全panel共享8+512=520 predict，但尚未实际执行；无等FLOPs或wall优势。

当前交付是实际可运行的调度/队列/回调消费者和saved闭包验证，不是新的CUDA神经runner。
仍缺：新rank队列的主机CUDA执行适配、逐API/样本/模块durable ledger、精确whole/segmented回放、独立physical/RGB消费者；
两臂各15条尾部真实评分；P1通过receipt与认证fresh有序root/goal清单、输入闭包及完整预算。
这些是未完成的实现/科学依赖，不假称外部包不存在，也不将现有GPU描述为不可用。
未探测或启动GPU，因为本次完成的是model-free预声明与saved工程验证。
科学P1/freshness/P3仍关闭；尚未验证等预算真实收益，不称P3完成。

复核（cwd=le-wm-main；不覆盖任何产物）：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/chat_p3_20261027_test_scheduler.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.chat_p3_20261027_scheduler verify --output results/chat_p3_20261027_queue_seal.json --sha256 2f39483bceb87e638f74b2da170db8cf7ddbf6a1050df4137bbbcf9dde37ed78
```
