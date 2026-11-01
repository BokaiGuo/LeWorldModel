# P3 wrapper/ledger/IO非神经复现

仅独占新文件；不改任何冻结producer/summary，不消费生产report，不运行torch/NN/physics。

## 源码定位

- effective `run`共用一个model，每node factory构造`native.NativeProgram`。
- `NativeProgram`继承`NativeSignal.observe`；后者保存每个module的原instance forward属性，
  在finally逆序恢复，原无instance属性则del。不能据单调wall猜测wrapper累积。
- `Session.plan`每node创建新`core.Ledger`；start/end均为同一clock减去该node起点。
- `core.Ledger.call`先记started，然后写resource_attempt sink，再执行fn，记ended，
  最后写resource_return sink。inclusive duration包含attempt日志与body；不含该次return日志。
  外层signal_compile同时包含内层操作及其日志，不能把inclusive各轴相加当总耗时。
- `effective.Store.check`每次对整个output递归rglob，逐项is_file及每文件stat，并检查disk free。
  它扫描文件metadata而非读取所有文件内容；计数通常随文件数增长，不等同按总字节线性增长。
- 实际episode用`live.NativeJournal`；每event先runtime check（实际run的check含store.check），
  再按payload长度store.check，然后write/flush/fsync。每个resource_attempt/return均走此路径。
- 实际Store.write除check/write/fsync/目录fsync外还返回整目标文件SHA；继承emit追加主journal时
  因而可能有已增长journal的重hash成本。native events stream本身每次只hash该record，不能混称全文件hash。

## fixture

`observe_fixture`直接执行原observe上下文，fake model/FakeInput无torch/神经；
多node共用fake model，核inside包装深度1、before/after0、精确instance属性恢复；
同时覆盖预存forward与body异常。

`io_fixture`只在pytest临时目录，继承原effective Store.check/write/put/namespace，
并调用原live.NativeJournal和core.Ledger。PathProbe只统计原rglob/stat过程，不额外扫描。
逐node增加小metadata文件，三个event（attempt/body/return）各执行两次check；
断言扫描次数相同而遍历entries/stat工作增长，不断言wall单调。
真实flush/fsync不模拟；读取真实journal核逐record/hashchain和LF，无过滤空行。
另用显式synthetic clock核attempt sink100+body7=107，return sink10000不计该op，
每node起点重新置零。该算术fixture不是生产时钟读数。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m pytest -q tests/test_agent_p3_journal_overhead_fixture_20261029_v1.py
```

## 结论边界

本fixture证实原路径可以随output metadata增加付出更多真实IO/检查工作，并验证原restore与计时口径；
不证明这些开销解释生产2s→11s的比例，也不排除GPU同步、payload大小、fsync/storage波动、
JSON/blob编码与deepcopy、真实模型计算等因素。不得将增长自动认作纯计时错误或wrapper bug。
若需定量归因，由MAIN另行授权同路径分项profiling；本任务不修改运行算法或生产预算。
