# E3 GD HOSTCUDA native数值节点与完整saved consumer

NEW独占三件套；冻结preparation/resolver及旧工具不修改。known2、非fresh、非Full，所有科学门false。worker只做RAM fakeCUDA/临时磁盘，不生产prepare/NN/physics/fullstrict。

当前actual16inputs只有control5资格。此版本真实执行固定16格各自control5节点（每格独立job、全16无替补/去重），总2176predict/512scalar reverse，不将其冒称完整48节点6528/1536。真实control10/15必须来自每格部署后的新RGB；目前无权威不能猜schema、不能拿imagined context补齐。完成本波也不签闭环/E3；后续反馈部署接线另需MAIN给权限与真实authority。

## 算法与原生图

新solve独立实现固定0init、eta1、32梯度、33完整候选、stable(cost,index)、3独立winner replay。只复用冻结mask_chart/project纯数值原语，不调用/改写RAM限定optimize_node，不冒标RAM。唯一planner输入是resolver的task.body，evaluator/native7无入口。H1/H3、Dx与原goal/identityQ/normalizer不变。

HOST fresh raw leaf为[5K,2]，连续Kpacket原torch normalize/cat/action_encoder/model.predict/roll；loss从最终LIVE z末帧形成f32 scalar，再autograd.grad(loss,leaf)一次。capture的detach仅写IO，不回流替代LIVE图。保存全部raw gradient、scalar、rawleaf及真实raw backward callback；解析tanh链恰好一次，模型不训练。原生hooks覆盖AE/predictor/proj全部输入/输出、predict显式开始返回、完整packet input与boundary，不改globals或model.predict。vision hooks禁止encoder/projector。所有hooks finally恢复。

## 硬运输证明

每packet实际17份f32数组：packet latent/history/raw/mean/std/token/outgoing7；AE input/output2；predictor input2/output1；projection input/output2；predict output1；boundary latent/history2。max576elements/2304bytes（scalar等更小），预算保守允许32份、每份768elements/3072bytes+4096NPY头。AD每次另存scalar/rawleaf/gradient3份，raw最大30f32；额外384数组已覆盖3nodes×32×3。重复数组不去重。

每格最多3nodes、204Kpacket；arraycap=32×204K+384；eventcap=16×204K+768，每event≤16384bytes+LF。metadata整体16MiB、failure/report/completion reserve64MiB。未压缩NPY/JSONL，无gzip或syncflush隐藏费用。

最坏metadata RAM夹具用3个H3节点、每packet超额33份完整descriptor、全部33候选/32梯度、每候选576维f32最大有限值和完整19968文件的report SHA清单；序列化总量再额外保留1MiB后仍小于16MiB。运行中对实际metadata总字节写前硬守卫；一旦超限只保留partial，不删字段。

H3：143130624 array bytes＋173025600 event bytes＋16777216 metadata＋67108864 reserve=**400042304bytes <536870912**。这是完整三数值节点包络；当前只运行一个节点，不能把余量称encoder/physics已付。真实native list已具体限定，超过数组大小/次数/字段名单即拒绝，不删IO扩cap。每job1800s、floor200MiB、GPU allocated3GiB/reserved4GiB；峰值含loader从进程CUDA原始peak不reset，K3图真实峰值只能MAIN运行后保存，不伪称静态证明fit。wall从verify前至report fsync/hash并独立completion。逾限/错误保留partial与原异常repr/message/traceback，不签完整成功。

## 权威与完整saved重算

绑定actual resolver prepare b44699e6...、原shared prepare7ca85050...，只读小prepare/模型source，不重读shared132MiB report或回调fullstrict。MAIN完整上游receipt为继承，full_upstream_rebuilt_here=False。prepare重新hash原shared封存Python源、该driver本地依赖与8个installed内建类模块源、原config/weights，运行逐named-module验证实际loaded classes全部在该闭包内；不声称重hash了Python解释器或所有stdlib/二进制运行时。原loader单buffer权重反序列化、严格remap、模型eval/全参数不训练，before/after tensor hash必须原identity。记录MATH-only/TF32off/autocastoff实际before/after及allocator峰值。

每个真实packet运行中也从已落盘数组按f32字节验证完整native连接和context（包含正负零），再派发下一packet。自有consumer不用rehearsal Tape/read_tape：bounded regular read、整个directory all-and-only census、每个artifact SHA、NPY读到EOF拒尾字节、完整实际数组引用名单、原source重hash。用SavedBackend完整重放33候选/32AD/3replay及所有原生IO/AD/selection chronology，不只比event总数。独立f64 square sum与标准gamma192 f32误差界核scalar loss，绝不要求NumPy与CUDA reduction bitparity；原candidate/replay exact gate不放宽。27轴从实际start/return事件重建，失败不退款，engine-backward/FLOPs未知不填0。

## MAIN命令（尚未生产，先审读）

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。所有文件参数canonical绝对路径。

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_e3_gd_host_cuda_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_e3_gd_host_cuda_20261029_v1 budget
.venv/bin/python -B -m experiments.spectral_reachability.agent_e3_gd_host_cuda_20261029_v1 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_gd_host_cuda_20261029_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_e3_gd_host_cuda_20261029_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_gd_host_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_SHA
.venv/bin/python -B -m experiments.spectral_reachability.agent_e3_gd_host_cuda_20261029_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_gd_host_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_SHA --cell 0 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_gd_host_cuda_20261029_v1_cell00_MAIN01 --authorization MAIN_E3_KNOWN16_CONTROL5_GD_HOSTCUDA_ONLY
.venv/bin/python -B -m experiments.spectral_reachability.agent_e3_gd_host_cuda_20261029_v1 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_gd_host_cuda_20261029_v1_cell00_MAIN01 --report-sha ACTUAL_REPORT_SHA --completion-sha ACTUAL_COMPLETION_SHA --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_gd_host_cuda_20261029_v1_consume_cell00_MAIN01.json
```

固定全部cell0..15，每格独立job目录，不按效果漏格；其他格只替换literal cell与独占目录名。MAIN选择何时授权生产；worker不调用这些生产命令。当前尚无真实HOST运行结论。
