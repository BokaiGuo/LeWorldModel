# P2 H3 重算动作的八槽物理探索

四根已知根、每根 joint/suffix，固定八槽。原 H3 producer 未保存 raw 位；本次在 RTX 主机 CUDA 用保存 logits 的 float32 `torch.tanh` 重新计算动作，不运行神经模型。三个 packet 的新归一化 token 均等于对应保存 context；这不证明新 raw 位等于先前原 raw。chart report SHA `840c66efdcfe752f4177138cae9d26bb9e25cb3ccccf2136677d73aaab6c0a30`。

chart 父进程在 CUDA 初始化前锁住 `/home/gbk/code/LeWorldModel/le-wm-main/results/e3_p2_host_cuda_cooperative.lock`，GPU 子进程退出后释放；没有删除或替换锁文件。协作锁不约束未接入入口。chart 完成后本次不再启动 GPU。

真实物理输出 `results/codex_p2_h3_rechart_physical_20261030_v1_RUN01`，report SHA `858f76701f8dcff60bb52a764e916685ba8e3919f3dd25032d5138e6ba7e57c6`；八槽全部15步完成、无早停、各端点 RGB 新鲜，原生 success 全为 false。root collection public step 0；warmup 120/120、selected 120/120 public step attempts/returns，reset 共40/40。内部 simulator 子步、渲染及历史 acquisition 的全成本未对齐，不能称总预算相等。运行时验证历史 root/goal、RGB、warmup 及配对初始状态；未训练、未运行神经网络。所有旧产物保留。

独立 saved 入口重读 report、binding、逐分支 journal/RGB 和 physical budget 事件，重建 attempt/return 及逐动作计划前缀，并检查无终止后动作；receipt `results/codex_p2_h3_physical_saved_20261030_v1_RECEIPT01.json` SHA `79fca5e7ac8eb4b9b0a32da8707ba81fbb5e519f06d2acb1f1ba9fb2e201ac39`。该检查复用原 physical branch 校验器，不是独立重新运行物理或证明所有模拟器内部成本。

新增未来 producer `codex_p2_symmetric_h3_rawproducer_20261030_v1.py` 使用独立 `RawCaptureBackend` 在每个 transition 前将实际 CUDA raw packet 字节送入耐久事件 sink；同样使用公共锁直到 CUDA 子进程退出。此新 producer 未实际执行，其 capture 机制和未来完整保存消费仍待真实 CUDA 验证；不对本次旧原始位倒签。

测试：物理/重算/未来 capture/旧早停回归共 `30 passed`；新 saved budget 篡改测试 `1 passed`。数值仅已知根探索，非预注册、独立确认或 P2 通过。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_p2_h3_rechart_physical_20261030_v1.py tests/test_codex_p2_h3_rechart_cuda_20261030_v1.py tests/test_codex_p2_h3_rawcapture_backend_20261030_v1.py tests/test_p3_matched_physical_20261027_v1.py tests/test_codex_p2_h3_physical_saved_20261030_v1.py
```
