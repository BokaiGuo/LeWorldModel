# P2 前置组合接口：真实主机 CUDA 验证

2026-10-26，主线程执行；不是 P2 科学准入或组合搜索效力验证。

新模块 `pusht_composition_cuda_replay_v2.py` 将显式每环境步 raw commands 按五步 packet 转换为 released normalization，用三帧 latent 和两段帧间 action history 逐段递推；不补造首帧前动作。调用时检查模型冻结、eval、CUDA device。保持同一完整历史，支持 witness 的逐边重放与整段回放。

runner：`le-wm-main/experiments/spectral_reachability/run_pusht_composition_replay_cuda_v2.py`。

最终产物：`le-wm-main/results/pusht_composition_replay4_v4_20261026_cuda/report.json`。退出码 0，completed，6 项完整性 gates 全 true。4 个已知根 ×16 个封存双 token 程序，384 次实际 predict 调用/样本，所有 context length=3。全部分段／整段递推上下文精确一致；与原封存 terminal forecast 在预声明 atol=rtol=1e-5 下匹配。模型参数、缓冲区、eval 与输入绑定不变；运行前 source seal 及结束核验完成。

每个程序保存可复算的组合收据、原始每步命令、两条边及完整上下文哈希。`progress` 是边类型声明，不是“物理进展已证明”；`program_kind_mechanism_verified=false`。provenance 绑定整个 checkpoint（包含 encoder）、来源封存、协议、度量和候选库；不把 checksum 当作物理可达性证明。

## 捕获并修复的代码问题

1. v2 输出目录：归一化统计用 `list(np.float32_array)` 得到 NumPy 标量，JSON 编码失败。改用 `.tolist()` 保留 float32 原值的原生 Python 数值。
2. v3 输出目录：上游 `history_rgb_sha256` 是三帧堆叠整体哈希，不是逐帧哈希列表；旧接线 `[-1]` 误取最后一个字符，被严格 SHA 合同拒绝。改为绑定完整 digest，并明确 `root_rgb_scope=stacked_three_frame_context_not_single_frame`。

这两个失败是新接线错误，不能计入方法负结果。两个失败目录和运行前源码快照保留不覆盖；修复改变了 runner 当前字节，因此失败目录的 current-source 验证不再可通过，历史快照仍可审计。原已完成编译／物理实验源码未被修改。

组合合同、适配器、runner 接线回归共 62 项通过，涵盖这两个 bug；纯数组测试不替代上述 CUDA 结果。

## 仍缺的研究证据

没有新谱前缀生成、组合搜索、真实未来能力改变或新物理执行。当前仅重放既有固定双段程序；P1 未通过，不能把该结果称作 P2 通过。还需真实观测后的重编译、新根非贪心机制、无组合 SRA/flat search 强对照及完整预算配对。
