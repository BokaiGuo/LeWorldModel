# P2 H3 物理回放前置

结论：四根历史物理来源可 join，但当前八槽不具备原始 raw float32 动作位证据；本轮没有执行物理，也没有重新运行神经模型。不能把重算的动作称为已保存的原始动作。结果是探索性的，不改变科学门。

使用 v2 原 report、prepare 和 saved numeric checks；对四根精确核对 root、完整 root context、goal192、投影、历史 root/goal RGB SHA，物理采样 seed `202753010`、oversample 4。参考环境为 `swm/PushT-v1`，历史路径每臂 warmup 15 个 public steps；真实 reset 后 RGB 和动作历史仍须运行时验证，不能由离线 join 代替。

源 `codex_p2_symmetric_h3_run_20261030_v2.py` 的 `winner` 仅保存 logits、三个输出 contexts、投影末态和成本；底层 `NativeCudaBackend._terminal` 用 CUDA float32 `torch.tanh` 生成 raw，再经 public mean/std 归一化，却没有保存 raw 本身或 raw bits。保存 normalized token 不足以独立认证实际 raw：例如 root0 joint 第一 packet 的第 7 个标量，float32 数值 `0.8510942459106445` 与相邻 `0.8510943055152893` 均可产生同一已保存 normalized 值 `4.258515357971191`（此为 CPU float32 算术诊断，两个数不被声称是实际 CUDA 输出）。保存 logits 提供可重算线索，但事后重算本身不是当时 raw bytes 的记录。

旧 `codex_p2_joined_native_physical_20261030_v1` 的 planner 和 `PhysicalMeter` 固定 H2/每槽10、selected总80，不能直接用于本 H3 八槽 120 步；不改旧文件或伪装 schema。底层 `p3_matched_physical_20261027_v1.guarded_execute` 接受15步，terminated 或 truncated 后立即停止，保留真实返回次数与剩余步数，不补动作。真正 H3 生产桥还需新建 selected cap120、warmup cap120、reset/renderer/任务和异常清理、每步 durable 原始动作与 RGB/状态、完整八槽 missing/early-stop 独立 saved consumer。早停的 endpoint 不能称完整15步。

新增 `save_cuda_raw_packet` 提供将未来 producer 的 CUDA `[5,2]` float32 tensor 原位读回并在 transition 前交给显式 sink 的接缝；sink 抛错时调用失败。它尚未接入封存 v2 producer，不证明 sink 耐久、真实设备、调用先后或原模型输出，也不能为过去的结果补证。测试中的假 tensor 只验证编码和错误处理，不是 CUDA 证据。

预检输出 `results/codex_p2_h3_physical_preflight_20261030_v1_CPU01.json` SHA `18c58217cc5b062e688a2f3e9fc59442ddebbe0530afb8ba1cf82d45593b83c7`，八槽 raw 保持 null。测试 `10 passed`。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_p2_h3_physical_preflight_20261030_v1.py tests/test_codex_p2_symmetric_h3_saved_checks_20261030_v1.py
```

下一步若需要同源原动作物理回放，应先明确允许一次仅执行 CUDA chart 的动作重建并把其定位为“重算动作的探索性回放”，不能说原 raw 位已被证明；或者在新的未来生产运行前将上述 hook 接入真正 producer 并完整验证。当时 raw 证据缺失无法靠事后补写消除。任何物理执行还需先完成上述 H3 runner/consumer 和资源测试。
