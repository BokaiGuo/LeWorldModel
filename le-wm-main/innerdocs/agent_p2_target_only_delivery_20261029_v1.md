# P2 known2 target-only 离线装配

唯一新stem `agent_p2_target_only_delivery_20261029_v1`，source/tests位于同名惯用目录；不改 tracker、三程序或 prior 的任何冻结文件。使用 experiment-code 的日志/回归要求；本任务仅 saved ABI 和 RAM arithmetic，不训练、绘图、加载模型、NN、physics 或运行 full strict。

## 固定交付边界

两根 known/off 各保留 prior FIRST、program 首 postcondition、arbitrary missing，共六槽：四槽 assembled_not_dispatched，两槽 missing。原 high 四分支中的 `/rows/0、2` 固定取off，不按 outcome 选；原program四plans固定 `/plans/1、3`。新artifact不比较结果，不宣称新根、同生成预算、raw可执行性或科学P2。

实际prior为 float32 `[1,192]`，严格等于原selected high rollout `[1,1,2,192]` 的 `[:,0,0,:]`；每根与原receipt high_target/low_goal双SHA一致。原high `z_init` 精确等于 parity history3 的末帧，`z_goal` 精确等于原goal编码；四张原RGB的SHA与blind相等。program task/blind、history3/action2/goal也必须逐位一致；首edge输入必须等于原root context。其输出末latent只增加batch轴得到 `[1,192]`，不做映射、refit或新编码。

`controller_payload` 严格只有三键：`z:[1,3,192]`、`a:[1,2,10]`、`target:[1,192]`，JSON数值由原float32精确往返。`a`是两条**过去真实causal action history**，不是未来witness。拒绝任何额外键（commands/witness/init_action等）；不向controller送program commands/logits、CEM宏、selection信息或raw future。metadata的origin只提供原authority pointer与target SHA；controller若以后运行，只取payload，不取整份artifact。

此工具没有run入口、不调controller。将来三臂由MAIN复用冻结 `agent_hilewm_history3_target_tracker_20261029_v1.py::track_cuda`，caller须经 `validate_payload` 白名单核对。共同controller spec不由delivery自授权，既有known2 smoke成功不等于本四target已追踪。零初始化/no-witness规则保留，不拿动作见证warmstart。

prior FIRST 的物理时间语义仍为null，不能当成物理cp5；program首target来自原raw第5步，而共同H3 tracker的terminal是另一个问题，artifact显式 `controller_H3_is_not_source_cp5=true`。本slice只证明target交付与共同输入配对，不宣称同时间三目标比较。arbitrary没有真实直接latent optimizer输出，保null，禁止用goal/random vector填。多程序后续接缝由Nash负责，本工具不实现或替代。

## 实际authority与来源范围

source的 `PINS` 固定11份完整小JSON，不是未来SHA；`authorities` CLI打印全部绝对路径与SHA。核心列表如下（路径均在 `/home/gbk/code/LeWorldModel/le-wm-main/results/`）：

| 来源 | 相对路径 | 实际SHA256 |
|---|---|---|
| parity prepare | agent_p2_hilewm_computation_parity_20261029_v1_prepare_MAIN01.json | 796c80ca14574b29b9920b0ef53b02cc6c3b3b47842708951e5bc0b17aadf4e1 |
| parity report | agent_p2_hilewm_computation_parity_20261029_v1_cuda_MAIN01/report.json | 12fe43cc4a796fffc18b98fcc867283f8534c22110d794729652ac112aedead8 |
| parity strict | agent_p2_hilewm_computation_parity_20261029_v1_consume_MAIN01.json | 3555696c02329062084d791c0c40958084d7352e6f7680938bade0068afe9480 |
| prior prepare | agent_hilewm_rgb_high_low_20261028_v1_prepare_MAIN01/manifest.json | a3255d38e9cf718f9560c1b9e381abe38750edd9e91d3f04e5ad332cf414d7e9 |
| prior report | agent_hilewm_rgb_high_low_20261028_v1_cuda_MAIN01/report.json | 083d0d9aa190b561e6ccf865b09d7224b97d404b16ba38c792bd8af386f5622a |
| prior strict | agent_hilewm_rgb_high_low_20261028_v1_consume_MAIN01.json | 40f28414682c9540e9e195b361a36dfc514f816358adbeacb379fedc460e4ef0 |
| program report | agent_p2_three_program_20261028_v1_cuda_MAIN01/report.json | 56e8b4f2df522f7a0c4cb9bc37ad74bcbc5d4f89e2c5cd601d42cd0eb071d8ef |
| program strict | agent_p2_three_program_20261028_v1_consume_MAIN01.json | c491555fda5bb543af69f70987541edb0abb99fa32c559f3aa6190901393017f |
| tracker prepare | agent_p2_history3_tracker_cuda_20261029_v1_prepare_MAIN01.json | 161af4ff5b70a656ad147f2775d169fb2087aa72a8e07e9b6e40aabce0ae5456 |
| tracker report | agent_p2_history3_tracker_cuda_20261029_v1_cuda_MAIN01/report.json | ab89ea3c8c7198ce001280a93c73c312c5454f552b4452232e04a6776adf6884 |
| tracker strict | agent_p2_history3_tracker_cuda_20261029_v1_consume_MAIN01.json | 080fe1173895621ea4c0565671509a267ba456135fedfac313ddac7894f98138 |

prepare读取完整固定字节、核report→prepare、receipt→report、tracker→parity整个authority与完整upstream receipt。保存完整parity/prior/tracker receipts（program只引用完整原receipt文件及SHA，防止把commands复制进新artifact）；不只读status，也不自造receipt。调用原 `parity.load_array` 对选定小NPY核header/shape/dtype/EOF/finite/SHA，再以独立字节比较核target来源与context配对。源码闭包复用 `tracker.closure()`，只读源码、metadata；不调用 `tracker.verify/consume` 或任何full strict，不遍历weights。

这属于**导入MAIN已完成strict并做所选数组算术join**，不是worker独立重建完整strict/梯度/模型。完整新artifact≤16MiB、floor200MiB、排他写/fsync文件+目录；verify完整重建artifact、读取所选数组并复核sources，不触发NN。未知原scaler继续null，无rawinverse。新模型/AD/physics/fullstrict调用全0；旧proposal预算不重执行也不抹掉，生成成本比较null，不把smoke controller预算称三proposal共同总预算。

## MAIN准确命令

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q tests/test_agent_p2_target_only_delivery_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_target_only_delivery_20261029_v1 authorities
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_target_only_delivery_20261029_v1 prepare \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_target_only_delivery_20261029_v1_prepare_MAIN01.json
# DELIVERY_MH 填上一步实际返回SHA，不填未来值。
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_target_only_delivery_20261029_v1 verify \
  --manifest /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_target_only_delivery_20261029_v1_prepare_MAIN01.json \
  --manifest-sha256 "$DELIVERY_MH"
```

MAIN审读后实际prepare/verify；worker只RAM/保存ABI测试，不生成实际workspace交付artifact。测试含真实11authority固定字节、小NPY join、FIRST/goal/context/来源突变、白名单、完整六分母、arbitrary missing、fixture prepare→verify/排他写、cold CLI无torch。还核所有复用helper与tracker实际prepare的原source闭包逐项精确匹配。仍未运行四target controller或三proposal；无新科学准入。
