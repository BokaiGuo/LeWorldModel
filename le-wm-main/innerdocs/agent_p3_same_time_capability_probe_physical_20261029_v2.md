# Stage C v2：固定五端点 U/Z2 原生延续

本件是可供 MAIN 审读、prepare、原生运行和独立 saved consumer 的新实现，不修改冻结 schema v1、full-witness 或 Stage B。worker 只执行 RAM/临时磁盘测试；尚无本件真实 physics 结果。所有科学、freshness、独立确认、equal-FLOPs 与 physical_release 声明保持 false。MAIN 的显式 run authorization 只授权这次工程运行，不改变上述声明。

## 固定输入与信任边界

`references()` 固定实际路径及以下实际 SHA，不接受未来占位或谱 summary 替代：

| authority | SHA256 |
| --- | --- |
| full witness MAIN01 | 2351ba5b05c91fb192ae4542e6839ab58b56bcd974170b94994246f61981f346 |
| Stage B prepare | 90b3a1a6284c5605594d4341a2f994870e4b8db9c1721ba40f0160244082e870 |
| Stage B report | a64fbf31cf2601cf6ea26d100ef5cb5c01efbe8b63be6cf33bc4d91d735c25fc |
| Stage B completion | 2d4f8364867ccfba4bb15f4794f0f665af34483cf4bdb1adf57af4efb1f27574 |
| Stage B receipt | 682a78366b10a60895c4530f2fc6cc401e6439b625f4ba0a8aa996dc2fed79a7 |
| U witness0 | 6029b37c7a02543a58ef962c4369002939239d3f34bda34bced155204eb3124a |
| U witness1 | 011d086c47cfd2ffd33fc16a895960d83e6cb0308c63168ecfd9bfe41abb8b21 |
| U witness2 | 5cd3933ffab7ed42b0124f26414bcba8de3c15075f6fe0d27815e5a4b8c3caa3 |
| U witness3 | e1e2bf9124070256da21b064e2c1316045c9b3dc22190c665f44b9c9a3146422 |
| U witness4 | ebf2b284e92c80905c2da561fc7b69f669d4cfd69bc5cd6f213ef34627ab7c55 |

根目录为 `/home/gbk/code/LeWorldModel/le-wm-main/results/`。Stage B stem 为 `agent_p3_same_time_capability_probe_cuda_20261029_v1`，prepare/receipt 分别为 `_prepare_MAIN01.json`/`_consume_MAIN01.json`，report/completion 在 `_cuda_MAIN01/`，witness 为 `_witness0_MAIN01.json` 至 `_witness4_MAIN01.json`。full witness 为 `agent_p3_actual_cp10_full_witness_20261029_v1_MAIN01.json`。

`material()` 调用冻结 full-witness 的局部 saved `verify`，不调用 Stage A/B 或原物理 fullstrict；继承 MAIN 完整认证并重建本件连接。`assemble()` 精确连接 B report/receipt/completion、全部五 plan/export、task/source/causal_raw10、whole/segmented、原 float32 raw 位及 native seal/call ids。full witness 内部 `stageB_ready=false` 原样保留：新 adapter 独立认证两个权限域，不改旧 flag。7J/1344VJP/24predict/7solve/5U/10replay/22Q 的完整原费用作为 inherited receipt 保留，本件新神经调用为零。

## 原生执行与两个硬门

顺序固定 `(r0,C),(r0,Z),(r0,P),(r1,Z),(r1,P)`，每端点先 U 后 Z2，共十个独立 env。U 仅接受已认证双重放 raw5；Z2 为正 float32 零五步。不重选、不 clip/tanh、不重新求解。

每 job 使用原 constructor、seed reset 和原 placement reset 输入，随后原 warm10、control10、continuation5。第二次 reset 的 state 是原 `requested_state`，不是 cp10 state；不 hydrate，不调用额外状态恢复，不重采 placement。原生 reset 内部 setter/physics 调用照实记账。`native_helpers()` 绑定原源码并检查 PushT constructor AST 在 hooks 前不调用 reset/step/render/setup/setter。

复用 `p1_moving_overlay_recompile_physical_20261027_v1.native_hooks/capture/validate_env`，以及冻结 moving `attach_goal/snapshot` 和 paired `check_return`。仅 MAIN 授权后 import 原生依赖。原 goal RGB 来自原 blind NPZ（`allow_pickle=False`）；shape/dtype/file SHA/pixel SHA 全核，不重编码。

`Meter.call()` 在 plan control6 dispatch 前重建 prefix51，在 control11 前重建完整66；guard 必须成功写入 journal 并 fsync 后才释放动作。检查原 pre/post/snapshot/receipt 完整原字段、原 raw/actions/实际 latest_action、reset/placement、warm0 和 actual_response10。仅以下运输差异允许局部映射：`index, previous_sha256, event_sha256, branch, arm, selection_digest, receipt/png`。全部原值保存在 proof，不放宽任意字段忽略。root/condition/kind 等不映射。原 snapshot 未记录的 RNG/public_state/contact-cache 语义保留 unknown，不伪填，也不声称动力学缓存恢复。

## 整数预算和失败运输

每 job 上限固定 64MiB（含 8MiB final reserve），磁盘 floor 200MiB，whole wall 900 秒。十个 job 顺序运行，各自计时；不是全批 900 秒。全批最大十份 64MiB，加单独小型 batch 清单；不得默默扩 cap。

完整 job：constructor1、close1、reset2、set_state4、goal_render2、goal_bind1、goal_setter3、public_step25、physics_step254、explicit_render81，共374 operations。748 intent/terminal 事件 +81 observation +2 guard +3 lifecycle =834 events。81 PNG 为 warm0..10 与 plan0..15，每位置 off/on/off 三张；每 job 均全保留，不跨 job 去重。十 job 上限810 PNG/250 public/2540 physics。

binding2MiB、journal20MiB、report14MiB、completion1MiB、failure1MiB；PNG 使用224×224 RGB无损 PNG 实际字节上界。`census()` 用完整真实 source 事件结构推导 observation 上界，不造 RNG 字段。prepare 如算得超额直接拒绝；运行期间仍执行单文件/总字节/事件上界。目录和文件排他创建、fsync；禁止覆盖失败结果。

PNG 上界修正（不扩64MiB）：不使用默认压缩参数才适用的 tight compressBound。RGB8 非交错图每行增加一个 filter byte，`n=224*(224*3+1)=150752`。保守 generic deflateBound 为 `D=n+ceil(n/8)+ceil(n/64)+11=171963`，包括 zlib 包装。固定 PNG signature8、IHDR25、IEND12 共45字节；每个 IDAT 含 length/type/CRC 共12字节。安装 Pillow `PngImagePlugin._save` 将无metadata的新 `Image.fromarray` 交给 `_idat`，每次 encoder write 一块；`ImageFile._save/_encode_tile` 的 buffer 为 `max(MAXBLOCK=65536,width*4)=65536`。保守允许 `ceil(D/65536)+1=4` 块（额外一块容纳可能的最终空/短 write），总上界 `D+45+4*12=172056` 字节。这里不是将“块最大65536”偷换为块数证明：`check_png_stream()` 对每份实际 writer 输出显式限制只有 IHDR/IDAT/IEND、IDAT长度/总长度/块数，超出就拒绝，不运输超界结果。consumer 同样独立检查该结构，再由 PIL 解码与像素SHA核验。未更改 `.save(format='PNG')` 参数或压缩算法，因此源 byte-gate 的 writer 保持原样。81张上界13,936,536字节，加38MiB元数据quota与8MiB reserve，合计62,171,032字节，小于67,108,864。测试实际写出81张固定seed随机RGB不可压缩图，逐张SHA/解码/块结构核验，并覆盖旧tight bound不足和第82张拒绝；不以压缩良好的常量图替代此预算测试。

原 terminated/truncated 保留，early-stop 不补动作/帧，cp15 缺失保持 null。dispatch intent 已付费，不退款。`failed_publication()` 使用 final reserve 保存 partial 文件 SHA、已声明 ledger 及 unknown_final_callbacks；`consume_failure()` 仅认证失败运输字节，counts=null，不能把未知 callback 当零或成功。普通 engineering_failed 与 transport_failed 分开，固定十槽不删失败。

## 本件 consumer

`reconstruct()` 从 journal 全字段、事件 hashchain、嵌套 operation、动作顺序与 return、guard、源连接独立重建行。`consume_job()` 重算响应、checkpoints、资源、null、81 PNG 解码/file+pixel SHA、inventory 与所有 report 字段；不调用 producer fullstrict。`response()` 保留原 native7 reward/success/angle 规则，并独立列 wrapped angle、xy/velocity；不把 wrapped 替换原成功门。外部 MAIN `batch_sha` 必须显式提供，整个十 job 和 source closure 再验。`all10_saved_native_jobs_reconstructed_NOT_admitted` 表示已重建十槽，不保证每槽完成或科学成功；须读各 row status/missing/guards。

## MAIN 精确命令

以下均从 `/home/gbk/code/LeWorldModel/le-wm-main` 执行。MAIN 自行把实际打印 SHA 赋值，不能把示例变量当证据。worker 未执行这些生产命令。

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p3_same_time_capability_probe_physical_20261029_v2.py

.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_same_time_capability_probe_physical_20261029_v2 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_capability_probe_physical_20261029_v2_prepare_MAIN01.json

.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_same_time_capability_probe_physical_20261029_v2 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_capability_probe_physical_20261029_v2_prepare_MAIN01.json --prepare-sha "$P3_C_PREPARE_SHA"

.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_same_time_capability_probe_physical_20261029_v2 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_capability_probe_physical_20261029_v2_prepare_MAIN01.json --prepare-sha "$P3_C_PREPARE_SHA" --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_capability_probe_physical_20261029_v2_MAIN01 --authorization MAIN_P3_SAME_TIME_10_NATIVE_PROBES_V2

sha256sum /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_capability_probe_physical_20261029_v2_MAIN01/batch.json

.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_same_time_capability_probe_physical_20261029_v2 consume --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_capability_probe_physical_20261029_v2_prepare_MAIN01.json --prepare-sha "$P3_C_PREPARE_SHA" --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_capability_probe_physical_20261029_v2_MAIN01 --batch-sha "$P3_C_BATCH_SHA" --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_capability_probe_physical_20261029_v2_consume_MAIN01.json
```

## 回归与未完成项

RAM callbacks 只模拟原事件 ABI，不认证实际 env 等值。临时盘完整覆盖十个834事件/81 PNG job 的 publish→consume、source mismatch、native early-stop、真实文件篡改、partial reserve、accepted guard 写入失败前禁止 dispatch、900秒边界和 constructor AST。fixture_only 默认不能进入生产消费。

剩余验证全部属于 MAIN：独立审读/测试、实际 prepare/verify、原生运行是否精确通过51/66门、实际磁盘/墙钟预算与全十槽 consume。现无 Stage A/B 资产缺失；任何真实门失败保留输出，不改阈值、字段白名单或预算以追求通过。
