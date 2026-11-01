# Hi-LeWM有界raw-DEFLATE数据解码器 v1

独占source/tests/本文；MAIN审计后才可实际decode。worker只运行RAM压缩字节fixture、已落地metadata/text读取，不解码真实113MB流，不反序列化、不执行包内/外部代码，不联网、不运行神经/物理。采用experiment-code技能的固定输入/记录/可复现要求，不采用其训练、挑最好结果或修改零结果规则。解码不是E1复现、trusted-loader认证或P1准入。

## 固定字节与闭包

输入固定为 `/home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_asset_resume_20261028_v1/continuation02/pusht_hi_lewm_epoch15_object.ckpt.deflate`。113104429 bytes，SHA256 `d169cf3c4a8d668ac5b0dcb17c6fd1682360764ce578d38daac7fec6dc76d086`。这是ZIP method8的raw-DEFLATE，非ZIP文件/可加载checkpoint。目标成员为`package/checkpoints/pusht/main/pusht_hi_lewm_epoch15_object.ckpt`，原归档范围634194454–747298882。解码必须完整EOF，无unused_data/unconsumed_tail/trailing/拼接第二流，恰好122424517 bytes、CRC32 `d59f865b`；实际decoded SHA现场计算、fsync后重新读回校验，不预填/冒充作者SHA。

原request/report/saved_verification/handoff的四SHA硬绑定；还绑定原source contract `c25edf542f38faa9cf0150df4822198d28753e2cf77a0d243e990ffabe849ded`、Zenodo API `2fc221128c12b609b86dfb7810eda46f2399bc796348f36c5d7e016ab25dd773` 和下列文本。源码/测试/doc SHA在实际decode前后保持相同并进入报告。CLI无input/SHA/CRC/大小覆盖参数。下载曾因相同重复Content-Type被解析器拒绝的原收据仍保留，不因后续完成抹去。

压缩流先全量hash，未通过时不解码/不建输出目录；同一fd rewind后解码，解码时再累计compressed SHA。输入同fd/path的dev/ino/size/mtime/ctime/nlink前后精确一致。全部路径祖先通过O_DIRECTORY/O_NOFOLLOW逐段打开，拒绝symlink；只读常规文件，不解pickle、YAML或对象内部容器。作者只公布整包MD5，未下载整包，故不声称整包校验通过；Range无强ETag/未回显Last-Modified的限制不因CRC通过消失。

## 防炸弹、资源与失败保全

每次输入<=128KiB，decompress(max_length)<=64KiB，无不受限flush；多出哪怕1 decoded byte也在写入前拒绝。非EOF截断、第二流、错误压缩SHA、输出长度/CRC及short-write失败均非0。墙钟默认120s，只可设置(0,600]，逐块合作检查；不是对阻塞磁盘/调度暂停的硬实时保证。

所有新输出目录必须位于inner `results/`，名字为本stem加`_MAIN01`等后缀；mkdir排他、固定`decoded.unverified.ckpt`和`report.json`均O_EXCL，绝不覆盖或删除。失败已写字节与rejected报告原样留存；不重试、不rename为“可信模型”。若建目录/读前检查失败则抛错而不创建报告；若磁盘已跌破报告保留量，报告写入可能也拒绝，已写字节仍保留，MAIN须保留stderr/非0退出。不要仅凭扩展名使用文件。

3GiB预算计入原asset_resume整个树（包括chunks、assembled副本、失败收据）+全部本decoder stem输出树+计划解码及2MiB报告保留；硬链接按每路径重复计账，symlink/特殊文件拒绝。原旧下载partial未复制，不计作新增字节。实际写入每块前检查合并预算和>=10GiB剩余空间（另保留当前块及报告）；首次保留完整122424517+2MiB。不删除任何旧chunk/partial。并发外部写入不是本工具可独占的全盘事务；MAIN需串行运行本stem并保管输入不可变。

## 只读license/config/可信loader定位

`plan`给绝对路径、原member、SHA、行号与原文片段；它不打开deflate/ckpt。全部文本在 `/home/gbk/code/LeWorldModel/le-wm-main/results/p1_hilewm_text_contract_20261027_v1/texts/`，文件名将member的`/`换成`__`并加`.source.txt`，逐文件SHA沿固定contract真实校验，非凭文件名猜来源。

|已存来源|行号/字段|含义与未决项|
|---|---|---|
|Zenodo缓存`results/p1_learned_subgoal_remote_inventory_20261027_v1/zenodo_api.json`|`metadata.license.id=cc-by-4.0`|仅record层许可；原inventory未发现独立LICENSE/COPYING不等于第三方许可齐备|
|`package/checkpoints/pusht/main/config.yaml`|1–117|img224/patch14、history3/embed192、high latent_action_dim32、state_dim7、pretrained低层路径、freeze开关、pusht_expert_train/frameskip5；只证文本，不证对象实值。seed3072是训练配置，不是本地评估seed证书|
|`package/code/THIRD_PARTY_LEWM.md`|19–39|官方要求外部LeWM checkout，仍为`<PINNED_COMMIT>`占位；需MAIN认证revision、许可及逐依赖hash|
|`package/code/h_le_wm/eval/hierarchical.py`|34–38、514–533|`_baseline_lewm_module`动态pickle class兼容、high/low同device、AutoCostModel；无可信完整class manifest，不执行原loader|
|同上|318–330|StandardScaler.fit现场数据统计，当前禁止原样运行；需原dataset identity与已冻结normalizer身份，不可偷偷fit|
|`package/code/h_le_wm/baseline/adapter.py`|11–16、39–63|依赖外部module.py/utils.py并exec_module；文本静态检查不认证反序列化安全|
|`package/code/environment-gpu.yml`|1–11|Python3.10/CUDA12.1/cuDNN8.9，但stable-worldmodel未精确锁版；不能据此声称现环境兼容|
|`package/code/h_le_wm/checkpoint_registry.yaml`|14–18|canonical default_epoch15路径与本次main成员名不同，身份等价unknown；不能改名/软链接冒充已闭合|

MAIN后续仍需认可来源/许可、可信类与依赖闭包、embedded low逐tensor/config配对、learned high/macro encoder对象结构、原dataset/normalizer/指定hierarchical协议；如-C另需empirical bank身份。冻结低层文本不等于整个Hi-LeWM不训练；这里不训练任何模型。输出`baseline_runnable=null, official_equivalence=false, scientific_admission=false`。

## MAIN命令（inner项目根）

```bash
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_bounded_deflate_decode_20261028_v1 plan
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_hilewm_bounded_deflate_decode_20261028_v1.py
# 仅MAIN审计后实际decode；worker未执行此条
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_bounded_deflate_decode_20261028_v1 decode --output-dir /home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_bounded_deflate_decode_20261028_v1_MAIN01 --wall-seconds 120
# MAIN取实际report.json SHA，绝不以未来/猜测SHA代替
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_bounded_deflate_decode_20261028_v1 consume --report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_bounded_deflate_decode_20261028_v1_MAIN01/report.json --sha256 ACTUAL_REPORT_SHA256
```

consume只重读原压缩字节SHA和新decoded字节SHA/CRC，不再次解码，不导入torch/pickle/归档代码，不赋予加载授权。无prepare或自动run：固定输入/字节合同已封于源码与已存下载闭包；MAIN审读实际调用decode。测试覆盖RAM正/负字节流、炸弹/尾随/截断、短写、时间/磁盘、排他/symlink/身份；冷子进程验证plan无模型/网络及对象读取。测试通过不是科学证据。实际测试结果及三SHA由最终交付回复给出；本三文件冻结后任何改动须新版本。
