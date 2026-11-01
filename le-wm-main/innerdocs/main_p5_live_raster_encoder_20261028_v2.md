# 新 live-raster CUDA encoder v2

明确接入独立 saved consumer v2，不修改任何封存renderer/run或旧encoder v1。继承v1的固定10图/16score、full256×384 tokens、HOST CUDA-only和原kernel；run_cuda/consume/Meter/Store/encode/scores/load_runtime/source_closure AST与原冻结kernel相同。

prepare显式source SHA绑定新v2消费者，并重新完整consume已有605步/85图报告。v2 receipt须有全source_seals、5项baseline_to_current_anchor及65项source_comparisons，四word跨时点RNG必须由一次原constructor randint全状态解释；颜色各shot自身before/after全状态严格相同。输入使用实际color after digest，不冒称baseline/global RNG或历史full_public相等，不删除原diff。旧renderer_release=false/historical null保持。

七显式render authority参数沿用v1 CLI，生产须用实际prepare/report/receipt/source SHA；无成功receipt就不能编码。所有10token先落盘再算16score，无挑词/掩码/训练/重试/CPU神经fallback。32MiB/200MiBfloor/600s和模型/source前后验证保持。上游physics605与新encoder10分账，旧coverage只并栏，不用于选择。

RAM测试额外拒绝旧时间接线receipt、消费者source漂移和缺baseline对照；不充当GPU或机制证据。prepare→verify→主机cuda --confirm-host-cuda→consume，模型结果仍须MAIN实际执行。不得将此单known构造根的颜色敏感性诊断称P1/C2/任务收益。

MAIN首轮prepare发现upstream snapshot digest的compact JSON/no-newline与本encoder manifest digest域不同，属于接线错误；尚无manifest或CUDA输出。已定向使用renderer.digest核其自身快照hash，新增精确domain/变异回归，未重跑physics、不改任何上游字节或阈值。下游自己的manifest/token/journal digest保持原kernel ABI。
