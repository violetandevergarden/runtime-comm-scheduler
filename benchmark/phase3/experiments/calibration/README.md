# 通信尺度校准

`multi-scale.json` 用于 `run_comm_profile` 的 4 KiB、1 MiB、16 MiB 严格 profile。
`two-groups.json` 用同一输入分别运行 Phase 1 bare 和 `max_outstanding=1` 路径，
用于记录本机裸并发/串行差异；这个差异不写回 profile p50。

推荐 profile 参数为 `--warmup 5 --iterations 30`。每次 replay 仍使用自己的
`--warmup-iterations`，profile 进程的预热不能替代 replay 内预热。
