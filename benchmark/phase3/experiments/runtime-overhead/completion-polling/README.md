# Completion polling sensitivity — Phase B

固定 Phase 3 LTF/precreate，仅交错比较 completion poll interval 1 ms 和 0.2 ms。输入复用
`baseline/L0-balanced.json`、`readiness/L1-head-misalignment.json` 与阶段 A 相同的严格 CPU/Gloo
profile。每个场景 5 seeds × 3 repeats；seed/repeat 配对，运行顺序随机。

同时检查应用 makespan、逐 job JCT、wait duration、completion observation 到应用继续执行的本地间隔、
completion probe 次数、rank 进程 CPU 时间和可用的上下文切换计数。0.2 ms 不会自动变成默认值；
本阶段不改变 `wait_host()` 语义、不启用忙轮询。

批次位置：`benchmark/phase3/results/runtime-overhead/completion-polling/<batch>/`。
