# Runtime overhead experiments

本组实验只覆盖两 rank CPU/Gloo、单个全局在途 collective。按顺序执行，不并发启动 replay。
L0/L1 输入继续引用现有 `baseline/L0-balanced.json`、`readiness/L1-head-misalignment.json`，
不复制 JSON；固定 profile 使用当前仓库中已保存的 compact CPU/Gloo profile。

运行脚本是 `examples.jobpacer.scripts.run_experiments`。每个 workload 独立建 batch，
`--seeds` 表示 epoch/扰动 seed；arm 顺序在每个 seed/repeat block 内随机化。正式运行时固定
线程环境为 1、`--compute-jitter 0.3`、`--wait-budget-s 0.02`、warmup 1、每个 seed 3 repeats。
每批 manifest 保存源码摘要、输入/profile 摘要、CPU affinity 和线程环境。

- `binding-preparation/`：阶段 A，比较 Phase 2 LTF、Phase 3 LTF/on-ready 和 LTF/precreate，
  L0/L1 各 5 seeds × 3 repeats。输入和运行产物分别见 results 下的 `20260924-phase-A-L0/L1`。
- `completion-polling/`：阶段 B，在 Phase 3 LTF/precreate 下交错比较 1 ms 与 0.2 ms，
  L0/L1 各 5 seeds × 3 repeats。除 JCT/makespan 外检查进程 CPU 时间、wait 和 probe 计数；产物见
  `20260924-phase-B-L0/L1`。
- `communication-chain/`：阶段 C 的九种输入已建立，按 4 KiB/1 MiB/16 MiB × 1/8/32 次
  collective 交叉。正式结果位于 `results/runtime-overhead/communication-chain/20260924-phase-C-main/`；
  每种配置有独立 batch 子目录。
- 阶段 D 在 precreate、1 ms 轮询下复测 L0/L1/L5，结果分别保存在 baseline/readiness 场景的
  `20260924-runtime-overhead-D-main`。

正式 batch 写入 `benchmark/phase3/results/runtime-overhead/` 新路径，不复用历史 compact 结果。
旧路径和 precreate 的结果统一报告应用 release 到应用结束的 makespan；额外同时保留准备区间、通信 drain
和 validation 边界。各 rank 时钟只在本地使用，中央 grant/完成间隔仅使用 coordinator 时钟。
固定输入的阶段 C 使用一个 seed 和 5 repeats，不据此生成 seed-bootstrap 区间；其用途是诊断成本随
collective 数量和 payload 大小的变化。阶段 A/B/D 的正式统计按 seed 内 repeat 中位数配对，并保留全部
有效运行。
