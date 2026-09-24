# Binding preparation timing — Phase A

对照路径：`old-ltf`、`new-ltf-on-ready`、`new-ltf-precreate`。输入分别引用
`baseline/L0-balanced.json` 和 `readiness/L1-head-misalignment.json`，profile 使用
`benchmark/phase3/results/calibration/multi-scale/phase3-compact-20260923/profile.json`。

每个场景 5 seeds × 3 repeats；固定 Gloo、world size 2、1 ms completion polling、jitter 0.3、
warmup 1、wait budget 20 ms。每个 `(seed, repeat)` block 内三条路径顺序随机。旧路径也在本批重新运行；
不拿历史 makespan 作直接配对样本。

关注同一 block 的 paired makespan/JCT，同时单独报告 Phase 2 tensor allocation、Phase 3
binding creation 和 pre-release preparation wall time。precreate 结果只支持“已知资源准备移出应用
计时区间后，应用关键路径减少了多少”的结论；不把 preparation 与 makespan 相加当完整程序耗时。

批次位置：`benchmark/phase3/results/runtime-overhead/binding-preparation/<batch>/`。
